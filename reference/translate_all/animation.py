"""Build a self-contained HTML that replays translation progress.

Each translatable text node starts as the original text and flips to its
translation at the wall-clock time it actually completed during measurement.
QA (parallel burst), Seq2Seq chunked (chunk waves) and Seq2Seq full
(sequential sweep) all use the same {tid, orig, trans, t} timeline format,
so the resulting HTMLs are directly comparable side by side.
"""
import json

_CONTROL_AND_SCRIPT = """
<div id="__anim_ctl" style="position:fixed;top:0;left:0;right:0;z-index:2147483647;
  background:#111;color:#eee;font-family:monospace;font-size:13px;
  padding:8px 12px;display:flex;gap:14px;align-items:center;
  box-shadow:0 1px 6px rgba(0,0,0,.4);">
  <button id="__anim_play" style="background:#2d7;border:0;color:#000;
    font-weight:bold;padding:5px 14px;cursor:pointer;border-radius:3px;
    min-width:80px;">&#9654; Play</button>
  <span id="__anim_clock">0.00 s / 0.00 s</span>
  <span id="__anim_done">0 / 0 translated</span>
  <label>Speed
    <select id="__anim_speed" style="background:#fff;color:#000;border:1px solid #444;
      padding:3px 6px;border-radius:3px;font-weight:bold;">
      <option style="color:#000;background:#fff" value="0.25">&#215;0.25</option>
      <option style="color:#000;background:#fff" value="0.5">&#215;0.5</option>
      <option style="color:#000;background:#fff" value="1" selected>&#215;1</option>
      <option style="color:#000;background:#fff" value="2">&#215;2</option>
      <option style="color:#000;background:#fff" value="5">&#215;5</option>
      <option style="color:#000;background:#fff" value="10">&#215;10</option>
    </select>
  </label>
  <span style="opacity:.6;margin-left:auto">__TITLE__</span>
</div>
<div id="__anim_bar" style="position:fixed;top:42px;left:0;right:0;height:8px;
  background:#222;cursor:pointer;z-index:2147483647;
  box-shadow:0 1px 4px rgba(0,0,0,.4);">
  <div id="__anim_bar_fill" style="height:100%;width:0;background:#2d7;
    transition:width 0.05s linear;"></div>
  <div id="__anim_bar_handle" style="position:absolute;top:-3px;width:4px;height:14px;
    background:#fff;border-radius:1px;left:0;pointer-events:none;
    transition:left 0.05s linear;"></div>
</div>
<div style="height:52px"></div>
<script>
(function(){
  // Single-rAF playback. simT = simulated translation time (s, matches e.t).
  // Within one rAF frame, all events with e.t <= simT come due; we DISPATCH
  // them on a stagger so a batch (e.g. one QA block of 128 questions sharing
  // a single block_finish timestamp) cascades visually rather than flipping
  // in a single frame. Stagger is rendering-only -- measured `e.t` values are
  // untouched -- and respects the playback speed.
  var TL = __TIMELINE__;
  TL.sort(function(a,b){ return a.t - b.t; });
  var els = {};
  TL.forEach(function(e){ els[e.tid] = document.querySelector('[data-tid="'+e.tid+'"]'); });
  var maxT = TL.length ? TL[TL.length-1].t : 0;
  var clock = document.getElementById('__anim_clock');
  var done  = document.getElementById('__anim_done');
  var bar   = document.getElementById('__anim_bar');
  var fill  = document.getElementById('__anim_bar_fill');
  var handle= document.getElementById('__anim_bar_handle');
  var btn   = document.getElementById('__anim_play');

  // stagger tuning: per-item delay capped so a big batch (~128) finishes its
  // visual wave within ~STAGGER_BATCH_CAP_MS. Both scale with playback speed.
  var STAGGER_BASE_MS = 25;
  var STAGGER_BATCH_CAP_MS = 1000;

  var raf = null, t0 = 0, speed = 1, playing = false;
  var committed = 0;     // # of events visually flipped to translated
  var scheduled = 0;     // # of events already dispatched to a stagger timer
  var staggerTimers = [];

  function simT(){ return (performance.now() - t0) * speed / 1000; }
  function fmt(t){ return t.toFixed(2) + ' s'; }
  function setClock(t){ clock.textContent = fmt(t) + ' / ' + fmt(maxT) + '  (×' + speed + ')'; }
  function setDone(){ done.textContent = committed + ' / ' + TL.length + ' translated'; }
  function setBar(t){
    var p = (maxT > 0) ? Math.min(100, Math.max(0, t/maxT*100)) : 0;
    fill.style.width = p + '%';
    handle.style.left = p + '%';
  }
  function setBtn(){
    btn.innerHTML = playing ? '&#10074;&#10074; Pause' : '&#9654; Play';
  }
  function clearStagger(){
    staggerTimers.forEach(clearTimeout); staggerTimers = [];
  }

  function setVisible(n){
    // Make the first n events (sorted by e.t) translated, the rest original.
    for(var k = 0; k < TL.length; k++){
      var e = TL[k];
      if(els[e.tid]) els[e.tid].textContent = (k < n) ? e.trans : e.orig;
    }
    committed = n;
    scheduled = n;
    setDone();
  }

  function applyState(t){
    // Force the visible state to match simT == t (used by seek + reset).
    clearStagger();
    var n = 0;
    while(n < TL.length && TL[n].t < t) n++;
    setVisible(n);
    setClock(t); setBar(t);
  }
  function seek(t){
    t = Math.max(0, Math.min(maxT, t));
    applyState(t);
    t0 = performance.now() - (t / speed) * 1000;
  }
  function stop(){
    if(raf){ cancelAnimationFrame(raf); raf = null; }
    clearStagger();
    scheduled = committed;   // anything beyond `committed` will be re-dispatched
    playing = false; setBtn();
  }
  function tick(){
    var t = simT();
    setClock(t); setBar(t);
    // Collect events that came due since the last dispatch, then stagger them
    // so a batch of co-due events animates as a cascading wave.
    var batch = [];
    for(var k = scheduled; k < TL.length && TL[k].t <= t; k++) batch.push(k);
    if(batch.length){
      scheduled = batch[batch.length - 1] + 1;
      var perItemMs = Math.min(STAGGER_BASE_MS,
                               STAGGER_BATCH_CAP_MS / batch.length) / speed;
      batch.forEach(function(k, off){
        var tm = setTimeout(function(){
          var e = TL[k];
          if(els[e.tid]) els[e.tid].textContent = e.trans;
          if(k + 1 > committed) committed = k + 1;
          setDone();
        }, off * perItemMs);
        staggerTimers.push(tm);
      });
    }
    if(t < maxT + 0.4) raf = requestAnimationFrame(tick);
    else stop();
  }
  function play(){
    if(playing){ stop(); return; }                   // toggle: playing -> pause
    if(committed >= TL.length) setVisible(0);        // finished -> restart
    var resumeT = committed > 0 ? TL[committed-1].t : 0;
    applyState(resumeT);                             // sync visible to committed
    t0 = performance.now() - (resumeT / speed) * 1000;
    playing = true; setBtn(); tick();
  }

  btn.onclick = play;
  document.getElementById('__anim_speed').onchange = function(){
    var newSpeed = parseFloat(this.value);
    if(playing){
      var t = simT(); speed = newSpeed;
      t0 = performance.now() - (t / speed) * 1000;
      setClock(t);
    } else {
      speed = newSpeed;
      var resumeT = committed > 0 ? TL[committed-1].t : 0;
      setClock(resumeT);
    }
  };

  // seek bar: click + drag
  var dragging = false;
  function barSeek(ev){
    var rect = bar.getBoundingClientRect();
    var frac = (ev.clientX - rect.left) / rect.width;
    seek(frac * maxT);
  }
  bar.addEventListener('mousedown', function(ev){
    dragging = true; barSeek(ev); ev.preventDefault();
  });
  document.addEventListener('mousemove', function(ev){
    if(dragging) barSeek(ev);
  });
  document.addEventListener('mouseup', function(){ dragging = false; });

  applyState(0); setBtn();
})();
</script>
"""


def inject_base_href(html_str, href):
    """Insert <base href=...> so relative resources (images/css/js) resolve
    against the original page directory. Absolute URLs are unaffected."""
    if not href:
        return html_str
    tag = f'<base href="{href}">'
    low = html_str.lower()
    h = low.find("<head>")
    if h != -1:
        pos = h + len("<head>")
        return html_str[:pos] + tag + html_str[pos:]
    ht = low.find("<html")
    if ht != -1:
        pos = low.find(">", ht) + 1
        return html_str[:pos] + "<head>" + tag + "</head>" + html_str[pos:]
    return tag + html_str


def build_animation_html(tagged_html_str, timeline, title="", base_href=None):
    """Return a standalone animated HTML string.

    Args:
        tagged_html_str: HTML whose animatable elements already carry a
                         `data-tid="<id>"` attribute.
        timeline:        list of {"tid", "orig", "trans", "t"} dicts;
                         `t` is seconds from translation start.
        title:           label shown in the control bar.
        base_href:       if given, a <base href> is injected so the page's
                         images/assets resolve.
    """
    # `t` is seconds from the page's translation start; keep it as-is so the
    # animation reflects the real wait-then-flip timing. Only clamp negatives.
    timeline = [dict(e, t=max(0.0, e["t"])) for e in timeline]

    # embed the timeline JSON inside <script>. json.dumps does NOT escape '<',
    # so a translated text containing '</script>' (or any '<') would break out
    # of the script tag. Escape '<' as the < JSON unicode escape (valid
    # JSON, parses back to '<').
    timeline_json = json.dumps(timeline, ensure_ascii=False).replace("<", "\\u003c")
    block = (_CONTROL_AND_SCRIPT
             .replace("__TIMELINE__", timeline_json)
             .replace("__TITLE__", _escape(title)))

    lowered = tagged_html_str.lower()
    idx = lowered.rfind("</body>")
    if idx != -1:
        result = tagged_html_str[:idx] + block + tagged_html_str[idx:]
    else:
        result = tagged_html_str + block
    return inject_base_href(result, base_href)


def _escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;"))
