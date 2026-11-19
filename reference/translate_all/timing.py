"""7-bucket wall-clock instrumentation for translation speed measurement.

Buckets (all measured as real wall-clock, summing to End-to-end):
  1. model_serving    - engine/weight load (one-time, reported separately)
  2. preprocessing    - HTML cleansing
  3. prompt_processing- CPU-side prompt/string construction + tokenization
  4. cache_management - QA prefix-cache preload prefill (Seq2Seq: 0)
  5. inference        - model.generate() decode work
  6. post_processing  - output parsing / re-insertion
  7. etc              - residual = e2e - sum(2..6); should be ~0

See /root/.claude/plans/compiled-spinning-frost.md for the design rationale.
"""
import time
from contextlib import contextmanager

# bucket name constants
MODEL_SERVING = "model_serving"
PREPROCESSING = "preprocessing"
PROMPT_PROCESSING = "prompt_processing"
CACHE_MANAGEMENT = "cache_management"
INFERENCE = "inference"
POST_PROCESSING = "post_processing"

_ALL_BUCKETS = [MODEL_SERVING, PREPROCESSING, PROMPT_PROCESSING,
                CACHE_MANAGEMENT, INFERENCE, POST_PROCESSING]
# buckets 2..6 sum into End-to-end; model_serving is one-time, reported apart.
_E2E_BUCKETS = [PREPROCESSING, PROMPT_PROCESSING, CACHE_MANAGEMENT,
                INFERENCE, POST_PROCESSING]


class BucketTimer:
    """Accumulates wall-clock time into mutually-exclusive buckets.

    Usage:
        timer = BucketTimer()
        timer.start_e2e()
        with timer.bucket(PREPROCESSING):
            ...
        timer.stop_e2e()
        row = timer.finalize()
    """

    def __init__(self):
        self.buckets = {b: 0.0 for b in _ALL_BUCKETS}
        self._depth = 0
        self._e2e_start = None
        self._e2e = None
        self.diag = {}  # diagnostic pass-through fields written into the row

    @contextmanager
    def bucket(self, name):
        assert name in self.buckets, f"unknown bucket: {name}"
        # buckets must be mutually exclusive and sequential - never nested,
        # otherwise the same wall-clock would be counted twice.
        assert self._depth == 0, f"bucket('{name}') opened while another bucket is active"
        self._depth += 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.buckets[name] += time.perf_counter() - t0
            self._depth -= 1

    def add(self, name, seconds):
        """Add a pre-measured duration to a bucket (non-context-manager path)."""
        assert name in self.buckets, f"unknown bucket: {name}"
        self.buckets[name] += seconds

    def start_e2e(self):
        self._e2e_start = time.perf_counter()

    def stop_e2e(self):
        assert self._e2e_start is not None, "start_e2e() not called"
        self._e2e = time.perf_counter() - self._e2e_start

    def elapsed(self):
        """Seconds since start_e2e(); used to timestamp animation events
        relative to the page's translation start."""
        assert self._e2e_start is not None, "start_e2e() not called"
        return time.perf_counter() - self._e2e_start

    def finalize(self):
        """Return a flat dict row: 6 buckets + etc residual + e2e + diagnostics."""
        sum_2_6 = sum(self.buckets[b] for b in _E2E_BUCKETS)
        e2e = self._e2e if self._e2e is not None else sum_2_6
        etc = e2e - sum_2_6
        row = {
            "t_model_serving": self.buckets[MODEL_SERVING],
            "t_preprocessing": self.buckets[PREPROCESSING],
            "t_prompt_processing": self.buckets[PROMPT_PROCESSING],
            "t_cache_management": self.buckets[CACHE_MANAGEMENT],
            "t_inference": self.buckets[INFERENCE],
            "t_post_processing": self.buckets[POST_PROCESSING],
            "t_etc": etc,
            "t_e2e": e2e,
            "sum_buckets_2_6": sum_2_6,
            "etc_fraction": (etc / e2e) if e2e > 0 else 0.0,
        }
        row.update(self.diag)
        return row


# --------------------------------------------------------------------------
# token accounting for the Seq2Seq required-output / max_new_tokens cap
# --------------------------------------------------------------------------

def split_markup_text_tokens(html_str, tokenizer):
    """Split an HTML string's tokens into (markup_tokens, text_tokens).

    text  = visible text nodes; expands by the EN->KO ratio r on translation.
    markup= tags/structure/whitespace; passes through ~1:1.

    Counted exactly the way the EN->KO expansion ratio r was measured:
    text_tokens = sum of per-string encodings of non-empty visible strings.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_str, "html.parser")
    for bad in soup(["script", "style", "noscript"]):
        bad.decompose()
    text_pieces = [s.strip() for s in soup.strings if s.strip()]
    text_tokens = sum(len(tokenizer.encode(p, add_special_tokens=False))
                      for p in text_pieces)
    full_tokens = len(tokenizer.encode(html_str, add_special_tokens=False))
    markup_tokens = max(0, full_tokens - text_tokens)
    return markup_tokens, text_tokens


def required_output_tokens(markup_tokens, text_tokens, r):
    """Token count a correct Seq2Seq translation must emit: markup (1:1) + text*r."""
    return int(round(markup_tokens + text_tokens * r))


def count_output_tokens(request_outputs):
    """Sum generated token count over vLLM RequestOutputs."""
    out = 0
    for ro in request_outputs:
        for o in getattr(ro, "outputs", None) or []:
            out += len(getattr(o, "token_ids", None) or [])
    return out


def common_prefix_len(token_lists):
    """Longest common token-prefix length across a list of token-id lists.
    Used to count how many input tokens are shared (prefilled once) when
    several prompts share a leading prefix."""
    if not token_lists:
        return 0
    minl = min(len(t) for t in token_lists)
    for i in range(minl):
        tok = token_lists[0][i]
        if any(t[i] != tok for t in token_lists):
            return i
    return minl


def expansion_ratio_for(name, ratio_table, override=None):
    """Pick the EN->KO token expansion ratio r by substring-matching a model/
    tokenizer name against the ratio table keys (e.g. 'qwen', 'llama')."""
    if override is not None:
        return float(override)
    low = (name or "").lower()
    for key in ratio_table:
        if key in low:
            return ratio_table[key]
    # fall back to the larger ratio (conservative: never truncates the cap)
    return max(ratio_table.values())


# --------------------------------------------------------------------------
# decode-throughput measurement (diagnostic, separate from the 7-bucket table)
# --------------------------------------------------------------------------

def measure_decode_rate(llm, sample_prompt, max_num_seqs,
                        n1=128, n2=640, n_trials=3):
    """Measure sustained decode throughput (tokens/sec) on the given vLLM engine.

    Two prompt lengths cancel the constant prefill term:
      dt = (t@n2) - (t@n1)  ->  decode time of (n2-n1) tokens.
    Returns {"single": rate_batch1, "saturated": rate_batchN}.
    """
    from vllm import SamplingParams
    results = {}
    for cond, bs in [("single", 1), ("saturated", max(1, max_num_seqs))]:
        prompts = [sample_prompt] * bs
        rates = []
        for _ in range(n_trials):
            sp1 = SamplingParams(temperature=0.0, max_tokens=n1, ignore_eos=True)
            sp2 = SamplingParams(temperature=0.0, max_tokens=n2, ignore_eos=True)
            t0 = time.perf_counter()
            llm.generate(prompts, sp1)
            t1 = time.perf_counter()
            llm.generate(prompts, sp2)
            t2 = time.perf_counter()
            dt = (t2 - t1) - (t1 - t0)
            dtok = bs * (n2 - n1)
            if dt > 0:
                rates.append(dtok / dt)
        results[cond] = (sum(rates) / len(rates)) if rates else float("nan")
    return results
