#https://stackoverflow.com/questions/23057631/clone0c98acab374beb118b1301d144b049cd0d827d61element-with-beautifulsoup
import re
import pdb
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.process_html import clean
import time
import math

import requests
from bs4 import BeautifulSoup, Tag, NavigableString
from configs import gen_params
import dotenv
dotenv.load_dotenv()

import tqdm
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from speedup_tests.huggingface_use_cache_test import generate_with_cached_prompt_batch, setup_model
# from llama_cpp_langchain_test import generate_with_cached_prompt_batch as generate_with_cached_prompt_batch_llama
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from timing import (BucketTimer, MODEL_SERVING, PREPROCESSING, PROMPT_PROCESSING,
                    CACHE_MANAGEMENT, INFERENCE, POST_PROCESSING,
                    split_markup_text_tokens, required_output_tokens,
                    expansion_ratio_for, measure_decode_rate, count_output_tokens,
                    common_prefix_len)
from animation import build_animation_html, inject_base_href

sampling_params = SamplingParams(temperature=gen_params.generate_args["temperature"], max_tokens=gen_params.generate_args["max_new_tokens"])
# preload only needs to PREFILL the prefix into the KV cache (caching happens
# during prefill regardless of decode length). vLLM requires max_tokens>=1, so
# use 1 instead of the 32 of `sampling_params` -- avoids generating ~31 throwaway
# tokens per prefix. This SamplingParams is for the QA preload ONLY, never for
# the actual translation (which keeps `sampling_params`, max_tokens=32).
preload_sampling_params = SamplingParams(temperature=gen_params.generate_args["temperature"], max_tokens=1)

# ==========================================================================
# API backend -- run the SAME QA / Seq2Seq method logic against an
# OpenAI-compatible endpoint instead of a local vLLM engine.
#
# APIModel is a drop-in for the vLLM `LLM` object: `.generate(contents,
# sampling_params)` takes a list of *user-message content strings* and returns
# vLLM-RequestOutput-compatible shims, while accumulating the billed token
# usage and the LiteLLM per-call cost (`x-litellm-response-cost` header).
#
# API_MODE gates the prompt-building: vLLM needs chat-templated strings (and,
# for QA, an assistant-prefill continuation); the API takes raw user content.
# ==========================================================================
import threading
from concurrent.futures import ThreadPoolExecutor

API_MODE = False                # set True by main() when --api is passed
API_CONTEXT = 1_000_000         # API context window (pages over it are skipped)
API_QA_MIN_CAP = 2048           # QA per-node max_tokens floor (reasoning headroom)
API_CAP_CEILING = 100000        # API max_tokens hard ceiling (gpt-5.4 accepts
#                                 up to 128k output; a whole translated page
#                                 must fit so Seq2Seq full is not truncated)
API_MAX_WORKERS = 8             # concurrent API calls


class _APIComp:
    def __init__(self, text, finish_reason, n_out):
        self.text = text
        self.finish_reason = finish_reason
        self.token_ids = [0] * n_out          # only len() is read downstream


class _APIOut:
    """vLLM RequestOutput shim: `.outputs[0].text/.finish_reason/.token_ids`."""
    def __init__(self, text="", finish_reason="stop", n_out=0):
        self.outputs = [_APIComp(text, finish_reason, n_out)]
        self.metrics = None                   # no per-request finished_time


class APIModel:
    """vLLM-LLM-compatible shim over an OpenAI-compatible chat endpoint."""

    def __init__(self, model_name, base_url, api_key, max_workers=API_MAX_WORKERS):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
                      "completion_tokens": 0, "reasoning_tokens": 0,
                      "cost": 0.0, "cost_calls": 0}

    def _record(self, resp, raw):
        u = getattr(resp, "usage", None)
        cost = None
        try:
            cost = raw.headers.get("x-litellm-response-cost")
        except Exception:
            pass
        with self._lock:
            self.usage["calls"] += 1
            if u is not None:
                self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                self.usage["completion_tokens"] += \
                    getattr(u, "completion_tokens", 0) or 0
                pd_ = getattr(u, "prompt_tokens_details", None)
                if pd_ is not None:
                    self.usage["cached_tokens"] += \
                        getattr(pd_, "cached_tokens", 0) or 0
                cd = getattr(u, "completion_tokens_details", None)
                if cd is not None:
                    self.usage["reasoning_tokens"] += \
                        getattr(cd, "reasoning_tokens", 0) or 0
            if cost is not None:
                try:
                    self.usage["cost"] += float(cost)
                    self.usage["cost_calls"] += 1
                except (TypeError, ValueError):
                    pass

    def _create(self, content, cap, use_completion_kw):
        key = "max_completion_tokens" if use_completion_kw else "max_tokens"
        raw = self.client.chat.completions.with_raw_response.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            **{key: cap})
        resp = raw.parse()
        self._record(resp, raw)
        ch = resp.choices[0]
        n_out = getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0
        return _APIOut(ch.message.content or "", ch.finish_reason or "stop", n_out)

    def _call(self, content, cap):
        use_completion_kw = False
        for attempt in range(5):
            try:
                return self._create(content, cap, use_completion_kw)
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                if any(s in low for s in ("context length", "maximum context",
                                          "context_length", "too long",
                                          "reduce the length", "string too long")):
                    print(f"  [API too-long] {msg[:140]}")
                    return _APIOut("", "length", 0)
                if "max_tokens" in low and "max_completion_tokens" in low:
                    use_completion_kw = True
                    continue
                if ("rate" in low or "429" in msg) and attempt < 4:
                    time.sleep(20 * (attempt + 1))
                    continue
                if attempt < 4:
                    time.sleep(5)
                    continue
                print(f"  [API error] {msg[:200]}")
                return _APIOut("", "error", 0)
        return _APIOut("", "error", 0)

    def generate(self, prompts, sampling_params, **_kw):
        """prompts: str or list[str] of user-message contents.
        sampling_params: a SamplingParams or a per-prompt list.
        A single SamplingParams with max_tokens<=1 is the QA prefix-preload ->
        no-op for the API (prompt caching is automatic, server-side)."""
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            return []
        single = not isinstance(sampling_params, (list, tuple))
        if single and (getattr(sampling_params, "max_tokens", 99) or 99) <= 1:
            return [_APIOut() for _ in prompts]          # preload no-op
        sp_list = ([sampling_params] * len(prompts) if single
                   else list(sampling_params))
        caps = [min(API_CAP_CEILING,
                    max(API_QA_MIN_CAP, getattr(s, "max_tokens", 0) or 0))
                for s in sp_list]
        results = [None] * len(prompts)
        # warm the shared prefix: first request alone, then the rest in parallel
        results[0] = self._call(prompts[0], caps[0])
        if len(prompts) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = {ex.submit(self._call, prompts[i], caps[i]): i
                        for i in range(1, len(prompts))}
                for f in futs:
                    results[futs[f]] = f.result()
        return results


def setup_api_model(model_name, tokenizer_name=None):
    """Build an APIModel + a proxy tokenizer (for the method code's local
    token-counting paths; billed tokens come from the API response)."""
    base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:4000")
    api_key = os.getenv("OPENAI_COMPAT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_COMPAT_API_KEY not set in .env")
    tok = AutoTokenizer.from_pretrained(
        tokenizer_name or "Qwen/Qwen2.5-Coder-7B-Instruct")
    return APIModel(model_name, base_url, api_key), tok

# Seq2Seq full-document translation prompt (ported from translate_all/main.py)
SEQ2SEQ_PROMPT_TEMPLATE = """
- Translate all texts in a fragment of HTML into Korean (한글).
- Do not translate HTML tags. (e.g. <div>, <p>, <a>, etc. should be left as they are)
- Thus the output should be consistent with the original HTML.
- Do not provide any explanation or preambles.
- Here is the original HTML:
"""
SEQ2SEQ_CHUNK_SIZE = 1024  # characters per chunk in chunked mode


def setup_vllm_model(model_name, gpu_memory_utilization=0.5, enforce_eager=True,
                     max_num_seqs=40, enable_prefix_caching=True, max_model_len=15872, tokenizer_name=None):

    llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len)
    if tokenizer_name is None:
        if model_name.endswith(".gguf"):
            # transformers extracts the tokenizer embedded in a GGUF file when
            # given its directory + the gguf_file kwarg.
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.dirname(os.path.abspath(model_name)),
                gguf_file=os.path.basename(model_name))
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return llm,  tokenizer

def clone(el):
    if isinstance(el, NavigableString):
        return type(el)(el)

    copy = Tag(None, el.builder, el.name, el.namespace, el.nsprefix)
    # work around bug where there is no builder set
    # https://bugs.launchpad.net/beautifulsoup/+bug/1307471
    copy.attrs = dict(el.attrs)
    for attr in ('can_be_empty_element', 'hidden'):
        setattr(copy, attr, getattr(el, attr))
    for child in el.contents:
        copy.append(clone(child))
    return copy


def get_clean_to_original_map(clean, original):
    cnt = 0
    c2o = {}
    for p_c, p_o in zip(clean.find_all(), original.find_all()):
        c2o[cnt] = p_o
        p_c.to_o_key = cnt
        cnt += 1

    return c2o


def add_id(soup):
    id_to_tag_map = {}
    ID_CNT = 0

    for p1 in soup.find_all():
        p1.attrs={}
        if p1.string is not None:
            id_to_tag_map[ID_CNT] = p1
            new_text = f"<T_{ID_CNT}> {p1.get_text()} </T_{ID_CNT}>"
            ID_CNT += 1
            p1.string.replace_with(new_text)

    return id_to_tag_map


model, tokenizer = None, None
def translate_soup_batch_qa(clean_soup, id_to_tag_map, args, timer):
    """Q&A translation. Fills PROMPT_PROCESSING / CACHE_MANAGEMENT / INFERENCE /
    POST_PROCESSING buckets of `timer`. Returns (translation_result_str,
    pointer_finish_ts) where pointer_finish_ts maps pointer id -> perf_counter
    timestamp at which that pointer's translation completed."""
    use_cache = not args.no_cache
    context_size_in_tags = args.context_size_in_tags
    split_side_size = args.split_side_size
    split_base_prompt = args.context_size_in_tags > 0

    assert context_size_in_tags > 0
    # max_num_seqs is a vLLM continuous-batching knob; irrelevant for the API.
    assert API_MODE or context_size_in_tags % args.max_num_seqs == 0

    global model, tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = setup_vllm_model(model_name=args.model_name, enable_prefix_caching=use_cache,
                                            gpu_memory_utilization=args.gpu_memory_utilization,
                                            enforce_eager=True, max_num_seqs=args.max_num_seqs,
                                            tokenizer_name=args.tokenizer_name)

    # -------- PROMPT_PROCESSING: prettify, normalize, build prompts/splits ----
    with timer.bucket(PROMPT_PROCESSING):
        html_str = str(clean_soup.prettify())
        html_str = html_str.replace("&lt;&lt;", "<<")
        html_str = html_str.replace("&gt;&gt;", ">>")
        html_str = html_str.replace("&lt;", "<")
        html_str = html_str.replace("&gt;", ">")

        base_prompt_template = """
- The following is a HTML of a website
####
{html}
####

- Find a text in a tag and translate it into Korean (한글). The text in the HTML can be identified by a <T_(NUMBER)> tags.
- If translation is not needed, just return the original text.
- Only translated text required. Do not provide any explanation or preambles.
- Response format
    <T_(NUMBER)> TEXT_BEFORE_TRANSLATION </T_(NUMBER)> --> TRANSLATED_TEXT_HERE
- Response Examples
    <T_123> Hello  --> 안녕하세요
    <T_124> Where are you from? --> 어디서 오셨어요?
    <T_125> 안녕하세요. -->  안녕하세요.
"""
        following_prompt_template = ("""- You need to translate <T_{id}> into Korean (한글)
- 당신의 답변:
""",
"""
<T_{id}>
""")  # (user following, assistant following)

        following_prompt_list = [tuple(t.format(id=id) for t in following_prompt_template)
                                 for id in range(html_str.count("<T_"))]

        split_size = context_size_in_tags  # context size in number of tags

        cache_prompt_list = []
        following_prompt_batch_list = []
        for i in range(0, len(following_prompt_list), split_size):
            split_start_index = max(html_str.index(f"<T_{i}>") - split_side_size, 0)
            try:
                split_end_index = min(html_str.index(f"</T_{i+split_size-1}>") + len(f"</T_{i+split_size-1}>") + split_side_size, len(html_str))
            except:
                split_end_index = len(html_str)

            split_html_string = html_str[split_start_index:split_end_index]
            split_base_prompt_str = base_prompt_template.format(html=split_html_string)

            cache_prompt_list.append(split_base_prompt_str)
            following_prompt_batch_list.append(following_prompt_list[i:i+split_size])

    assert split_base_prompt, "split_base_prompt must be True"

    # direct token accounting -- DIAGNOSTIC ONLY. Deliberately OUTSIDE every
    # timer.bucket() so this instrumentation tokenization is NOT attributed to
    # a method bucket (it would inflate PROMPT_PROCESSING); it falls into the
    # small ETC residual instead. Counts are approximate: prefix and suffix are
    # tokenized separately, so chat-template tokens / boundary effects are not
    # exactly reflected.
    prefix_tok = [len(tokenizer.encode(cp)) for cp in cache_prompt_list]
    qa_input_naive = 0               # tokens to prefill WITHOUT prefix reuse
    qa_input_prefilled = sum(prefix_tok)  # tokens actually prefilled WITH reuse
    for _bidx, _fol in enumerate(following_prompt_batch_list):
        for (_us, _as) in _fol:
            _suf = len(tokenizer.encode(_us + _as))
            qa_input_prefilled += _suf
            qa_input_naive += prefix_tok[_bidx] + _suf

    # QA cannot run if a block's prefix exceeds the context window. This
    # happens on a page carrying a large text-less markup blob that clean()
    # did not strip (e.g. page 39: ~7k text tokens but a ~30k-token prefix).
    # Flag & skip -- the aggregator drops it and compares on the common subset.
    max_prefix = max(prefix_tok) if prefix_tok else 0
    if max_prefix > (API_CONTEXT if API_MODE else 15872) - 256:
        print(f"[QA] prefix {max_prefix} tok exceeds context window -- skipping page")
        return "", {}, {"qa_prompt_too_long": True,
                        "max_prefix_tokens": max_prefix,
                        "input_tokens": qa_input_naive,
                        "cached_tokens": qa_input_naive - qa_input_prefilled,
                        "output_tokens": 0}

    NUM_CACHE_AT_A_TIME = args.num_parallel_cache_blocks
    translated_result = []
    pointer_finish_ts = {}
    output_tokens = 0

    for i in range(0, len(cache_prompt_list), NUM_CACHE_AT_A_TIME):
        cache_prompt_batch = cache_prompt_list[i:i+NUM_CACHE_AT_A_TIME]
        following_prompt_batch = following_prompt_batch_list[i:i+NUM_CACHE_AT_A_TIME]

        # ---- CACHE_MANAGEMENT: prefix-cache preload prefill ----
        # API mode skips the preload: prompt caching on the endpoint is
        # automatic & server-side, and an explicit preload call would be billed.
        if args.no_cache == False and not API_MODE:
            with timer.bucket(CACHE_MANAGEMENT):
                model.generate(cache_prompt_batch, preload_sampling_params)

        # ---- PROMPT_PROCESSING: per-question prompt build ----
        # vLLM: chat-template with an assistant-prefill continuation (<T_id>).
        # API: raw user-message content (no assistant prefill -- chat endpoints
        # cannot continue an assistant turn); the block prefix is identical
        # across the block's questions so the endpoint caches it automatically.
        with timer.bucket(PROMPT_PROCESSING):
            complete_prompt = []
            for j in range(len(cache_prompt_batch)):
                for k in range(len(following_prompt_batch[j])):
                    user_content = cache_prompt_batch[j] + following_prompt_batch[j][k][0]
                    if API_MODE:
                        complete_prompt.append(user_content)
                        continue
                    messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": following_prompt_batch[j][k][1]}
                    ]
                    complete_prompt.append(tokenizer.apply_chat_template(messages, tokenize=False,
                                                                         apply_generation_prompt=False,
                                                                         continue_final_message=True))

        # ---- INFERENCE: suffix prefill + decode of all questions in the block ----
        with timer.bucket(INFERENCE):
            generated_result = model.generate(complete_prompt, sampling_params)
        output_tokens += count_output_tokens(generated_result)
        block_finish = timer.elapsed()  # seconds from page translation start

        # record per-pointer completion time for the progress animation.
        # Within a block all questions run in parallel (continuous batching),
        # so they share the block's completion timestamp.
        for j in range(len(cache_prompt_batch)):
            for k in range(len(following_prompt_batch[j])):
                pid = (i + j) * split_size + k
                pointer_finish_ts[pid] = block_finish

        for result in generated_result:
            print("result:", result.outputs[0].text)
            translated_result.append("<T_0> " + result.outputs[0].text)

    # ---- POST_PROCESSING: regex parse of generated text ----
    with timer.bucket(POST_PROCESSING):
        post_processed_translated_result = []
        for i, elem in enumerate(translated_result):
            matches = list(re.finditer(r"<T_\d+>", elem))
            matches = list(re.finditer(r"-->", elem)) if not matches else matches
            if not matches:
                print(elem)
                raise Exception(f"No match found for {i}th element")
            last_match = matches[-1]
            start_index = last_match.start()
            if "-->" in elem[start_index:]:
                after_arrow = elem[start_index:].split("-->")[-1]
                before_closing_tag = after_arrow.split("</")[0]
                processed_elem = f"<T_{i}>{before_closing_tag}"
            else:
                before_closing_tag = elem[start_index:].split("</")[0]
                processed_elem = before_closing_tag
            post_processed_translated_result.append(processed_elem)

    tok_stats = {"input_tokens": qa_input_naive,
                 "cached_tokens": qa_input_naive - qa_input_prefilled,
                 "output_tokens": output_tokens,
                 "qa_prompt_too_long": False}
    return "\n".join(post_processed_translated_result), pointer_finish_ts, tok_stats


def apply_translation(id_to_tag_map, c2o, translation_result):
    """Re-insert translations into the original HTML. Also stamps each
    translated original tag with a `data-tid` attribute and returns
    {pointer_id: (original_text, translated_text)} for the animation."""
    applied = {}
    for elem in translation_result.split("<T_"):
        if len(elem) == 0:
            continue
        id_and_text = elem.split(">")
        if id_and_text[0].isdigit() == False or len(id_and_text) < 2:
            continue
        elem_id = int(id_and_text[0])
        elem_text = id_and_text[1].strip()
        if elem_id in id_to_tag_map.keys():
            clean_tag = id_to_tag_map[elem_id]
            original_tag = c2o[clean_tag.to_o_key]

            orig_text = original_tag.get_text()

            id_to_tag_map[elem_id].string.replace_with(elem_text)

            if original_tag.string is not None:
                original_tag.string.replace_with(elem_text)
            else:
                TRANSLATED = False
                for child in original_tag.children:
                    if isinstance(child, NavigableString) and not isinstance(child, Tag):
                        child.replace_with(elem_text)
                        TRANSLATED = True
                        break
                assert TRANSLATED

            original_tag['data-tid'] = str(elem_id)
            applied[elem_id] = (orig_text, elem_text)
    return applied


def _reinsert_into_original(clean_soup, s2s_soup, orig_soup, c2o):
    """If the Seq2Seq translation preserved the cleansed HTML's tag structure,
    copy each translated text node into the ORIGINAL markup (so the saved page
    keeps its images/scripts/layout); otherwise keep the cleansed translated
    soup. Tags the chosen output's text nodes with data-tid (document order)
    for the animation.

    Returns (final_soup, structure_consistent, pairs) where pairs is a list of
    (original_text, translated_text) in data-tid order.

    The structure check doubles as the markup-consistency metric for Seq2Seq."""
    s2s_root = s2s_soup.find('html') or s2s_soup
    clean_tags = clean_soup.find_all()
    s2s_tags = s2s_root.find_all()
    pairs = []

    if [t.name for t in clean_tags] != [t.name for t in s2s_tags]:
        # structure broken -> save & animate the cleansed translated soup
        clean_text = [t.get_text() for t in clean_tags if t.string is not None]
        k = 0
        for stag in s2s_tags:
            if stag.string is None:
                continue
            stag['data-tid'] = str(k)
            trans = stag.get_text()
            orig = clean_text[k] if k < len(clean_text) else trans
            pairs.append((orig, trans))
            k += 1
        return s2s_soup, False, pairs

    k = 0
    for ctag, stag in zip(clean_tags, s2s_tags):
        if ctag.string is None:
            continue
        orig_tag = c2o.get(getattr(ctag, "to_o_key", None))
        trans = stag.get_text()
        orig_text = trans
        if orig_tag is not None:
            orig_text = orig_tag.get_text()
            if orig_tag.string is not None:
                orig_tag.string.replace_with(trans)
            else:
                for child in list(orig_tag.children):
                    if isinstance(child, NavigableString) and not isinstance(child, Tag):
                        child.replace_with(trans)
                        break
            orig_tag['data-tid'] = str(k)
        pairs.append((orig_text, trans))
        k += 1
    return orig_soup, True, pairs


def _chunk_html(html_str, size=SEQ2SEQ_CHUNK_SIZE):
    """Split HTML into <=size character chunks, cutting before an incomplete
    trailing tag. Ported from translate_all/main.py (exit(0) bug removed)."""
    chunks = []
    cursor = 0
    n = len(html_str)
    while cursor < n:
        frag = html_str[cursor:cursor+size]
        last_lt = frag.rfind('<')
        last_gt = frag.rfind('>')
        if last_lt > 0 and last_gt < last_lt:
            # incomplete trailing tag -> cut before it
            frag = frag[:last_lt]
            chunks.append(frag)
            cursor += last_lt
            continue
        chunks.append(frag)
        cursor += len(frag)
    return [c for c in chunks if c]


def _seq2seq_prompt(html_chunk):
    """Build the Seq2Seq prompt for one HTML chunk. API mode returns the raw
    user-message content (the API client adds the chat template); vLLM mode
    returns the chat-templated string fed straight to model.generate()."""
    global tokenizer
    content = SEQ2SEQ_PROMPT_TEMPLATE + html_chunk
    if API_MODE:
        return content
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def translate_soup_batch_seq2seq_generator(clean_soup, args, timer):
    """Seq2Seq full-document / chunked translation on the shared vLLM engine.

    full mode -- two cases:
      * doc fits the context window (prompt + output cap) -> measured full-doc
        pass (diag['full_estimated']=False).
      * doc too long -> ESTIMATE: split into the largest chunks that still
        leave room for the translated output (input + output <= max_model_len),
        translate ONE representative chunk, scale its time by the (fractional)
        chunk count. This is the whole-doc time the model cannot run directly.
        diag['full_estimated']=True, diag['prompt_too_long']=True.
    chunked mode -- fixed-size chunks, all submitted in one batched generate().

    Fills PROMPT_PROCESSING / INFERENCE / POST_PROCESSING buckets
    (CACHE_MANAGEMENT stays 0 - Seq2Seq has no prefix-cache preload).
    Returns (translated_soup, node_time, diag)."""
    global model, tokenizer
    diag = {"full_estimated": False, "prompt_too_long": False}
    r = args.resolved_r
    cap_margin = args.seq2seq_cap_margin
    # API context dwarfs the local 15872 -- a whole page fits as one request.
    max_model_len = API_CONTEXT if API_MODE else 15872

    # ---- PROMPT_PROCESSING: prettify ----
    with timer.bucket(PROMPT_PROCESSING):
        html_str = str(clean_soup.prettify())
        html_str = html_str.replace("&lt;&lt;", "<<").replace("&gt;&gt;", ">>")
        html_str = html_str.replace("&lt;", "<").replace("&gt;", ">")

    if args.seq2seq_mode == "full_dirty":
        # raw (uncleaned) HTML -> always far over the context window; the
        # whole-doc fit check would tokenize tens of MB needlessly, so go
        # straight to the estimate.
        return _seq2seq_estimate_full(clean_soup, html_str, args, timer, diag, None)

    if args.seq2seq_mode == "full":
        # whole document as one prompt -- does prompt + output cap fit?
        with timer.bucket(PROMPT_PROCESSING):
            mk, tx = split_markup_text_tokens(html_str, tokenizer)
            req = required_output_tokens(mk, tx, r)
            cap = max(8, int(math.ceil(req * cap_margin)))
            prompt = _seq2seq_prompt(html_str)
            ptoks = len(tokenizer.encode(prompt))
        if ptoks + cap <= max_model_len:
            return _seq2seq_translate_chunks(
                clean_soup, [html_str], [prompt], [cap], req, args, timer, diag)
        return _seq2seq_estimate_full(clean_soup, html_str, args, timer, diag, req)

    # ---- chunked mode: fixed-size chunks, one batched generate() ----
    with timer.bucket(PROMPT_PROCESSING):
        chunks = _chunk_html(html_str, args.seq2seq_chunk_size)
        prompts, caps = [], []
        total_required = 0
        for ch in chunks:
            cmk, ctx = split_markup_text_tokens(ch, tokenizer)
            creq = required_output_tokens(cmk, ctx, r)
            total_required += creq
            caps.append(max(8, int(math.ceil(creq * cap_margin))))
            prompts.append(_seq2seq_prompt(ch))
    return _seq2seq_translate_chunks(
        clean_soup, chunks, prompts, caps, total_required, args, timer, diag)


def _seq2seq_translate_chunks(clean_soup, chunks, prompts, caps, total_required,
                              args, timer, diag):
    """Run vLLM generate() over a prompt list (1 prompt for measured-full,
    N for chunked) and fill the INFERENCE / POST_PROCESSING buckets."""
    global model, tokenizer
    diag["required_output_tokens"] = total_required
    diag["num_chunks"] = len(chunks)

    # ---- INFERENCE: actual generate() with per-chunk runaway cap ----
    sp_list = [SamplingParams(temperature=0.0, max_tokens=c) for c in caps]
    wall_start = time.time()
    t_inf_start_rel = timer.elapsed()  # seconds from page translation start
    with timer.bucket(INFERENCE):
        t_inf0 = time.perf_counter()
        outputs = model.generate(prompts, sp_list)
        t_inf1 = time.perf_counter()
    inference_time = t_inf1 - t_inf0

    finish_reasons = [o.outputs[0].finish_reason for o in outputs]
    diag["non_termination"] = sum(1 for fr in finish_reasons if fr == "length") / max(1, len(finish_reasons))

    # direct token accounting: chunks share the instruction prefix, which the
    # engine caches after the first chunk -> only that shared prefix is "cached".
    ptoks_list = [tokenizer.encode(p) for p in prompts]
    _common = common_prefix_len(ptoks_list)
    _input = sum(len(t) for t in ptoks_list)
    _computed = _common + sum(len(t) - _common for t in ptoks_list)
    diag["input_tokens"] = _input
    diag["cached_tokens"] = _input - _computed
    diag["output_tokens"] = count_output_tokens(outputs)

    # ---- POST_PROCESSING: reassemble translated HTML ----
    with timer.bucket(POST_PROCESSING):
        chunk_texts = [o.outputs[0].text for o in outputs]
        translated_html = "".join(chunk_texts)
        out_soup = BeautifulSoup(translated_html, 'html.parser')

    node_time = _seq2seq_node_times(clean_soup, chunks, outputs, args.seq2seq_mode,
                                    wall_start, t_inf_start_rel, inference_time)
    return out_soup, node_time, diag


def _seq2seq_estimate_full(clean_soup, html_str, args, timer, diag, req):
    """Estimate full-document Seq2Seq time for a page that exceeds the context
    window. Split into the largest chunks whose input AND translated output
    both fit (chunk_tok * (1 + cap_margin*r) <= budget), translate ONE chunk,
    and scale its time by the fractional chunk count doc_tok / chunk_tok.

    The whole-doc pass the model cannot run is approximated as
        t_one_chunk * (doc_tokens / chunk_tokens)
    -- equivalent to translating max-size chunks one after another and summing.
    Sizing the chunk for OUTPUT room (not just input) is essential: a chunk
    filled to the input limit would truncate its output, breaking markup and
    measuring a too-short generate(). diag['full_estimated']=True."""
    global model, tokenizer
    r = args.resolved_r
    cap_margin = args.seq2seq_cap_margin
    max_model_len = 15872
    diag["full_estimated"] = True
    diag["prompt_too_long"] = True

    # largest chunk leaving room for the translation:
    #   overhead + chunk_tok + cap_margin*r*chunk_tok <= max_model_len
    # the 0.88 factor is headroom: the chunker is char-based, so a chunk can
    # carry more tokens than the doc-average chars/token estimate predicts.
    # Without it, prompt + output cap can tip over max_model_len and the
    # generation truncates (finish_reason='length') -> the per-chunk time
    # would be measured too short.
    overhead = len(tokenizer.encode(_seq2seq_prompt("")))
    chunk_budget_tok = max(64, int(0.88 * (max_model_len - overhead)
                                   / (1.0 + cap_margin * r)))
    doc_tok = len(tokenizer.encode(html_str, add_special_tokens=False))
    chars_per_tok = len(html_str) / max(1, doc_tok)
    chunk_chars = max(256, int(chunk_budget_tok * chars_per_tok))

    with timer.bucket(PROMPT_PROCESSING):
        chunks = _chunk_html(html_str, chunk_chars)
    chunk0 = chunks[0]

    # translate ONE representative chunk, measured raw, then scale by count.
    t_p0 = time.perf_counter()
    cmk, ctx = split_markup_text_tokens(chunk0, tokenizer)
    ccap = max(8, int(math.ceil(required_output_tokens(cmk, ctx, r) * cap_margin)))
    prompt0 = _seq2seq_prompt(chunk0)
    t_p1 = time.perf_counter()

    t_i0 = time.perf_counter()
    outputs = model.generate([prompt0],
                             [SamplingParams(temperature=0.0, max_tokens=ccap)])
    t_i1 = time.perf_counter()

    t_o0 = time.perf_counter()
    out_soup = BeautifulSoup(outputs[0].outputs[0].text, "html.parser")
    t_o1 = time.perf_counter()

    chunk0_tok = max(1, len(tokenizer.encode(chunk0, add_special_tokens=False)))
    n_chunks = doc_tok / chunk0_tok  # fractional -> the partial last chunk counts

    # scale the per-chunk wall-clock by the chunk count into the page buckets
    timer.add(PROMPT_PROCESSING, (t_p1 - t_p0) * n_chunks)
    timer.add(INFERENCE, (t_i1 - t_i0) * n_chunks)
    timer.add(POST_PROCESSING, (t_o1 - t_o0) * n_chunks)

    fr = outputs[0].outputs[0].finish_reason
    diag["non_termination"] = 1.0 if fr == "length" else 0.0
    diag["num_chunks"] = n_chunks
    # full_dirty skips the whole-doc fit check -> req not precomputed; estimate
    # it from the document token count.
    if req is None:
        req = doc_tok * r
    diag["required_output_tokens"] = int(round(req))
    diag["input_tokens"] = doc_tok
    diag["cached_tokens"] = 0
    diag["output_tokens"] = int(round(count_output_tokens(outputs) * n_chunks))
    diag["estimate_chunk_tokens"] = chunk0_tok
    return out_soup, [], diag


def _seq2seq_node_times(clean_soup, chunks, outputs, mode,
                        wall_start, t_inf_start_rel, inference_time):
    """Flip time (seconds from page translation start) per cleansed text node,
    reflecting Seq2Seq's measured speed:
      full    -> sequential top-to-bottom sweep over inference_time
      chunked -> each node flips when its chunk's request finished."""
    N = len([t for t in clean_soup.find_all() if t.string is not None])

    if mode == "chunked":
        nchunks = len(chunks)
        chunk_finish = []
        for c in range(nchunks):
            ft = getattr(getattr(outputs[c], "metrics", None), "finished_time", None)
            if ft is not None:
                chunk_finish.append(t_inf_start_rel + max(0.0, ft - wall_start))
            else:
                chunk_finish.append(t_inf_start_rel + (c + 1) / nchunks * inference_time)
        node_chunk_counts = []
        for ch in chunks:
            cs = BeautifulSoup(ch, "html.parser")
            node_chunk_counts.append(max(1, len([s for s in cs.strings if s.strip()])))
        node_time = []
        ci = 0
        seen = 0
        for k in range(N):
            while ci < nchunks - 1 and seen >= node_chunk_counts[ci]:
                seen = 0
                ci += 1
            node_time.append(chunk_finish[ci])
            seen += 1
        return node_time
    else:
        # full mode: sequential sweep, node k appears at (k+1)/N of inference
        return [t_inf_start_rel + (k + 1) / max(1, N) * inference_time
                for k in range(N)]


# ==========================================================================
# tag_json (Method 4): cleansed -> every visible text node wrapped <T_i> ->
# chunked tagged HTML -> per-chunk JSON{i: translation} -> reassemble.
# Shares chat-templating + chunked batched generate + common-prefix cached
# accounting with the Seq2Seq chunked path; differs in (a) input is the tagged
# HTML and (b) output is parsed as a JSON map of tag-id -> translation, then
# untagged back into the original markup.
# ==========================================================================
TAG_JSON_PROMPT_TEMPLATE = """
- The HTML fragment below has visible text wrapped in <T_0>, <T_1>, ... <T_N> tags.
- Translate ONLY the text inside each <T_i> tag into Korean (한글).
- Return a JSON object that maps each tag number (as a string key) to its Korean translation.
  Example: {"0": "번역", "1": "번역"}
- Output ONLY the JSON object: no preamble, no code fences, no other text.
- Here is the HTML fragment:
"""

_T_MARKER = re.compile(r'<T_(\d+)>(.*?)</T_\1>', re.DOTALL)


def _tag_text_nodes_main(html_str):
    """Wrap every non-whitespace visible text node in literal <T_i> markers,
    numbered in document order. Returns (tagged_html_str, {id: original_text}).
    Matches run_quality_experiment.tag_text_nodes for cross-experiment parity."""
    from bs4 import Comment as _Comment
    soup = BeautifulSoup(html_str, "html.parser")
    nodes = [ns for ns in soup.find_all(string=True)
             if not isinstance(ns, _Comment) and ns.strip()]
    originals = {}
    for i, ns in enumerate(nodes):
        text = ns.strip()
        originals[i] = text
        ns.replace_with(f"<T_{i}> {text} </T_{i}>")
    return str(soup).replace("&lt;", "<").replace("&gt;", ">"), originals


def _tag_json_prompt(html_chunk):
    """Chat-templated tag_json prompt (vLLM); raw user content (API)."""
    global tokenizer
    content = TAG_JSON_PROMPT_TEMPLATE + html_chunk
    if API_MODE:
        return content
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def _parse_json_tagmap(text):
    """Tolerant parser: returns (dict[int,str], parsed_ok). Strips ```json fences,
    then loads the {...} block; falls back to regex on quote/colon pairs."""
    import json as _json
    t = (text or "").strip()
    t = re.sub(r'^```(?:json)?', '', t).strip()
    t = re.sub(r'```$', '', t).strip()
    s, e = t.find('{'), t.rfind('}')
    if s != -1 and e != -1 and e > s:
        try:
            obj = _json.loads(t[s:e + 1])
            return {int(k): str(v) for k, v in obj.items()}, True
        except Exception:
            pass
    out = {}
    for m in re.finditer(r'"(\d+)"\s*:\s*"(.*?)"', t, re.DOTALL):
        out[int(m.group(1))] = m.group(2)
    return out, False


def _untag_tagged_html(tagged_html, merged):
    """Replace each <T_i>...</T_i> marker with merged[i]; keep the original
    English when there is no translation, so markup structure stays intact."""
    def repl(m):
        tr = merged.get(int(m.group(1)))
        return tr if tr else m.group(2).strip()
    return _T_MARKER.sub(repl, tagged_html)


def translate_soup_batch_tag_json(clean_soup, args, timer):
    """tag_json translation. Fills PROMPT_PROCESSING / INFERENCE / POST_PROCESSING
    (CACHE_MANAGEMENT stays 0 -- like Seq2Seq, no explicit prefix preload).
    Returns (translated_soup, node_time=[], diag); node_time is empty because
    tag_json post-processing is a single regex untag, not a streamed sweep."""
    global model, tokenizer
    diag = {"prompt_too_long": False, "json_parse_failures": 0,
            "full_estimated": False}
    r = args.resolved_r
    cap_margin = args.seq2seq_cap_margin

    # ---- PROMPT_PROCESSING: prettify -> tag -> chunk -> build prompts + caps --
    with timer.bucket(PROMPT_PROCESSING):
        html_str = str(clean_soup.prettify())
        html_str = html_str.replace("&lt;&lt;", "<<").replace("&gt;&gt;", ">>")
        html_str = html_str.replace("&lt;", "<").replace("&gt;", ">")
        tagged_html, originals = _tag_text_nodes_main(html_str)
        chunks = _chunk_html(tagged_html, args.seq2seq_chunk_size)
        prompts, caps = [], []
        total_required = 0
        for ch in chunks:
            ids_in_chunk = [int(mm.group(1))
                            for mm in re.finditer(r'<T_(\d+)>', ch)]
            text_tok = sum(len(tokenizer.encode(originals.get(i, ""),
                                                add_special_tokens=False))
                           for i in ids_in_chunk)
            n_tags = len(ids_in_chunk)
            # JSON output = Σ Korean translation + ~10 tok per `"i":"…",` pair
            req = int(round(text_tok * r)) + n_tags * 10 + 5
            total_required += req
            caps.append(max(8, int(math.ceil(req * cap_margin))))
            prompts.append(_tag_json_prompt(ch))
    diag["required_output_tokens"] = total_required
    diag["num_chunks"] = len(chunks)

    # ---- INFERENCE: one batched generate() over all chunk prompts ----
    sp_list = [SamplingParams(temperature=0.0, max_tokens=c) for c in caps]
    wall_start = time.time()
    t_inf_start_rel = timer.elapsed()
    with timer.bucket(INFERENCE):
        t_inf0 = time.perf_counter()
        outputs = model.generate(prompts, sp_list)
        t_inf1 = time.perf_counter()
    inference_time = t_inf1 - t_inf0
    finish_reasons = [o.outputs[0].finish_reason for o in outputs]
    diag["non_termination"] = (sum(1 for fr in finish_reasons if fr == "length")
                               / max(1, len(finish_reasons)))

    # ---- direct token accounting: same scheme as _seq2seq_translate_chunks ----
    ptoks_list = [tokenizer.encode(p) for p in prompts]
    _common = common_prefix_len(ptoks_list)
    _input = sum(len(t) for t in ptoks_list)
    _computed = _common + sum(len(t) - _common for t in ptoks_list)
    diag["input_tokens"] = _input
    diag["cached_tokens"] = _input - _computed
    diag["output_tokens"] = count_output_tokens(outputs)

    # ---- POST_PROCESSING: per-chunk JSON parse -> merge -> untag -> soup ----
    with timer.bucket(POST_PROCESSING):
        merged = {}
        for out in outputs:
            text = out.outputs[0].text
            trans, ok = _parse_json_tagmap(text)
            if not ok:
                diag["json_parse_failures"] += 1
            merged.update(trans)
        out_html = _untag_tagged_html(tagged_html, merged)
        out_soup = BeautifulSoup(out_html, "html.parser")

    # ---- per-node animation timestamps: each tag's events fire when its
    # chunk's request finished. Mirrors Seq2Seq chunked, but counts <T_i>
    # markers per chunk (BS-parsing the tagged HTML mis-parses the literals).
    nchunks = len(chunks)
    chunk_finish = []
    for c in range(nchunks):
        ft = getattr(getattr(outputs[c], "metrics", None), "finished_time", None)
        if ft is not None:
            chunk_finish.append(t_inf_start_rel + max(0.0, ft - wall_start))
        else:
            chunk_finish.append(t_inf_start_rel
                                + (c + 1) / nchunks * inference_time)
    counts = [len(re.findall(r'<T_(\d+)>', ch)) for ch in chunks]
    N = sum(counts)
    node_time = []
    ci, seen = 0, 0
    for _k in range(N):
        while ci < nchunks - 1 and seen >= counts[ci]:
            seen = 0
            ci += 1
        node_time.append(chunk_finish[ci])
        seen += 1
    return out_soup, node_time, diag


# ==========================================================================
# contextless (per-node, no surrounding context): each visible text node is
# translated in isolation -- one model request per node, NO shared HTML
# context (mirrors Chrome's per-text-node translation). Differs from QA in
# that the only shared prefix across requests is the tiny instruction text;
# no block-prefix caching, no node-window context.
# ==========================================================================
CONTEXTLESS_PROMPT_TEMPLATE = (
    "- Translate the following text into Korean (한글).\n"
    "- Do not provide any explanation or preamble.\n"
    "- Return ONLY the Korean translation of the text.\n"
    "- Text: {text}\n"
    "- Korean:"
)


def _contextless_prompt(text):
    """Chat-templated contextless prompt (vLLM); raw user content (API)."""
    global tokenizer
    content = CONTEXTLESS_PROMPT_TEMPLATE.format(text=text)
    if API_MODE:
        return content
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def translate_soup_batch_contextless(clean_soup, args, timer):
    """contextless translation: every visible text node is sent to the model on
    its own, with only the instruction text as shared prefix (no surrounding
    HTML, no node-window context). One request per node; total requests =
    number of text nodes on the page. Fills PROMPT_PROCESSING / INFERENCE /
    POST_PROCESSING."""
    global model, tokenizer
    diag = {"prompt_too_long": False, "json_parse_failures": 0,
            "full_estimated": False}
    r = args.resolved_r
    cap_margin = args.seq2seq_cap_margin

    # ---- PROMPT_PROCESSING: collect text nodes, build per-node prompts + caps.
    # Nodes whose (prompt + cap) exceed the model context (e.g. a markup blob
    # clean() did not strip -- same condition that gives QA's qa_prompt_too_long)
    # are SKIPPED: they are kept as the original English text in the output.
    max_model_len = API_CONTEXT if API_MODE else 15872
    with timer.bucket(PROMPT_PROCESSING):
        from bs4 import Comment as _Comment
        texts = []
        for ns in clean_soup.find_all(string=True):
            if isinstance(ns, _Comment):
                continue
            t = str(ns).strip()
            if t:
                texts.append(t)
        keep_mask = []
        prompts, caps = [], []
        total_required = 0
        for t in texts:
            text_tok = len(tokenizer.encode(t, add_special_tokens=False))
            req = int(round(text_tok * r)) + 5
            cap = max(8, int(math.ceil(req * cap_margin)))
            p = _contextless_prompt(t)
            if len(tokenizer.encode(p)) + cap > max_model_len:
                keep_mask.append(False)
                continue
            keep_mask.append(True)
            prompts.append(p)
            caps.append(cap)
            total_required += req
    diag["required_output_tokens"] = total_required
    diag["num_chunks"] = len(prompts)        # one request per kept node
    diag["nodes_skipped_too_long"] = sum(1 for k in keep_mask if not k)

    # ---- INFERENCE: one batched generate over all node prompts ----
    sp_list = [SamplingParams(temperature=0.0, max_tokens=c) for c in caps]
    wall_start = time.time()
    t_inf_start_rel = timer.elapsed()
    with timer.bucket(INFERENCE):
        t_inf0 = time.perf_counter()
        outputs = model.generate(prompts, sp_list)
        t_inf1 = time.perf_counter()
    inference_time = t_inf1 - t_inf0
    finish_reasons = [o.outputs[0].finish_reason for o in outputs]
    diag["non_termination"] = (sum(1 for fr in finish_reasons if fr == "length")
                               / max(1, len(finish_reasons)))

    # ---- direct token accounting: same scheme as Seq2Seq chunked / tag_json --
    ptoks_list = [tokenizer.encode(p) for p in prompts]
    _common = common_prefix_len(ptoks_list)
    _input = sum(len(t) for t in ptoks_list)
    _computed = _common + sum(len(t) - _common for t in ptoks_list)
    diag["input_tokens"] = _input
    diag["cached_tokens"] = _input - _computed
    diag["output_tokens"] = count_output_tokens(outputs)

    # ---- POST_PROCESSING: round-trip clean_soup to a fresh tree, then replace
    # each visible text node by its translation. Skipped (too-long) nodes keep
    # the original text; kept nodes use the model's output, aligned through
    # keep_mask.
    with timer.bucket(POST_PROCESSING):
        from bs4 import Comment as _Comment
        translations = [out.outputs[0].text.strip() for out in outputs]
        full_trans = []
        j = 0
        for i, keep in enumerate(keep_mask):
            if keep:
                full_trans.append(translations[j] if j < len(translations) else texts[i])
                j += 1
            else:
                full_trans.append(texts[i])     # kept untranslated (too long)
        out_soup = BeautifulSoup(str(clean_soup), "html.parser")
        out_nodes = [ns for ns in out_soup.find_all(string=True)
                     if not isinstance(ns, _Comment) and str(ns).strip()]
        for ns, tr in zip(out_nodes, full_trans):
            if tr:
                ns.replace_with(tr)

    # ---- per-node animation timestamps: each kept node fires when its own
    # request finished (one request per node). Skipped (too-long) nodes share
    # the final finish time -- they were never translated, so the "flip" only
    # restates the original text in place.
    n_kept = len(prompts)
    kept_finish = []
    for c in range(n_kept):
        ft = getattr(getattr(outputs[c], "metrics", None), "finished_time", None)
        if ft is not None:
            kept_finish.append(t_inf_start_rel + max(0.0, ft - wall_start))
        else:
            kept_finish.append(t_inf_start_rel
                                + (c + 1) / max(1, n_kept) * inference_time)
    final_t = kept_finish[-1] if kept_finish else (t_inf_start_rel + inference_time)
    node_time = []
    j = 0
    for keep in keep_mask:
        if keep:
            node_time.append(kept_finish[j])
            j += 1
        else:
            node_time.append(final_t)
    return out_soup, node_time, diag


def translate_html_(bs_input, args, timer):
    """Bucket-instrumented orchestrator. Returns (translated_soup, anim_html,
    timeline, diag)."""
    diag = {}
    if args.translation_approach == "qa":
        assert args.seq2seq_model == False, "QA translation does not support seq2seq model"

        with timer.bucket(PREPROCESSING):
            soup = BeautifulSoup(bs_input, 'html.parser')
        with timer.bucket(PROMPT_PROCESSING):
            clean_soup = clone(soup)
            c2o = get_clean_to_original_map(clean_soup, soup)
        with timer.bucket(PREPROCESSING):
            if args.no_clean == False:
                clean_soup = clean(clean_soup, as_new_bs=False)
        with timer.bucket(PROMPT_PROCESSING):
            id_to_tag_map = add_id(clean_soup)

        translation_result, pointer_finish_ts, tok_stats = translate_soup_batch_qa(
            clean_soup, id_to_tag_map, args, timer)
        diag.update(tok_stats)

        # QA could not run (a block prefix exceeds the context window). Flag &
        # skip: emit a row with qa_prompt_too_long=True; the aggregator drops it
        # and compares speedup on the page subset common to both approaches.
        if tok_stats.get("qa_prompt_too_long"):
            diag["structure_mismatch"] = False
            diag["num_pointers"] = len(id_to_tag_map)
            diag["non_termination"] = float("nan")
            diag["required_output_tokens"] = float("nan")
            diag["prompt_too_long"] = False
            diag["full_estimated"] = False
            return soup, str(soup), [], diag

        applied = {}
        structure_mismatch = False
        with timer.bucket(POST_PROCESSING):
            try:
                applied = apply_translation(id_to_tag_map, c2o, translation_result)
            except Exception as e:
                structure_mismatch = True
                print("apply_translation failed:", repr(e))

        timeline = [{"tid": pid, "orig": o, "trans": tr,
                     "t": pointer_finish_ts[pid]}
                    for pid, (o, tr) in applied.items() if pid in pointer_finish_ts]
        diag["structure_mismatch"] = structure_mismatch
        diag["num_pointers"] = len(id_to_tag_map)
        diag["non_termination"] = float("nan")
        diag["required_output_tokens"] = float("nan")
        diag["prompt_too_long"] = False
        diag["full_estimated"] = False
        return soup, str(soup), timeline, diag
    elif args.translation_approach == "tag_json":
        # tag_json: cleansed -> tag every text node -> chunked JSON output
        with timer.bucket(PREPROCESSING):
            soup = BeautifulSoup(bs_input, 'html.parser')
        with timer.bucket(PROMPT_PROCESSING):
            clean_soup = clone(soup)
            c2o = get_clean_to_original_map(clean_soup, soup)
        with timer.bucket(PREPROCESSING):
            if args.no_clean == False:
                clean_soup = clean(clean_soup, as_new_bs=False)
        tj_soup, node_time, tj_diag = translate_soup_batch_tag_json(
            clean_soup, args, timer)
        diag.update(tj_diag)
        diag.setdefault("num_pointers", float("nan"))
        # untag preserves structure by construction; verify and (if so) re-insert
        # the translations into the original markup, like Seq2Seq.
        with timer.bucket(POST_PROCESSING):
            final_soup, consistent, pairs = _reinsert_into_original(
                clean_soup, tj_soup, soup, c2o)
            timeline = [{"tid": k, "orig": o, "trans": tr,
                         "t": node_time[min(k, len(node_time) - 1)] if node_time else 0.0}
                        for k, (o, tr) in enumerate(pairs)]
        diag["structure_mismatch"] = not consistent
        diag["num_pointers"] = len(pairs)
        return final_soup, str(final_soup), timeline, diag
    elif args.translation_approach == "contextless":
        # contextless: cleansed -> per-text-node translation, no context
        with timer.bucket(PREPROCESSING):
            soup = BeautifulSoup(bs_input, 'html.parser')
        with timer.bucket(PROMPT_PROCESSING):
            clean_soup = clone(soup)
            c2o = get_clean_to_original_map(clean_soup, soup)
        with timer.bucket(PREPROCESSING):
            if args.no_clean == False:
                clean_soup = clean(clean_soup, as_new_bs=False)
        cl_soup, node_time, cl_diag = translate_soup_batch_contextless(
            clean_soup, args, timer)
        diag.update(cl_diag)
        diag.setdefault("num_pointers", float("nan"))
        with timer.bucket(POST_PROCESSING):
            final_soup, consistent, pairs = _reinsert_into_original(
                clean_soup, cl_soup, soup, c2o)
            timeline = [{"tid": k, "orig": o, "trans": tr,
                         "t": node_time[min(k, len(node_time) - 1)] if node_time else 0.0}
                        for k, (o, tr) in enumerate(pairs)]
        diag["structure_mismatch"] = not consistent
        diag["num_pointers"] = len(pairs)
        return final_soup, str(final_soup), timeline, diag
    else:
        # seq2seq approach
        with timer.bucket(PREPROCESSING):
            soup = BeautifulSoup(bs_input, 'html.parser')
        with timer.bucket(PROMPT_PROCESSING):
            clean_soup = clone(soup)
            c2o = get_clean_to_original_map(clean_soup, soup)
        with timer.bucket(PREPROCESSING):
            # full_dirty deliberately skips clean(): the Seq2Seq baseline is
            # charged with the RAW uncleaned HTML (scripts/styles/base64),
            # reproducing the paper-style raw-HTML full-doc estimate.
            if args.no_clean == False and args.seq2seq_mode != "full_dirty":
                clean_soup = clean(clean_soup, as_new_bs=False)

        s2s_soup, node_time, s2s_diag = translate_soup_batch_seq2seq_generator(
            clean_soup, args, timer)
        diag.update(s2s_diag)
        diag.setdefault("num_pointers", float("nan"))

        if s2s_diag.get("prompt_too_long"):
            diag["structure_mismatch"] = False
            return s2s_soup, str(s2s_soup), [], diag

        # if Seq2Seq preserved the structure, re-insert into the original markup;
        # otherwise keep the cleansed translated output. The animation plays on
        # whichever was chosen (the translated output), not on the cleansed input.
        with timer.bucket(POST_PROCESSING):
            final_soup, consistent, pairs = _reinsert_into_original(
                clean_soup, s2s_soup, soup, c2o)
            timeline = [{"tid": k, "orig": o, "trans": tr,
                         "t": node_time[min(k, len(node_time) - 1)] if node_time else 0.0}
                        for k, (o, tr) in enumerate(pairs)]
        diag["structure_mismatch"] = not consistent
        return final_soup, str(final_soup), timeline, diag


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='HTML Translation Script')
    parser.add_argument('--no-cache', action='store_true', default=False,
                      help='Enable/disable caching (default: True)')
    parser.add_argument('--seq2seq-model', action='store_true', default=False,
                      help='specify whether `model-name` is a seq2seq model (default: False)')
    parser.add_argument('--model-name', type=str,
                      default='meta-llama/Llama-3.1-8B-Instruct',
                      help='Model name to use for translation')
    parser.add_argument('--translation-approach', type=str,
                      default='qa',
                      choices=['qa', 'seq2seq', 'tag_json', 'contextless'],
                      help='Translation approach to use')
    parser.add_argument('--seq2seq-mode', type=str,
                      default='chunked',
                      choices=['full', 'chunked', 'full_dirty'],
                      help='Seq2Seq baseline mode (default: chunked). '
                           'full_dirty = full-document on the RAW uncleaned HTML '
                           '(reproduces the paper-style raw-HTML estimate).')
    parser.add_argument('--lang-pair', type=str,
                      default='en2ko',
                      help='language-pair subdir under preprocessed-survey-data')
    parser.add_argument('--context-size-in-tags', type=int,
                      default=32,
                      help='Context size in number of tags')
    parser.add_argument('--num-parallel-cache-blocks', type=int,
                    default=16,
                    help='Number of parallel cache blocks')
    parser.add_argument('--split-side-size', type=int,
                    default=10,
                    help='left/right num characters of contexts')
    parser.add_argument('--seq2seq-cap-margin', type=float,
                    default=1.1,
                    help='max_new_tokens cap = required_output_tokens * margin')
    parser.add_argument('--seq2seq-chunk-size', type=int,
                    default=1024,
                    help='characters per chunk in chunked Seq2Seq mode')
    parser.add_argument('--token-expansion-ratio', type=float,
                    default=None,
                    help='override the EN->KO token expansion ratio r')

    parser.add_argument('--translate-all', type=bool,
                    default=False,
                    help='translate all html files')

    parser.add_argument('--page-start', type=int,
                    default=0,
                    help='first page index (inclusive) for --translate-all; for GPU sharding')
    parser.add_argument('--page-end', type=int,
                    default=73,
                    help='last page index (exclusive) for --translate-all; for GPU sharding')
    parser.add_argument('--num-shards', type=int, default=1,
                    help='round-robin GPU sharding: total shard count')
    parser.add_argument('--shard-id', type=int, default=0,
                    help='round-robin GPU sharding: this process handles pages '
                         'where (page_index %% num_shards == shard_id)')

    parser.add_argument('--no-clean', action='store_true', default=False,
                      help='Enable/disable cleaning (default: False)')

    parser.add_argument('--src', type=str,
                help='source html file path')
    parser.add_argument('--dst', type=str,
                help='target html file path')

    parser.add_argument('--quantization', type=str,
                default=None,
                help='quantization to use')

    parser.add_argument('--gpu-memory-utilization', type=float,
                default=0.95,
                help='gpu memory utilization (keep identical across approaches)')

    parser.add_argument('--max-num-seqs', type=int,
                default=40,
                help='max number of sequences')

    parser.add_argument('--tokenizer-name', type=str,
                default=None,
                help='tokenizer name')

    parser.add_argument('--api', action='store_true', default=False,
                help='run against an OpenAI-compatible API endpoint instead of '
                     'a local vLLM engine; --model-name is then the API model id')

    return parser.parse_args()


def _decode_rate_sample(lang_pair, tokenizer):
    """Build a representative Seq2Seq prompt from a real cleansed HTML chunk
    for the decode-throughput measurement."""
    for i in range(73):
        p = f"./data/survey/preprocessed-survey-data/{lang_pair}/{i}/cleaned.html"
        if os.path.exists(p):
            soup = clean(BeautifulSoup(open(p).read(), 'html.parser'), as_new_bs=True)
            chunk = str(soup.prettify())[:SEQ2SEQ_CHUNK_SIZE]
            messages = [{"role": "user", "content": SEQ2SEQ_PROMPT_TEMPLATE + chunk}]
            return tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
    return SEQ2SEQ_PROMPT_TEMPLATE + "<html><body><p>sample</p></body></html>"


def main():
    global model, tokenizer, API_MODE
    args = parse_args()
    API_MODE = args.api
    print("[***] args.no-clean:", args.no_clean, "| API_MODE:", API_MODE)
    args.resolved_r = expansion_ratio_for(args.tokenizer_name or args.model_name,
                                          gen_params.TOKEN_EXPANSION_RATIO,
                                          args.token_expansion_ratio)
    print(f"[***] EN->KO token expansion ratio r = {args.resolved_r}")

    # ---- bucket 1: model serving (one-time) ----
    ms_timer = BucketTimer()
    with ms_timer.bucket(MODEL_SERVING):
        if API_MODE:
            model, tokenizer = setup_api_model(args.model_name, args.tokenizer_name)
            print(f"[***] API endpoint: {model.client.base_url} "
                  f"model={model.model_name}")
        else:
            model, tokenizer = setup_vllm_model(args.model_name,
                                                gpu_memory_utilization=args.gpu_memory_utilization,
                                                enforce_eager=True, max_num_seqs=args.max_num_seqs,
                                                enable_prefix_caching=not args.no_cache,
                                                tokenizer_name=args.tokenizer_name)
    model_serving_time = ms_timer.buckets[MODEL_SERVING]
    print(f"[***] model serving time: {model_serving_time:.2f}s (one-time)")

    model_name = args.model_name.strip("./").replace("/", "_")
    approach = args.translation_approach
    mode = args.seq2seq_mode if approach == "seq2seq" else "none"
    os.makedirs("./results/eswa_rebuttal/speed", exist_ok=True)

    # ---- decode throughput (Seq2Seq diagnostic, 2 conditions) ----
    # only shard 0 measures decode rate, to avoid 8x redundant measurement
    # when sharding across GPUs.
    if approach == "seq2seq" and args.shard_id == 0 and not API_MODE:
        sample = _decode_rate_sample(args.lang_pair, tokenizer)
        rates = measure_decode_rate(model, sample, args.max_num_seqs)
        print(f"[***] decode rate: {rates}")
        pd.DataFrame([{"mode": "single", "batch_condition": "batch1",
                       "tokens_per_sec": rates.get("single")},
                      {"mode": "chunked/full", "batch_condition": f"batch{args.max_num_seqs}",
                       "tokens_per_sec": rates.get("saturated")}]).to_csv(
            f"./results/eswa_rebuttal/speed/{model_name}_{mode}_decode_rate.csv", index=False)

    if args.translate_all:
        data_list = []
        for i in range(args.page_start, min(73, args.page_end)):
            # round-robin shard filter: this process only handles its own pages
            if i % args.num_shards != args.shard_id:
                continue
            # Input is the raw (non-cleaned) page. The pipeline cleanses a copy
            # internally for prompting; QA re-inserts translations back into
            # THIS original markup, so the saved HTML is the non-cleaned page.
            PATH = f"./data/survey/preprocessed-survey-data/{args.lang_pair}/{i}/original.html"
            if not os.path.exists(PATH):
                continue
            HTML = open(PATH).read()

            timer = BucketTimer()
            timer.add(MODEL_SERVING, model_serving_time)
            u_before = dict(model.usage) if API_MODE else None
            timer.start_e2e()
            out_soup, anim_html, timeline, diag = translate_html_(HTML, args, timer)
            timer.stop_e2e()

            row = timer.finalize()
            row.update({"id": i, "approach": approach, "seq2seq_mode": mode})
            row.update(diag)
            # ---- API: real billed per-page token usage + LiteLLM cost ----
            if API_MODE:
                u = model.usage
                d = {k: u[k] - u_before[k] for k in u}
                row["api_calls"] = d["calls"]
                row["api_prompt_tokens"] = d["prompt_tokens"]
                row["api_cached_tokens"] = d["cached_tokens"]
                row["api_completion_tokens"] = d["completion_tokens"]
                row["api_reasoning_tokens"] = d["reasoning_tokens"]
                row["api_cost"] = d["cost"]
                row["api_cost_calls"] = d["cost_calls"]
                print(f"  [page {i}] API calls={d['calls']} "
                      f"prompt={d['prompt_tokens']:,} (cached={d['cached_tokens']:,}) "
                      f"completion={d['completion_tokens']:,} "
                      f"cost=${d['cost']:.4f} | running total ${u['cost']:.4f}",
                      flush=True)
            # estimated full-doc pages: e2e is synthetic (scaled per-chunk
            # time), not the measured wall-clock of the single chunk run.
            # Set e2e = sum of buckets, etc = 0 (this row IS an estimate).
            if diag.get("full_estimated"):
                s26 = (row["t_preprocessing"] + row["t_prompt_processing"]
                       + row["t_cache_management"] + row["t_inference"]
                       + row["t_post_processing"])
                row["t_e2e"] = s26
                row["sum_buckets_2_6"] = s26
                row["t_etc"] = 0.0
                row["etc_fraction"] = 0.0
            data_list.append(row)
            print(f"[page {i}] e2e={row['t_e2e']:.2f}s etc_frac={row['etc_fraction']:.3f}")

            # Per-page output folder: translated HTML + animation. Resources
            # (images/css/js) are referenced via a co-located `resource/`
            # symlink to the raw page directory so the saved HTML works both
            # in-place AND after a dereferencing download (rsync -L / cp -rL /
            # tar -h). `base href="./"` resolves relative refs against the
            # translated.html's own folder -- portable across machines.
            page_out_dir = f"./results/eswa_rebuttal/speed/pages/{model_name}_{approach}_{mode}/{i}"
            os.makedirs(page_out_dir, exist_ok=True)
            resource_link = os.path.join(page_out_dir, "resource")
            if not os.path.lexists(resource_link):
                raw_resource = os.path.abspath(
                    f"./data/survey/raw-survey-data/{args.lang_pair}/{i}/resource")
                if os.path.isdir(raw_resource):
                    rel_target = os.path.relpath(raw_resource, page_out_dir)
                    os.symlink(rel_target, resource_link)
            base_href = "./"
            with open(f"{page_out_dir}/translated.html", "w") as f:
                f.write(inject_base_href(str(out_soup), base_href))

            anim = build_animation_html(anim_html, timeline,
                                        title=f"{approach}/{mode} page {i}",
                                        base_href=base_href)
            with open(f"{page_out_dir}/anim.html", "w") as f:
                f.write(anim)

        df = pd.DataFrame(data_list)
        # shard id in the filename so sharded runs (one per GPU) do not
        # overwrite each other. aggregate_buckets.py globs *_buckets.csv and
        # concatenates all shards.
        out_csv = (f"./results/eswa_rebuttal/speed/{model_name}_{approach}_{mode}"
                   f"_{args.no_cache}_{args.context_size_in_tags}"
                   f"_shard{args.shard_id}of{args.num_shards}_buckets.csv")
        df.to_csv(out_csv, index=False)
        print(f"[***] wrote {out_csv}")

        # ---- API: per-stage measured-cost summary (this run = one method) ----
        if API_MODE:
            u = model.usage
            n_pages = len(data_list)
            print("\n" + "=" * 78)
            print(f"MEASURED API COST  model={model_name}  approach={approach}/{mode}"
                  f"  pages={n_pages}")
            print("=" * 78)
            print(f"  API calls         : {u['calls']:,}  "
                  f"(priced {u['cost_calls']:,})")
            print(f"  prompt tokens     : {u['prompt_tokens']:,}  "
                  f"(cached {u['cached_tokens']:,})")
            print(f"  completion tokens : {u['completion_tokens']:,}  "
                  f"(reasoning {u['reasoning_tokens']:,})")
            print(f"  TOTAL COST        : ${u['cost']:.4f}   "
                  f"(${u['cost']/max(1,n_pages):.4f}/page)")
            print("=" * 78)
            import json as _json
            cost_path = (f"./results/eswa_rebuttal/speed/{model_name}_{approach}_"
                         f"{mode}_shard{args.shard_id}of{args.num_shards}_apicost.json")
            with open(cost_path, "w") as f:
                _json.dump({"model": model_name, "approach": approach, "mode": mode,
                            "pages": n_pages, "usage": u}, f, indent=2)
            print(f"[***] wrote {cost_path}")

    else:
        assert args.src is not None and args.dst is not None, "src and dst must be provided"

        HTML = open(args.src).read()
        timer = BucketTimer()
        timer.add(MODEL_SERVING, model_serving_time)
        timer.start_e2e()
        out_soup, anim_html, timeline, diag = translate_html_(HTML, args, timer)
        timer.stop_e2e()
        row = timer.finalize()
        row.update(diag)
        print("buckets:", row)
        base_href = "file://" + os.path.dirname(os.path.abspath(args.src)) + "/"
        with open(args.dst, "w") as f:
            f.write(inject_base_href(str(out_soup), base_href))
        anim = build_animation_html(anim_html, timeline, title=f"{approach}/{mode}",
                                    base_href=base_href)
        with open(args.dst + ".anim.html", "w") as f:
            f.write(anim)


if __name__ == "__main__":
    main()
