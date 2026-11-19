#https://stackoverflow.com/questions/23057631/clone0c98acab374beb118b1301d144b049cd0d827d61element-with-beautifulsoup
import re
import pdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.process_html import clean
import time

import requests
from bs4 import BeautifulSoup, Tag, NavigableString
import dotenv
dotenv.load_dotenv()

import tqdm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from speedup_tests.huggingface_use_cache_test import generate_with_cached_prompt_batch, setup_model
# from llama_cpp_langchain_test import generate_with_cached_prompt_batch as generate_with_cached_prompt_batch_llama


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


def translate_soup_batch_seq2seq_generator(bs_input, args):
    assert False, "seq2seq approach - seq2seq model is not implemented"
    # soup = BeautifulSoup(bs_input, 'html.parser')
    # soup = clean(soup, as_new_bs=False)
    # bs_input = str(soup.prettify())

    global model, tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = setup_model(model_name=args.model_name)
        # model, tokenizer = setup_model(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

    t = time.time()

    llm_func_to_use = generate_with_cached_prompt_batch # generate_with_cached_prompt_batch_llama

    base_prompt_template = """
- Translate all texts in a fragment of HTML into Korean (한글).
- Do not translate HTML tags. (e.g. <div>, <p>, <a>, etc. should be left as they are)
- Thus the output should be consistent with the original HTML.
- Do not provide any explanation or preambles.
- Here is the original HTML:
"""

    SEQ2SEQ_CONTEXT_SIZE = 1024
    sample_list = []
    
    # for i in tqdm.tqdm(range(0, len(bs_input), SEQ2SEQ_CONTEXT_SIZE)):
    #     sample_list.append(bs_input[i:i+SEQ2SEQ_CONTEXT_SIZE])
    cursor = 0
    while cursor < len(bs_input):
        tmp_fragment = bs_input[cursor:cursor+SEQ2SEQ_CONTEXT_SIZE]
        # Check if fragment ends with an incomplete tag
        last_lt_idx = tmp_fragment.rfind('<')
        if last_lt_idx != -1:
            last_gt_idx = tmp_fragment.rfind('>')
            if last_gt_idx < last_lt_idx:
                # Incomplete tag found, truncate to before it
                if last_lt_idx == 0:
                    last_lt_idx = SEQ2SEQ_CONTEXT_SIZE

                tmp_fragment = tmp_fragment[:last_lt_idx]
                sample_list.append(tmp_fragment)
                cursor = cursor + last_lt_idx
                print("tmp_fragment:", cursor, tmp_fragment)
                if last_lt_idx == 0:
                    exit(0) 
                continue
        
        sample_list.append(tmp_fragment)
        cursor += SEQ2SEQ_CONTEXT_SIZE
        print("tmp_fragment:", cursor, tmp_fragment)
    print("[*] sample split completed")
    all_translation = ""
    batch_size = args.batch_size

    for i in tqdm.tqdm(range(0, len(sample_list), batch_size)):
        split_html_str = sample_list[i:i+batch_size]
        base_prompt = base_prompt_template
        following_prompt_list = [(b, "") for b in split_html_str]
        max_new_tokens = max(len(b) for b in split_html_str)

        # model: AutoModelForCausalLM,
        # tokenizer: AutoTokenizer,
        # base_prompt: str,
        # prompt_suffixes: List[str],
        # max_new_tokens: int = 64,
        # batch_size: int = 16,
        # use_cache: bool = True

        translated_result = llm_func_to_use(model, tokenizer, base_prompt=base_prompt,\
                        prompt_suffixes=following_prompt_list, max_new_tokens=max_new_tokens,
                        batch_size=batch_size, use_cache=True)
        
        segment_translation = [tr.split("assistant")[-1] for tr in translated_result]
        segment_translation_str = "".join(segment_translation)
        all_translation += segment_translation_str
        for b, st in zip(split_html_str, segment_translation):
            print("b:", b)
            print("st:", st)
            print("="*100)

    print("Translation time:", time.time() - t)
    return all_translation


model, tokenizer = None, None
def translate_soup_batch_qa(clean_soup, id_to_tag_map, args): 
    # context_size_in_tags, batch_size
    # 64, 64    : 28.54 초
    # 32, 16    : 15    초      <-- best
    # 32, 32    : 21.57 초
    # 32, 8     : 22.76 초
    # 16, 8     : 18.30 초
    use_cache = not args.no_cache
    batch_size = args.batch_size
    context_size_in_tags = args.context_size_in_tags
    split_side_size = args.split_side_size
    split_base_prompt = args.context_size_in_tags > 0

    assert context_size_in_tags >= batch_size
    assert context_size_in_tags % batch_size == 0
    
    global model, tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = setup_model(model_name=args.model_name)
        #model, tokenizer = setup_model(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

    t = time.time()

    llm_func_to_use = generate_with_cached_prompt_batch # generate_with_cached_prompt_batch_llama

    html_str = str(clean_soup.prettify())
    html_str = html_str.replace("&lt;&lt;", "<<")
    html_str = html_str.replace("&gt;&gt;", ">>")

    html_str = html_str.replace("&lt;", "<")
    html_str = html_str.replace("&gt;", ">")

    # chat template을 써서해보기

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


    base_prompt = base_prompt_template.format(html=html_str)

    following_prompt_template = ("""- You need to translate <T_{id}> into Korean (한글)
- 당신의 답변:
""",
"""
<T_{id}> 
""") # (user following, assistant following)

    # model: AutoModelForCausalLM,
    # tokenizer: AutoTokenizer,
    # base_prompt: str,
    # prompt_suffixes: List[str],
    # max_new_tokens: int = 64,
    # batch_size: int = 16,
    # use_cache: bool = True
    

    following_prompt_list = [tuple(t.format(id=id) for t in following_prompt_template)  for id in range(html_str.count("<T_"))]

    batch_size = batch_size             # number of tags (=samples) in batch
    split_size = context_size_in_tags # context size in the number of tags
    translated_result = []
    if split_base_prompt:
        for i in range(0, len(following_prompt_list), split_size):
            split_start_index = max(html_str.index(f"<T_{i}>") - split_side_size, 0)
            try:
                split_end_index = min(html_str.index(f"</T_{i+split_size}>") + len(f"</T_{i+split_size}>") + split_side_size, len(html_str))
            except:
                split_end_index = len(html_str)

            split_html_string = html_str[split_start_index:split_end_index]
            split_base_prompt = base_prompt_template.format(html=split_html_string)

            translated_result_ = llm_func_to_use(model, tokenizer, base_prompt=split_base_prompt,\
                prompt_suffixes=following_prompt_list[i:i+split_size], batch_size=batch_size, use_cache=use_cache)

            translated_result.extend(translated_result_)

    else:
        assert False, "split_base_prompt is False"
        translated_result = llm_func_to_use(model, tokenizer, base_prompt=base_prompt,\
                        prompt_suffixes=following_prompt_list,
                        batch_size=batch_size, use_cache=use_cache)

    TRANSLATION_TIME = time.time() - t
    

    # Post processing translated_result
    post_processed_translated_result = []

    #import pdb; pdb.set_trace()
    for i, elem in enumerate(translated_result):
        # print("[*****]", i, elem)
        matches =list(re.finditer(r"<T_\d+>", elem))
        if not matches:
            raise Exception(f"No match found for {i}th element")
        last_match = matches[-1]
        start_index = last_match.start()
        print(elem[start_index:])
        if "-->" in elem[start_index:]:
            after_arrow = elem[start_index:].split("-->")[-1]
            before_closing_tag = after_arrow.split("</")[0]
            processed_elem = f"<T_{i}>{before_closing_tag}"
        else:
            before_closing_tag = elem[start_index:].split("</")[0]
            processed_elem = before_closing_tag
        
        # 
        post_processed_translated_result.append(processed_elem)

    print("Translation time:", TRANSLATION_TIME)
    return "\n".join(post_processed_translated_result)

def apply_translation(id_to_tag_map, c2o, translation_result):
    elem_list = []
    for elem in translation_result.split("<T_"):
        if len(elem) == 0:
            continue
        id_and_text = elem.split(">")
        if id_and_text[0].isdigit() == False:
            continue
        print("*****", id_and_text)
        elem_id = int(id_and_text[0])
        elem_text = id_and_text[1].strip()
        elem_list.append([elem_id, elem_text])
        if elem_id in id_to_tag_map.keys():
            clean_tag = id_to_tag_map[elem_id]
            original_tag = c2o[clean_tag.to_o_key]

            id_to_tag_map[elem_id].string.replace_with(elem_text)
            direct_text = ''.join(child for child in original_tag.children if isinstance(child, str)).strip()

            if original_tag.string is not None:
                original_tag.string.replace_with(elem_text)
            else:
                TRANSLATED = False
                for child in original_tag.children:
                    print("child:",child, type(child))
                    if isinstance(child, NavigableString) and not isinstance(child,Tag):
                        child.replace_with(elem_text)
                        TRANSLATED = True
                        break
                assert TRANSLATED


def translate_html_(bs_input, args):
    if args.translation_approach == "qa":
        assert args.seq2seq_model == False, "QA translation does not support seq2seq model"

        soup = BeautifulSoup(bs_input, 'html.parser')
        clean_soup = clone(soup)
        c2o = get_clean_to_original_map(clean_soup, soup)
        clean_soup = clean(clean_soup, as_new_bs=False)
        id_to_tag_map = add_id(clean_soup)
        translation_result = translate_soup_batch_qa(clean_soup, id_to_tag_map, args)
        apply_translation(id_to_tag_map, c2o, translation_result)
        print("translation_result:", translation_result)

        return soup
    else:
        # seq2seq approach
        # need to implement preprocessing and post processing if speed comparison is needed.
        # not need to implement them if only cosnsistency check is needed.
        if args.seq2seq_model:
            assert False, "seq2seq approach - seq2seq model is not implemented"
            translated_html = translate_soup_batch_seq2seq_autoencoder(bs_input, args)
            soup = BeautifulSoup(translated_html, 'html.parser')
            return soup
        else:
            translated_html = translate_soup_batch_seq2seq_generator(bs_input, args)
            soup = BeautifulSoup(translated_html, 'html.parser')
            return soup
        

# experiment args:
# - model_name in {meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-Coder-32B-Instruct}
# - approach in {seq2seq_QA, seq2seq_HTML, use_batch, use_cache}
# 

# Experiments
# - Translation time depending on cache (or model quantization)
# - Tag consistency

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
                      choices=['qa', 'seq2seq'],
                      help='Translation approach to use')
    parser.add_argument('--context-size-in-tags', type=int,
                      default=32,
                      help='Context size in number of tags')
    parser.add_argument('--split-side-size', type=int,
                    default=10,
                    help='left/right num characters of contexts')
    parser.add_argument('--batch-size', type=int,
                    default=16,
                    help='batch size')

    parser.add_argument('--translate-all', type=bool,
                    default=False,
                    help='translate all html files')

    parser.add_argument('--src', type=str,
                help='source html file path')
    parser.add_argument('--dst', type=str,
                help='target html file path')

    return parser.parse_args()


def main():
    global model, tokenizer
    args = parse_args()

    # t = time.time()
    # soup = translate_html_(open("./data/test_html4.html").read(), args)
    # print(f"time taken: {time.time() - t}")
    # # print(soup)

    # o_file = open("./tmp/test_html_translated.html", "w")
    # o_file.write(str(soup))
    # o_file.close()
    
    model, tokenizer = setup_model(model_name=args.model_name)

    if args.translate_all:
        data_list = []
        for i in range(0, 73):#39
            PATH = f"./data/survey/preprocessed-survey-data/{i}/cleaned.html"
            # PATH = f"./data/dummy_html.html"
            if os.path.exists(PATH):
                HTML = open(PATH).read()
                model = args.model_name
                model = model.replace("/", "_")
                t = time.time()
                translated_result = translate_html_(HTML, args)
                translation_time = time.time() - t
                print(f"Translation time {i}:", translation_time)
                data_list.append({"id":i, "translation_time":translation_time})
                OUTPUT_PATH = f"./results/qa_translated/{model}_{i}.html"
                with open(OUTPUT_PATH, "w") as f:
                    f.write(str(translated_result))
                    f.close()
        df = pd.DataFrame(data_list)
        df.to_csv(f"./results/qa_translated/{model}_info.csv", index=False)

    else:
        assert args.src is not None and args.dst is not None, "src and dst must be provided"

        HTML = open(args.src).read()
        t = time.time()
        translated_result = translate_html_(HTML, args)
        translation_time = time.time() - t
        print(f"Translation time:", translation_time)
        with open(args.dst, "w") as f:
            f.write(str(translated_result))
            f.close()


if __name__ == "__main__":
    main()
    import pdb
    pdb.set_trace()

