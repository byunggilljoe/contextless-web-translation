import os
import sys
import csv
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import json
import pdb
from common.process_html import inject_translate_tag, extract_translate_tag, clean
import configs.gen_params as gen_params
from bs4 import BeautifulSoup
import argparse

TRANSLATE_TAG = gen_params.TRANSLATE_TAG

sample_survey_data_path = os.path.join(os.path.dirname(__file__), "../data/survey/preprocessed-survey-data/zh2ko/webpage_chrome_manual.json")
sample_json_data = json.load(open(sample_survey_data_path, "r"))

def get_translation_approaches(json_data):
    # Get unique translation approaches across all samples
    approaches = set()
    for data in json_data:
        approaches.update(data["translation_approach"].keys())
    return sorted(list(approaches))

parser = argparse.ArgumentParser()
parser.add_argument("--translation_approach", nargs="+", choices=get_translation_approaches(sample_json_data) + ["manual"],
                    help="List of translation approaches to include", default=None)
parser.add_argument("--output_path", type=str, required=True,
                    help="Path to output CSV file")
parser.add_argument("--survey_data_path", type=str, required=True,
                    help="Path to survey data JSON file")
parser.add_argument("--do_clean", type=bool, default=True,
                    help="Whether to clean the HTML")

parser.add_argument("--translation_language_pair", type=str, default="en2ko", choices=["en2ko", "es2ko", "de2ko", "zh2ko", "en2de", "en2es", "en2zh", "de2en", "de2es", "de2zh", "es2en", "es2de", "es2zh", "zh2en", "zh2de", "zh2es"],
                    help="Translation language pair")

args = parser.parse_args()
json_data = json.load(open(args.survey_data_path, "r"))
OUTPUT_PATH = args.output_path
QUESTION_CNT = 0
FAIL_INSERTION_LIST = []


SOURCE_LANGUAGE = {"en":"english", "es":"spanish", "de":"germany", "zh":"chinese"}[args.translation_language_pair.split("2")[0]]
DESTINATION_LANGUAGE = {"en":"english", "es":"spanish", "de":"germany", "zh":"chinese"}[args.translation_language_pair.split("2")[1]]



# {'sample_id': 524, 'file_original': 'data/survey//raw-survey-data/72/original.html', 'URL': 'https://www.gsmarena.com/', 'original': 'SHARP', 'translation_approach': {'chrome': '날카로운', 'huggingface_qwen_2_5_coder': '샤프', 'huggingface_llama_3_1_8b': '샤프', 'azure_gpt_4o_mini': '샤프', 'azure_gpt_4o': '샤프'}, 'translation_manual': 'SHARP', 'exclude_from_user_survey': False, 'survey_indices': [False, False, False]}

translation_window_left = 512
translation_window_right = 512
EXAMPLE = "(e.g., “보내기” vs. “보내다.” Keep one consistent in the UI?)" if DESTINATION_LANGUAGE == "ko" else ""

CRITERIA = \
f"""1. Meaning is incorrect; the cause of the error cannot be determined.
2. Meaning is incorrect.
3. Meaning is conveyed correctly in context.
4. Meaning is correct, but the {DESTINATION_LANGUAGE} expression is awkward. {EXAMPLE}
5. Meaning is correct and the {DESTINATION_LANGUAGE} expression sounds natural in the web context.
"""

PROMPT = \
f"""Your are a {SOURCE_LANGUAGE}-to-{DESTINATION_LANGUAGE} translator.
Here is 
    - a fragment HTML of a webpage where the translation target string is marked with <{TRANSLATE_TAG}_0> tag.
    - Translation of the target string in {DESTINATION_LANGUAGE}.
    - Evaluation criteria for the translation.
You are required to score the {DESTINATION_LANGUAGE} translation according to the criteria.

== Fragment of a webpage ==
{{HTML_STR}}

== {DESTINATION_LANGUAGE} Translation of the target string ==
{{TRANSLATION_RESULT}}

== Evaluation Criteria ==
{CRITERIA}

== Score ==
Your answer should be a number between 1 and 5.
Finish your answer with a newline only with score in the format of "score: <score>".
"""



if os.path.dirname(OUTPUT_PATH) != "":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    writer = csv.writer(f)

    writer.writerow(["question_ID", "question_text", "evaluation_criteria", "tags", "aux_data"])
    for data in json_data:
        soup = BeautifulSoup(open(data["file_original"], "r").read(), "html.parser")
        if args.do_clean:
            soup = clean(soup)
        target_string = data["original"]
        
        html_str = inject_translate_tag(soup, target_string, f"{TRANSLATE_TAG}")
        if html_str is None:
            print(f"Failed to inject translate tag for {target_string}... continue")
            FAIL_INSERTION_LIST.append(data["sample_id"])
            continue
        
        html_str = extract_translate_tag(html_str, f"{TRANSLATE_TAG}_0", translation_window_left, translation_window_right)
        translation_approach_list = args.translation_approach if args.translation_approach is not None else data["translation_approach"].keys()
        for approach in translation_approach_list:
            if approach == "manual":
                continue
            print("-----")
            final_prompt = PROMPT.format(HTML_STR=html_str, TRANSLATION_RESULT=data["translation_approach"][approach])
            print(final_prompt)
            aux_data = {
                "sample_id": data["sample_id"],
                "original": data["original"],
                "translation_approach": approach,
                "translation_result": data["translation_approach"][approach],
            }

            # question_ID, question_text, evaluation_criteria, tags, aux_data
            csv_row = [
                QUESTION_CNT,
                final_prompt,
                "", # NO EVALUATION CRITERIA
                "[\"web_translation\"]", # No Tags
                json.dumps(aux_data, ensure_ascii=False)
            ]
            writer.writerow(csv_row)
            QUESTION_CNT += 1
        

        if args.translation_approach is not None and "manual" not in args.translation_approach:
            continue

        final_prompt = PROMPT.format(HTML_STR=html_str, TRANSLATION_RESULT=data["translation_manual"])
        aux_data = {
            "sample_id": data["sample_id"],
            "original": data["original"],
            "translation_approach": "manual",
            "translation_result": data["translation_manual"],
        }
        csv_row = [QUESTION_CNT, final_prompt, "", "[\"web_translation\"]", json.dumps(aux_data, ensure_ascii=False)]
        writer.writerow(csv_row)
        QUESTION_CNT += 1

print("FAILED INSERTION LIST:", FAIL_INSERTION_LIST)
print("len(FAILED INSERTION LIST):", len(FAIL_INSERTION_LIST))
print("QUESTION_CNT:", QUESTION_CNT)