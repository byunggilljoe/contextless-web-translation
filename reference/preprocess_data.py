import os
import sys
import argparse

from common.process_html import read_html_file, custom_text_clean, custom_strip, custom_text_clean, clean

import csv
import tqdm

import json
from bs4 import BeautifulSoup
from bs4 import Comment
from email import parser, policy
from email.message import EmailMessage
DATA_PATH = "data/survey/"

def process_csv(csv_file_path, translation_language_pair):
    result = []
    previous_first_col = None  # 이전 row의 첫 번째 컬럼 값을 저장할 변수
    previous_url_col = None
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        head = next(csv_reader)  # 헤더 row 건너뛰기
        for idx, row in enumerate(csv_reader, 1):
            # 현재 row의 첫 번째 컬럼이 비어있으면 이전 값 사용
            current_first_col = row[0] if row[0].strip() else previous_first_col
            current_url_col = row[1] if row[1].strip() else previous_url_col
            
            language_pair_split = translation_language_pair.split('2')
            # only 2ko directory contains original.html
            language_pair_contains_original_file = language_pair_split[0]+"2ko"
            chrome_translation_index = 3 if language_pair_split[1] == "ko" else head.index("2"+language_pair_split[1])
            chrome_translation = row[chrome_translation_index]

            translation_manual = row[4] if language_pair_split[1] == "ko" else "N/A"
            exclude_from_user_survey = True if language_pair_split[1] == "ko" and row[7] == "TRUE"  else False
            item = {
                'sample_id': idx,
                'file_original': f"{DATA_PATH}/raw-survey-data/{language_pair_contains_original_file}/{current_first_col}/original.html",
                'URL': current_url_col,
                'original': row[2],
                'translation_approach': {'chrome': chrome_translation},
                'translation_manual': translation_manual,
                'exclude_from_user_survey': exclude_from_user_survey,
                'survey_indices':[r == "TRUE" for r in row[8:]]
            }
            print("exclude_from_user_survey", item['exclude_from_user_survey'], row[7])
            result.append(item)
            # 다음 iteration을 위해 현재 값 저장
            previous_first_col = current_first_col
            previous_url_col = current_url_col
    return result




def highlight_original_text(soup, original_text):
    if not original_text.strip():  # 빈 텍스트인 경우 처리하지 않음
        return soup
    
    # 원본 텍스트를 소문자로 변환
    original_text_lower = custom_text_clean(original_text).lower()
    
    # CSS 애니메이션을 위한 style 태그 생성
    style_tag = soup.new_tag('style')
    style_tag.string = '''
        @keyframes blink {
            0% { 
                border-color: red;
                transform: scale(1);
            }
            50% { 
                border-color: red;
                transform: scale(1.2);
            }
            100% { 
                border-color: red;
                transform: scale(1);
            }
        }
        .llm_translation_target {
            border: 2px solid red;
            padding: 2px;
            animation: blink 2s infinite;
            transform-origin: center; /* 확대/축소의 중심점을 요소의 중앙으로 설정 */
            display: inline-block;
        }
    '''
    
    # style 태그를 head에 추가
    if soup.head:
        soup.head.append(style_tag)
    else:
        head_tag = soup.new_tag('head')
        head_tag.append(style_tag)
        if soup.html:
            soup.html.insert(0, head_tag)
        else:
            soup.insert(0, head_tag)
    
    EDIT_ANY = False
    # 모든 텍스트 노드를 순회
    # for element in soup.find_all(string=True):
    #     # direct_text = ''.join(child for child in element.children if isinstance(child, str)).strip()
    #     # 현재 텍스트 노드의 내용을 소문자로 변환하여 비교
    #     # element_text_lower = custom_text_clean(direct_text).lower()
    #     element_text_lower = custom_text_clean(element.string).lower()

    for element in soup.find_all():
        text_children = [child for child in element.children  if isinstance(child, str) and not isinstance(child, Comment)]

        direct_text = ''.join(child for child in element.children 
                            if isinstance(child, str) and not isinstance(child, Comment)).strip()
        # 현재 텍스트 노드의 내용을 소문자로 변환하여 비교
        element_text_lower = custom_text_clean(direct_text).lower()
        
        if "dep" in original_text_lower:
            print("[*****]", original_text_lower, "||",  direct_text)

        if original_text_lower == element_text_lower:
            # 새로운 span 태그 생성
            highlight_tag = soup.new_tag('span')
            highlight_tag['class'] = 'llm_translation_target'
            
            # option 태그인 경우 상위 select 태그를 감싸기
            if element.parent.name == 'option':
                select_parent = element.parent.find_parent('select')
                if select_parent:
                    select_parent.wrap(highlight_tag)
                    EDIT_ANY = True
                continue
            
            # 그 외의 경우 원본 텍스트 노드를 새로운 span 태그로 감싸기
            for tc in text_children:
                tc.wrap(highlight_tag)

            EDIT_ANY = True
    assert EDIT_ANY, "No target text found"
    return soup


def extract_html_from_mhtml(file_path):
    """
    email 모듈을 사용하여 MHTML 파일에서 HTML 컨텐츠를 추출하여 BeautifulSoup 객체를 반환합니다.
    """
    # 이메일 파서 생성
    email_parser = parser.Parser(policy=policy.default)
    
    # MHTML 파일 읽기 - 'utf-8' 대신 'latin-1'이나 'cp949'로 시도
    with open(file_path, 'r', encoding='latin-1') as file:
        message: EmailMessage = email_parser.parse(file)
    
    # HTML 컨텐츠 찾기
    for part in message.walk():
        if part.get_content_type() == 'text/html':
            # 인코딩 정보 가져오기
            charset = part.get_content_charset() or 'utf-8'
            html_content = part.get_content()
            return BeautifulSoup(html_content, 'html.parser', from_encoding=charset)
    
    return BeautifulSoup(message.get_body().get_content(), 'html.parser')

def save_html(soup, original_file_path, new_file_name):


    source_resource_folder = os.path.join(os.path.dirname(original_file_path), 'resource')
    source_resource_folder_exists = os.path.exists(source_resource_folder)
    
    target_save_path = original_file_path.replace('raw-survey-data', 'preprocessed-survey-data')
    save_path = target_save_path.replace('original.html', new_file_name)
    save_dir = os.path.dirname(save_path)
    target_resource_folder = os.path.join(save_dir, 'resource')
    
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    if source_resource_folder_exists:
        try:
            os.symlink(os.path.abspath(source_resource_folder), os.path.abspath(target_resource_folder))
        except FileExistsError:
            pass
    else:
        assert False, "resource folder not found"

    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(str(soup))



def validate_html(soup, target_string):
    soup = BeautifulSoup(str(soup), "html.parser")
    # 모든 텍스트가 있는 태그를 순회
    for tag in soup.find_all():
        direct_text = ''.join(child for child in tag.children 
                    if isinstance(child, str) and not isinstance(child, Comment)).strip()
        cleaned_string = custom_text_clean(direct_text)

        if "dep" in target_string.lower() and "dep" in cleaned_string.lower():
            print("[*****]", target_string, "|00|", cleaned_string)
        
        if len(cleaned_string) > 0:
            # Remove HTML comments and strip whitespace
            if cleaned_string.lower() == target_string.strip().lower():
                return tag
    
    for tag in soup.find_all(string=True):
        cleaned_string = custom_text_clean(tag.string)
        if "dep" in cleaned_string:
            print(target_string, "vs", cleaned_string)
        if cleaned_string.lower() == target_string.strip().lower():
            print("validated:", target_string)
            return tag

    return None



def main():
    parser = argparse.ArgumentParser(description='Preprocess survey data for translation evaluation')
    parser.add_argument(
        '--language-pair', '-l',
        type=str,
        required=True,
        help='Translation language pair (e.g., zh2ko, en2ko, de2ko)'
    )
    args = parser.parse_args()
    
    translation_language_pair = args.language_pair
    source_language = args.language_pair.split('2')[0]
    csv_file_path = f'{DATA_PATH}/raw-survey-data/csv_files/webpage_chrome_manual_{source_language}.csv'

    data = process_csv(csv_file_path, translation_language_pair)


    # 결과 확인
    preprocessed_data = []
    failed_list = []
    for item in tqdm.tqdm(data):
        html_file = item['file_original']
        # soup = extract_html_from_mhtml(html_file)

        print(html_file)

            
        # BeautifulSoup으로 HTML 파싱
        soup = read_html_file(html_file)
        if soup is None:
            failed_list.append(item)
            continue

        # 1) original
        original_path = html_file #.replace('raw-survey-data', 'preprocessed-survey-data')
        # 2) clean
        clean_soup = clean(BeautifulSoup(str(soup), 'html.parser'))
        
        print("Validity Check |", item['original'])
        if validate_html(clean_soup, item['original']) is None:
            failed_list.append(item)
            continue

        
        save_html(soup, original_path, 'original.html')
        save_html(clean_soup, original_path, 'cleaned.html')

        # 3) Apply target string indicator - original
        original_rect_content = highlight_original_text(BeautifulSoup(str(soup), 'html.parser'), item['original'])
        save_html(original_rect_content, original_path, f'rect/original_rect_{item["sample_id"]}.html')

        # 4) Apply target string indicator - cleaned
        cleaned_rect_content = highlight_original_text(BeautifulSoup(str(clean_soup), 'html.parser'), item['original'])
        save_html(cleaned_rect_content, original_path, f'rect/cleaned_rect_{item["sample_id"]}.html')

        preprocessed_data.append(item)
    
    parent_path = f'{DATA_PATH}/preprocessed-survey-data/{translation_language_pair}'
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    json.dump(preprocessed_data, open(f'{parent_path}/webpage_chrome_manual.json', 'w', encoding='utf-8'),
                                 ensure_ascii=False, indent=4)
    
    print("====failed_list====")
    for i, item in enumerate(failed_list):
        print(i, item['sample_id'], item['file_original'], item['original'])
if __name__ == "__main__":
    main()