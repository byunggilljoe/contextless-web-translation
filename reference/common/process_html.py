from bs4 import BeautifulSoup, Comment

def clean(soup, as_new_bs=True):
    mc = soup.find('html')
    if mc is None:
        mc = soup.find('body')
    assert mc is not None
    
    #import pdb; pdb.set_trace()

    headerlink = mc.find('a', class_='headerlink')
    if headerlink is not None:
        headerlink.decompose()

    #for tag in mc.find_all('div'):
    for tag in mc.find_all():
        # if tag.string is None:
        direct_text = ''.join(child for child in tag.children if isinstance(child, str)).strip()
        if direct_text == '' and tag.find(True):
            # 자기 자신에게는 텍스트가 없고 자식 태그가 있는 경우 자식만 보존
            tag.unwrap()
        elif len(tag.get_text(strip=True)) == 0:
            # 자식 포함 모두 텍스트가 없는 경우 tag 삭제
            tag.decompose()
        else:
            pass
            # print(tag)
            # print("----->", tag.get_text())
            # print("----->", tag.string)


    for tag in mc.find_all('style'):
        tag.decompose()
    
    for tag in mc.find_all('path'):
        tag.decompose()

    for tag in mc.find_all('svg'):
        tag.decompose()
    
    for tag in mc.find_all('script'):
        tag.decompose()

    # Remove comments
    for comment in mc.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove empty or nested tags
    for tag in mc.find_all():
        # Remove all attributes except essential ones
        allowed_attrs = [] # ['href', 'id']
        for attr in list(tag.attrs):
            if attr not in allowed_attrs:
                del tag.attrs[attr]
    
    if as_new_bs:
        new_soup = BeautifulSoup(str(mc), 'html.parser')
        return new_soup
    else:
        return mc


def extract_text_only(soup, seperator=" "):
    """
    HTML에서 모든 태그를 제거하고 눈에 보이는 텍스트만 추출하는 함수
    
    Args:
        soup: BeautifulSoup 객체 또는 HTML 문자열
    
    Returns:
        str: 태그가 제거된 텍스트만 포함된 문자열
    """
    # HTML 문자열인 경우 BeautifulSoup 객체로 변환
    if isinstance(soup, str):
        soup = BeautifulSoup(soup, 'html.parser')
    
    # html 또는 body 태그 찾기
    mc = soup.find('html')
    if mc is None:
        mc = soup.find('body')
    if mc is None:
        mc = soup
    
    # 스크립트, 스타일, SVG 등 제거
    for tag in mc.find_all('style'):
        tag.decompose()
    
    for tag in mc.find_all('path'):
        tag.decompose()

    for tag in mc.find_all('svg'):
        tag.decompose()
    
    for tag in mc.find_all('script'):
        tag.decompose()

    # 주석 제거
    for comment in mc.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # 모든 태그를 제거하고 텍스트만 추출
    text = mc.get_text(separator=seperator, strip=True)
    
    # 여러 공백을 하나로 정리
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def custom_strip(text):
    # First remove LRM and RLM markers
    text = text.replace('&lrm;', '').replace('\u200e', '')  # Left-to-Right Mark
    text = text.replace('&rlm;', '').replace('\u200f', '')  # Right-to-Left Mark
    # Then strip whitespace
    return text.strip()

def custom_text_clean(text):
    text = text.replace('<!--', '').replace('-->', '')
    text = custom_strip(text.replace(":", '')) # Remove RLM Unicode character
    return text

def read_html_file(html_file):
    """
    Read HTML file with multiple encoding attempts and return BeautifulSoup object
    Returns None if file cannot be read with any encoding
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(html_file, 'r', encoding=encoding) as file:
                html_content = file.read()
                return BeautifulSoup(html_content, "html.parser")
        except UnicodeDecodeError:
            if encoding == encodings[-1]:  # If this was the last encoding to try
                print(f"Failed to read {html_file} with any encoding")
                return None
            continue

def inject_translate_tag(soup, target_string, translate_tag_name_prefix):
    count = 0
    soup = BeautifulSoup(str(soup), "html.parser")
    # 모든 텍스트가 있는 태그를 순회
    for tag in soup.find_all():
        if tag.string:
            # Remove HTML comments and strip whitespace
            cleaned_string = custom_text_clean(str(tag.string))
            if cleaned_string.lower() == target_string.lower():
                # 새로운 TRANSLATE_IT 태그 생성
                translate_tag = soup.new_tag(f"{translate_tag_name_prefix}_{count}")
                translate_tag.string = tag.string
                # 기존 텍스트를 새로운 태그로 교체
                tag.string.replace_with(translate_tag)
                count += 1
    if count == 0:
        return None

    html_str = str(soup.prettify())
    html_str = html_str.replace("&lt;&lt;", "<<")
    html_str = html_str.replace("&gt;&gt;", ">>")

    
    return html_str


def extract_translate_tag(html_str, target_tag_name, translation_window_left=512, translation_window_right=512):
    translate_tag_start = html_str.find(f"<{target_tag_name}>")
    if translate_tag_start != -1:
        translate_tag_end = html_str.find(f"</{target_tag_name}>") + len(f"</{target_tag_name}>")
        
        # Calculate window boundaries
        window_start = max(0, translate_tag_start - translation_window_left)
        window_end = min(len(html_str), translate_tag_end + translation_window_right)
        
        # Extract the substring with the context window
        html_str = html_str[window_start:window_end]
        return html_str

    return None
