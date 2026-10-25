"""
This file was written with assistance from generative AI. 
All AI-generated code were read-over and improved by hand for correctness.

Uses of AI include
    - Code for argument parsing and file path handling
    - Implementation of proper regex for pattern matching for all cleaning steps
    - Implementation of advanced cleaning logic for file link and table wrapper,
      including drop_file_links and drop_tables
    - Scan dev.txt files and suggest new cleaning steps on top of ones I had
    - Code proofreading and improvement
"""


import os
import re
import html
import argparse
from pathlib import Path


def drop_file_links(text):
    """Removes image links of the form [[File:...]]"""

    # compilation of file namespaces appeared in the dumps
    FILE_TARGET = re.compile(
        r"(?i)^(?::|[a-z]{1,20}:)*(?:datei|bild|file|image|tập tin|tập_tin|hình|"
        r"tiedosto|kuva|fájl|kép|файл|изображение|ملف|صورة|文件|图像|圖像|檔案)\s*:"
    )

    pieces = []
    index = 0
    length = len(text)
    while index < length:
        start = text.find("[[", index)
        if start < 0:
            pieces.append(text[index:])
            break
        cursor = start + 2
        depth = 0
        target_end = None
        while cursor < length:
            if text.startswith("[[", cursor):
                depth += 1
                cursor += 2
            elif text.startswith("]]", cursor):
                if depth == 0:
                    target_end = cursor
                    break
                depth -= 1
                cursor += 2
            elif text[cursor] == "|" and depth == 0:
                target_end = cursor
                break
            else:
                cursor += 1
        target = text[start + 2:target_end].strip() if target_end is not None else ""
        if target_end is None or not FILE_TARGET.match(target):
            pieces.append(text[index:start + 2])
            index = start + 2
            continue
        depth = 1
        cursor = start + 2
        while cursor < length and depth:
            if text.startswith("[[", cursor):
                depth += 1
                cursor += 2
            elif text.startswith("]]", cursor):
                depth -= 1
                cursor += 2
            else:
                cursor += 1
        if depth:
            pieces.append(text[index:])
            break
        pieces.append(text[index:start])
        index = cursor
    return "".join(pieces)


def _table_opens_at(text, pos):
    """Returns whether {| starts a line, aside from spaces or one colon."""
    cursor = pos
    while cursor > 0 and text[cursor - 1] in " \t":
        cursor -= 1
    if cursor > 0 and text[cursor - 1] == ":":
        cursor -= 1
        while cursor > 0 and text[cursor - 1] in " \t":
            cursor -= 1
    return cursor == 0 or text[cursor - 1] == "\n"


def _find_table_open(text, index):
    """Finds the first occurrence of {| at or after index."""
    while True:
        start = text.find("{|", index)
        if start < 0:
            return -1
        if _table_opens_at(text, start):
            return start
        index = start + 2


def drop_tables(text):
    """Removes wiki tables from {| to matching |}."""
    pieces = []
    index = 0
    length = len(text)
    while index < length:
        start = _find_table_open(text, index)
        if start < 0:
            pieces.append(text[index:])
            break
        depth = 1
        cursor = start + 2
        while cursor < length and depth:
            if text.startswith("{|", cursor) and _table_opens_at(text, cursor):
                depth += 1
                cursor += 2
            elif text.startswith("|}", cursor):
                depth -= 1
                cursor += 2
            else:
                cursor += 1
        if depth:
            pieces.append(text[index:])
            break
        if cursor < length and text[cursor] == "\n":
            cursor += 1
        pieces.append(text[index:start])
        index = cursor
    return "".join(pieces)


def resolve_chinese_variants(text):
    """Replaces -{...}- with zh-hans, otherwise zh-cn, otherwise zh-sg."""

    def pick(match):
        """Choose one written form from a variant switch."""
        choices = {}
        plain = []
        for part in match.group(1).split(";"):
            part = part.strip()
            if not part:
                continue
            if "|" in part:
                flag, rest = part.split("|", 1)
                if not flag.strip().lower().startswith("zh"):
                    part = rest.strip()
            if ":" in part:
                code, value = part.split(":", 1)
                code = code.strip().lower().replace("_", "-")
                if code.startswith("zh"):
                    choices[code] = value.strip()
                    continue
            if part:
                plain.append(part)
        # zh-hans, then zh-cn, then zh-sg
        VARIANT_PRIORITY = ("zh-hans", "zh-cn", "zh-sg")
        for code in VARIANT_PRIORITY:
            if choices.get(code):
                return choices[code]
        if plain:
            return plain[0]
        if choices:
            return next(iter(choices.values()))
        return ""

    previous = None
    while previous != text:
        previous = text
        # -{...}- on one line, not -{{template}}
        text = re.sub(r"-{(?!\{)(.*?)}-", pick, text)
    return text


def clean_text(text):
    # remove wiki tables
    # {|...|}
    text = drop_tables(text)

    # decode HTML entities
    # &nbsp; &#x200B; &shy;
    text = html.unescape(text)
    text = text.replace('\xa0', ' ').replace('\u200b', '').replace('\u00ad', '')

    # remove HTML comments
    # <!--...-->
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # remove reference tags
    # <ref .../> <ref>...</ref>
    text = re.sub(r'<ref\b[^>]*/>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<ref\b[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # remove templates
    # {{...}}
    for _ in range(3):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text, flags=re.DOTALL)

    # resolve Chinese variant switches
    # -{...}-
    text = resolve_chinese_variants(text)

    # remove category and file links
    # [[Category:...]] [[File:...]]
    CATEGORY_LINK = re.compile(
        r"\[\[[ \t]*(?::|[A-Za-z]{1,20}:)*"
        r"(?:[Cc]ategory|[Kk]ategorie|[Kk]ategória|[Ll]uokka|Категория|категория|"
        r"تصنيف|[Tt]hể loại|分类|分類|[Cc]at)\s*:[^\[\]\n]*\]\]"
    )
    text = CATEGORY_LINK.sub("", text)
    text = drop_file_links(text)

    # unwrap wiki links
    # [[target|shown]] [[target]]
    text = re.sub(r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # unwrap external links
    # [https://... label] [https://...]
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', ' ', text)

    # remove bold and italic markers
    # ''' ''
    text = text.replace("'''", '').replace("''", '')

    # unwrap headings
    # ==...== (up to 6 equal signs)
    text = re.sub(r'^\s*={2,6}\s*(.*?)\s*={2,6}\s*$', r'\1', text, flags=re.MULTILINE)

    # remove list markers and horizontal rules
    # * # ; : ----
    text = re.sub(r'^\s*[*#;:]+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*-{4,}\s*$', '', text, flags=re.MULTILINE)

    # replace line-break tags and strip leftover tags
    # <br>, <br />, <sup>...</sup>, etc.
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?[a-zA-Z][^>\n]*>', '', text)

    # collapse consecutive empty lines 
    # remove trailing whitespace and collapse consecutive whitespace
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip('\n') + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # data/raw/fi/dev.txt -> artifacts/part1/cleaned/fi/dev.txt
    for root, _, files in os.walk(input_path):
        for filename in files:
            input_file = Path(root) / filename

            relative_path = input_file.relative_to(input_path)
            output_file = output_path / relative_path

            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()

            cleaned_text = clean_text(text)

            with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
                f.write(cleaned_text)
            
            print(f'Cleaned {input_file} -> {output_file}')

if __name__ == '__main__':
    main()