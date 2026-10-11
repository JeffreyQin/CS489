import os
import re
import html
import argparse
from pathlib import Path


# The part before the colon in an image link. Each Wikipedia uses its own
# name for the file namespace; these are the ones that appear in the dumps.
FILE_NAMESPACE = (
    "datei|bild|file|image|tập tin|tập_tin|hình|tiedosto|kuva|"
    "fájl|kép|файл|изображение|ملف|صورة|文件|图像|圖像|檔案"
)
FILE_TARGET = re.compile(
    rf"(?i)^(?::|[a-z]{{1,20}}:)*(?:{FILE_NAMESPACE})\s*:"
)
# A category link is one closed [[Category:...]] with no nested link and no
# line break. Balancing brackets across a missing ]] deletes the rest of the file.
CATEGORY_LINK = re.compile(
    r"\[\[[ \t]*(?::|[A-Za-z]{1,20}:)*"
    r"(?:[Cc]ategory|[Kk]ategorie|[Kk]ategória|[Ll]uokka|Категория|категория|"
    r"تصنيف|[Tt]hể loại|分类|分類|[Cc]at)\s*:[^\[\]\n]*\]\]"
)


def drop_file_links(text):
    """Delete [[File:...]] links, including a caption and any link nested in it.

    An image is not an article link. Keeping the text after the pipe leaves
    the caption parameters (mini|, liên_kết=, 250px|) in the prose. The link
    is removed only when the target starts with a file namespace, so an
    ordinary link such as [[Kuopio]] is unchanged.
    """
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
    """A table opener is {| at the start of a line, aside from spaces or one colon.

    {| {| on one line is a single opener. {| inside a sentence, as in `{|}~`,
    is not a table.
    """
    cursor = pos
    while cursor > 0 and text[cursor - 1] in " \t":
        cursor -= 1
    if cursor > 0 and text[cursor - 1] == ":":
        cursor -= 1
        while cursor > 0 and text[cursor - 1] in " \t":
            cursor -= 1
    return cursor == 0 or text[cursor - 1] == "\n"


def _find_table_open(text, index):
    while True:
        start = text.find("{|", index)
        if start < 0:
            return -1
        if _table_opens_at(text, start):
            return start
        index = start + 2


def drop_tables(text):
    """Delete a wiki table from {| through the matching |}, including nested tables.

    The rows are markup. Leaving them in place fills the text with lines such as
    |- and ! Abbildung, and pushes the following section down. Only a {| that
    starts a line opens a table, so one extra {| on the same line does not make
    the match run into the next article.
    """
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


# Inside -{ }-, keep the first of these labels that is present.
VARIANT_PRIORITY = ("zh-hans", "zh-cn", "zh-sg")


def resolve_chinese_variants(text):
    """Replace a -{ }- variant switch with one written form.

    zh-hans wins, then zh-cn, then zh-sg. A switch with none of those labels
    keeps the unmarked text, so -{于}- stays 于.
    """

    def pick(match):
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
        # -{{template}} starts with the same two characters. The switch is -{ not
        # followed by another brace, and it stays on one line.
        text = re.sub(r"-{(?!\{)(.*?)}-", pick, text)
    return text


def clean_text(text):
    # A table has to go before headings are unwrapped. Deleting it also
    # removes the line break it occupied, so the next heading moves up.
    text = drop_tables(text)

    # convert &nbsp; -> space, &#x200B; -> nothing
    text = html.unescape(text)
    text = text.replace('\xa0', ' ').replace('\u200b', '').replace('\u00ad', '')

    # remove comments
    # <!-- comment --> => ''
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # <ref name="x" /> must go first. If it is left in place, the next
    # pattern treats the tag as the start of a citation and deletes
    # everything up to the following </ref>.
    text = re.sub(r'<ref\b[^>]*/>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<ref\b[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # remove templates
    # {{template}} => ''
    for _ in range(3):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text, flags=re.DOTALL)

    # -{zh-hant:馬拉拉·優素福扎伊;zh-hans:马拉拉·优素福扎伊;}- becomes
    # 马拉拉·优素福扎伊. Done before link unwrapping so a pipe inside the
    # switch is not read as a link pipe.
    text = resolve_chinese_variants(text)

    # An image link is not prose. [[Datei:x.jpg|mini|caption]] is deleted
    # whole, before the ordinary link rule can keep the caption.
    text = CATEGORY_LINK.sub("", text)
    text = drop_file_links(text)

    # remove wiki link brackets
    # [[Finland|Finnish]] => Finnish, [[Finland]] => Finland
    text = re.sub(r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # remove wiki link brackets
    # [https://example.com Example] => Example
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', ' ', text)

    # remove bold and italic text
    # '''word''' => word
    text = text.replace("'''", '').replace("''", '')

    # section headers
    # == Events == -> Events
    text = re.sub(r'^\s*={2,6}\s*(.*?)\s*={2,6}\s*$', r'\1', text, flags=re.MULTILINE)

    # remove list items and horizontal rules
    # * item => item, ---- => ''
    text = re.sub(r'^\s*[*#;:]+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*-{4,}\s*$', '', text, flags=re.MULTILINE)

    # convert <br> to newline, then remove leftover tags
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?[a-zA-Z][^>\n]*>', '', text)

    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # A removed comment or template leaves the spaces that stood on either
    # side of it. "–  cselló" and "Alphabet.  Als" are that case.
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

    # recursively clean every input file
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