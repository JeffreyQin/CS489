import os
import argparse
from pathlib import Path


def count_bytes(file_path):
    with open(file_path, 'rb') as f:
        return len(f.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-input', type=str, required=True)
    parser.add_argument('--clean-input', type=str, required=True)
    args = parser.parse_args()

    raw_path = Path(args.raw_input)
    clean_path = Path(args.clean_input)

    print(
        f'{"Language":<12} '
        f'{"Raw Train":>12} '
        f'{"Clean Train":>12} '
        f'{"Raw Dev":>12} '
        f'{"Clean Dev":>12}'
    )

    print('-' * 64)

    for lang in sorted(os.listdir(raw_path)):
        raw_lang_path = raw_path / lang
        clean_lang_path = clean_path / lang

        if not raw_lang_path.is_dir():
            continue

        raw_train = raw_lang_path / 'train.txt'
        raw_dev = raw_lang_path / 'dev.txt'

        clean_train = clean_lang_path / 'train.txt'
        clean_dev = clean_lang_path / 'dev.txt'

        raw_train_bytes = count_bytes(raw_train)
        clean_train_bytes = count_bytes(clean_train)

        raw_dev_bytes = count_bytes(raw_dev)
        clean_dev_bytes = count_bytes(clean_dev)

        print(
            f'{lang:<12} '
            f'{raw_train_bytes:>12,} '
            f'{clean_train_bytes:>12,} '
            f'{raw_dev_bytes:>12,} '
            f'{clean_dev_bytes:>12,}'
        )


if __name__ == '__main__':
    main()