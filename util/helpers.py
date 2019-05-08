#!/usr/bin/env python
import codecs
import re
import json
import os


def count_string(string, text):
    pattern = f'\b{string}\b'
    matches = re.findall(string, text)
    return len(matches)


def repr_json(file_name):
    with open(file_name, 'r') as f:
        _ = json.loads(f.read())
    return json.dumps(_, indent=1)


def write_ndjson(filename, data_iterable, append=False, **kwargs):
    """
    Generator that writes newline-delimited json to a file and returns items
    from an iterable.

    - code from searchtweets.utils
    """
    write_mode = "ab" if append else "wb"
    with codecs.open(filename, write_mode, "utf-8") as outfile:
        for item in data_iterable:
            outfile.write(json.dumps(item) + "\n")
            yield item


def weekly_summary():
    ROOT_PATH = os.path.realpath('..')
    files = os.listdir(os.path.join(ROOT_PATH, 'output'))
    for file in files:
        pos = 0
        neg = 0
        neu = 0
        with open(os.path.join(ROOT_PATH,'output',file), 'r') as f:
            sentiment_summary = json.loads(f.readlines()[0])
        for k,v in sentiment_summary.items():
            if 'pos' in v.keys():
                pos += v.get('pos')
            if 'neg' in v.keys():
                neg += v.get('neg')
            if 'neu' in v.keys():
                neu += v.get('neu')
        fmt = '{}\n' \
              'Positive: {}\n' \
              'Negative: {}\n' \
              'Neutral: {}\n'
        print(fmt.format(file, pos, neg, neu))


def main():
    weekly_summary()


if __name__ == "__main__":
    main()