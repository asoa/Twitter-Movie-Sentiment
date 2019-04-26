#!/usr/bin/env python
import codecs
import re
import json


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
