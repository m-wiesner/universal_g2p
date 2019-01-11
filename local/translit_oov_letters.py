from __future__ import print_function
import sys
import os
import unicodedata
import codecs
sys.stdout = codecs.getwriter('utf8')(sys.stdout)


def main():
    letters = set()
    with codecs.open(sys.argv[1], "r", encoding="utf-8") as f:
        for l in f:
            for c in l.strip().split('\t', 1)[0]:
                letters.add(c)
    
    with codecs.open(sys.argv[2], "r", encoding="utf-8") as f:
        for l in f:
            try:
                word, pron = l.strip().split('\t', 1)
                word_ = replace_oov_letters(word, letters)
                print(u"{}\t{}".format(word_, pron))
            except ValueError:
                word = l.strip()
                word_ = replace_oov_letters(word, letters)
                print(u"{}".format(word_))


def replace_oov_letters(input_str, letters):
    output_str = u""
    for c in input_str:
        if c not in letters:
            normalized = unicodedata.normalize("NFKD", c)
            output_str += u"".join([c for c in normalized if not unicodedata.combining(c)])
        else:
            output_str += c
    return output_str


if __name__ == "__main__":
    main()
