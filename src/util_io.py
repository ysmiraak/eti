from collections import Counter
from itertools import chain
import pickle
import re


def clean(s, p= re.compile(r"&#91;|&#93;|&amp;|&apos;|&gt;|&lt;|&quot;|@-@|� s")
          , tr= lambda m, t= {
              "� s": "'s"
              , "&#91;": "["
              , "&#93;": "]"
              , "&amp;": "&"
              , "&apos;": "'"
              , "&gt;": ">"
              , "&lt;": "<"
              , "&quot;": "\""
              , "@-@": "-"}: t[m.group()]):
    return p.sub(tr, s)


def clean_load(filename, cap_src, cap_tgt):
    with open(filename) as lines:
        for line in lines:
            s, t, _ = clean(line).split("\t")
            if not s or not t \
               or cap_src < len(s) \
               or cap_tgt < len(t):
                continue
            yield s, t


def load_txt(filename):
    """yields lines from text file."""
    with open(filename) as file:
        for line in file:
            yield line.strip()


def save_txt(filename, lines):
    """writes lines to text file."""
    with open(filename, 'w') as file:
        for line in lines:
            print(line, file= file)


def load_pkl(filename):
    """loads pickle file."""
    with open(filename, 'rb') as dump:
        return pickle.load(dump)


def save_pkl(filename, obj):
    "saves to pickle file."
    with open(filename, 'wb') as dump:
        pickle.dump(obj, dump)


def vocab(xs, specials= "\xa0\n", min_freq= 2, top= 256):
    """-> (list a) where

    xs       : seq a
    specials : seq a
    min_freq : nat
    top      : nat | None

    returns the `top` most frequent items with `min_freq` in `xs`, and
    ensures that the `specials` are included with the highest ranks.

    """
    freq = Counter(xs)
    for x in specials: del freq[x]
    freq = [(-n, x) for x, n in freq.items() if min_freq <= n]
    freq.sort()
    vocab = list(specials)
    vocab.extend((x for _, x in freq))
    return vocab[:top] if top else vocab
