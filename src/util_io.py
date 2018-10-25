from collections import Counter
from itertools import chain
from itertools import islice
import pickle
import random
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


def sieve(src_tgt, cap_src, cap_tgt):
    for src, tgt in src_tgt:
        src = clean(src)
        tgt = clean(tgt)
        if not src or not tgt \
           or cap_src < len(src) \
           or cap_tgt < len(tgt):
            continue
        yield src, tgt


def batch(src_tgt, len_bat, shuffle= 2**14, seed= 0):
    assert not shuffle % len_bat
    while True:
        buf = list(islice(src_tgt, shuffle))
        if not buf: break
        random.seed(seed)
        random.shuffle(buf)
        yield from (zip(*buf[i:j]) for i, j in partition(shuffle, len_bat))


def load_wmt(filename):
    with open(filename) as lines:
        yield from (line.split("\t")[0:2] for line in lines)


def load_txt(filename):
    """yields lines from text file."""
    with open(filename) as file:
        yield from (line.strip() for line in file)


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
