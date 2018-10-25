from collections import Counter
from os.path import expanduser, join
from util import Record
import pickle
import re


path = Record(
    log = expanduser("~/cache/tensorboard-logdir/eti")
    , wmt = expanduser("~/data/wmt/de-en")
    , pred = "../trial/pred"
    , ckpt = "../trial/ckpt"
    , data = "../trial/data"
    , voc = "voc.tsv"
    , idx = "idx.pkl"
    , src = "src.npy"
    , tgt = "tgt.npy"
)


def pform(path, *names, sep= ''):
    """format a path as `path` followed by `names` joined with `sep`."""
    return join(path, sep.join(map(str, names)))


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


def sieve(lines, cap_src, cap_tgt):
    for line in lines:
        src, tgt, _ = clean(line).split("\t")
        src = " ".join(src.split())
        tgt = " ".join(tgt.split())
        if not src or not tgt \
           or cap_src < len(src) \
           or cap_tgt < len(tgt):
            continue
        yield src, tgt


def load_txt(filename):
    """yields lines from text file."""
    with open(filename) as file:
        yield from (line for line in file)


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


def vocab(xs, specials= "\xa0\n ", min_freq= 2, top= 256):
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
