from collections import Counter
from itertools import chain


def load(filename):
    """yields lines."""
    with open(filename) as file:
        for line in file:
            yield line.strip()


def save(filename, lines):
    """writes lines."""
    with open(filename, 'w') as file:
        for line in lines:
            print(line, file= file)


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


def encode(index, sent, pad_end= (), pad_start= ()):
    """-> list int

    encodes chained `pad_start, sent, pad_end : seq str` according to
    `index : PointedIndex`.

    """
    return list(map(index, chain(pad_start, sent, pad_end)))


def decode(index, idxs, end= "\n", sep= ""):
    """-> str

    decodes `idxs : seq int` according to `index : PointedIndex`.

    stops at `end` and joins the results with `sep`.

    """
    end = index(end)
    tgt = []
    for idx in idxs:
        if idx == end: break
        tgt.append(index[idx])
    return sep.join(tgt)
