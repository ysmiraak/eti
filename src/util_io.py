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
