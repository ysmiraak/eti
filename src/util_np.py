from itertools import chain
from util import partial
import numpy as np


def vpack(arrays, fill, dtype= None):
    """like `np.vstack` but for `arrays` of different lengths in the first
    axis.  shorter ones will be padded with `fill` at the end.

    """
    if not hasattr(arrays, '__len__'): arrays = list(arrays)
    arr = arrays[0]
    if dtype is None: dtype = arr.dtype
    a = np.full((len(arrays), max(map(len, arrays))) + arr.shape[1:], fill, dtype)
    for row, arr in zip(a, arrays):
        row[:len(arr)] = arr
    return a


def partition(n, m, discard= True):
    """yields pairs of indices which partitions `n` nats by `m`.  if not
    `discard`, also yields the final incomplete partition.

    """
    steps = range(0, 1 + n, m)
    yield from zip(steps, steps[1:])
    if n % m and not discard:
        yield n - (n % m), n


def sample(n, m, seed= 0):
    """yields `m` samples from `n` nats."""
    assert 0 < m <= n
    data = np.arange(n)
    while True:
        np.random.seed(seed)
        np.random.shuffle(data)
        yield from (data[i:j] for i, j in partition(n, m))


def encode(index, sent, padr= (), padl= (), dtype= np.uint8):
    """-> array dtype

    encodes chained `padl, sent, padr : seq str` according to
    `index : PointedIndex`.

    """
    return np.fromiter(map(index, chain(padl, sent, padr)), dtype)


def decode(index, idxs, end= "\n", sep= ""):
    """-> list str

    decodes `idxs : array int` according to `index : PointedIndex`.

    stops at `end` and joins the results with `sep`.

    """
    return sep.join([index[i] for i in idxs[:np.argmax(idxs == index(end))]])
