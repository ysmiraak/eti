from itertools import islice, takewhile
from util import identity
import numpy as np


def vpack(arrays, shape, fill, dtype= None):
    """like `np.vstack` but for `arrays` of different lengths in the first
    axis.  shorter ones will be padded with `fill` at the end.

    """
    array = np.full(shape, fill, dtype)
    for row, arr in zip(array, arrays):
        row[:len(arr)] = arr
    return array


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


def batch(stream, len_bat, shuffle= 2**14, seed= 0, discard= False):
    """yields batches from `stream`."""
    assert not shuffle % len_bat
    while True:
        buf = list(islice(stream, shuffle))
        if not buf: break
        np.random.seed(seed)
        np.random.shuffle(buf)
        yield from (buf[i:j] for i, j in partition(shuffle, len_bat, discard= discard))


def decode(index, array, sep= "", end= "\n"):
    """-> list str

    decodes `array : array int` according to `index : PointedIndex`.
    stops at `end` and joins the results with `sep`.  if `array` has a
    higher rank, generates the results instead.

    """
    if 1 < array.ndim: return (decode(index, arr, sep, end) for arr in array)
    return sep.join([index[i] for i in array[:sum(takewhile(identity, array != index(end)))]])


def encode(index, sents, length= None, dtype= np.int, pad= "\n"):
    """-> array dtype

    encodes `sents : seq seq str` according to `index : PointedIndex`.
    returns a rank 2 array whose second axis is be padded to `length`
    or the maximum length.

    """
    sents = [np.fromiter(map(index, sent), dtype) for sent in sents]
    if length is None: length = max(map(len, sents))
    return vpack(sents, (len(sents), length), index(pad), dtype)
