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


def decode(index, idxs, sep= "", end= "\n"):
    """-> list str

    decodes `idxs : array int` according to `index : PointedIndex`.
    stops at `end` and joins the results with `sep`.  if `idxs` has a
    higher rank, generates the results instead.

    """
    if 1 < idxs.ndim: return (decode(index, xs, end, sep) for xs in idxs)
    return sep.join([index[i] for i in idxs[:np.argmax(idxs == index(end))]])


def encode(index, sent, length= None, dtype= np.uint8, pad= "\n"):
    """-> array dtype

    encodes `sent : seq seq str | seq str` according to `index :
    PointedIndex`.  always returns a rank 2 array.  the second axis
    will be padded to `length` or the maximum length.

    """
    if not hasattr(sent, '__len__'): sent = list(sent)
    if isinstance(sent[0], str): sent = sent,
    sent = [np.fromiter(map(index, s), dtype) for s in sent]
    if length is None: length = max(map(len, sent))
    return vpack(sent, (len(sent), length), index(pad), dtype)
