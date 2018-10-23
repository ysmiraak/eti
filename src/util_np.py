import numpy as np


def vpack(arrays, fill, offset= 0, extra= 0):
    """like `np.vstack` but for `arrays` of different lengths in the first
    axis.  shorter ones will be padded with `fill` at the end.
    additionally `offset` and `extra` number of columns will be padded
    at the beginning and the end.

    """
    if not hasattr(arrays, '__len__'): arrays = list(arrays)
    arr = arrays[0]
    a = np.full((len(arrays), offset + max(map(len, arrays)) + extra) + arr.shape[1:], fill, arr.dtype)
    for row, arr in zip(a, arrays):
        row[offset:offset+len(arr)] = arr
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
