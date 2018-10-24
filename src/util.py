from collections.abc import Mapping
from functools import partial


def identity(x):
    """a -> a"""
    return x


def comp(g, f, *fs):
    """(b -> c) -> (a -> b) -> (a -> c)"""
    if fs: f = comp(f, *fs)
    return lambda x: g(f(x))


def diter(xs, depth= 2):
    """like `iter` but yields items at `depth`."""
    if depth:
        for x in xs:
            yield from diter(x, depth= depth - 1)
    else:
        yield from xs


class Record(Mapping):
    """a `dict`-like type whose instances are partial finite mappings from
    attribute keys to arbitrary values.

    like a dict, a record is transparent about its content.

    unlike a dict, a record can access its content as object
    attributes without the hassle of string quoting.

    """

    def __init__(self, *records, **entries):
        for rec in records:
            for key, val in rec.items():
                setattr(self, key, val)
        for key, val in entries.items():
            setattr(self, key, val)

    def __repr__(self):
        return repr(self.__dict__)

    def __bool__(self):
        return bool(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)


def select(record, *keys):
    """returns a Record with only entries under `keys` in `record`."""
    return Record({k: record[k] for k in keys})


class PointedIndex:
    """takes a vector of unique elements `vec` and a base index `nil`
    within its range, returns a pointed index `idx`, such that:

    `idx[i]` returns the element at index `i`;

    `idx(x)` returns the index of element `x`;

    bijective for all indices and elements within `vec`;

    otherwise `idx(x) => nil` and `idx[i] => vec[nil]`.

    """

    def __init__(self, vec, nil= 0):
        self._nil = nil
        self._i2x = vec
        self._x2i = {x: i for i, x in enumerate(vec)}

    def __repr__(self):
        return "{}(vec= {}, nil= {})".format(
            type(self).__name__
            , repr(self._i2x)
            , repr(self._nil))

    def __getitem__(self, i):
        try:
            return self._i2x[i]
        except IndexError:
            return self._i2x[self._nil]

    def __call__(self, x):
        try:
            return self._x2i[x]
        except KeyError:
            return self._nil

    def __len__(self):
        return len(self._i2x)

    @property
    def vec(self):
        return self._i2x

    @property
    def nil(self):
        return self._nil
