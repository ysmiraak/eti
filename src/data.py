#!/usr/bin/env python3

from trial import pform, path as P, config as C
from util import partial, diter, PointedIndex
from util_io import load, vocab
from util_np import np, vpack, encode

# load training data
src = list(load(pform(P.data, P.train_src)))
tgt = list(load(pform(P.data, P.train_tgt)))

# build indices
idx_src = PointedIndex("".join(vocab(diter(src), top= 256)))
idx_tgt = PointedIndex("".join(vocab(diter(tgt), top= 256)))
enc_src = partial(encode, idx_src)
enc_tgt = partial(encode, idx_tgt, padr= "\n")

# assume 1 is the index for the end symbol, and top= 256
assert 1 == idx_src("\n") == idx_tgt("\n")
pack = partial(vpack, fill= 1)

# prepare and save training and validation data
np.save(pform(P.data, P.index_src), idx_src.vec)
np.save(pform(P.data, P.index_tgt), idx_tgt.vec)
np.save(pform(P.data, P.train_src), pack(map(enc_src, src)))
np.save(pform(P.data, P.train_tgt), pack(map(enc_tgt, tgt)))
np.save(pform(P.data, P.valid_src), pack(map(enc_src, load(pform(P.data, P.valid_src)))))
np.save(pform(P.data, P.valid_tgt), pack(map(enc_tgt, load(pform(P.data, P.valid_tgt)))))
