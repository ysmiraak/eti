#!/usr/bin/env python3

from collections import Counter
from itertools import chain
from trial import config as C
from util import diter, PointedIndex
from util_io import path as P, pform, sieve, load_txt, save_txt, save_pkl, vocab
from util_np import np, encode

##################
# load and split #
##################

src_tgt = list(sieve(
    zip(load_txt(pform(P.raw, "europarl-v7.de-en.de"))
        , load_txt(pform(P.raw, "europarl-v7.de-en.en")))
    , C.cap))
np.random.seed(C.seed)
np.random.shuffle(src_tgt)
src_valid, tgt_valid = zip(*src_tgt[:C.batch_valid * 5])
src_train, tgt_train = zip(*src_tgt[C.batch_valid * 5:])
del src_tgt

###############
# build index #
###############

chars = Counter(chain(diter(src_train, 2), diter(tgt_train, 2)))
index = "".join(vocab(chars, specials= "\xa0\n "))
save_pkl(pform(P.data, P.idx), index)
save_txt(pform(P.data, P.src), src_valid)
save_txt(pform(P.data, P.tgt), tgt_valid)

#############
# save data #
#############

idx = PointedIndex(index)
np.save(pform(P.data, "valid_src.npy"), encode(idx, src_valid, C.cap, np.uint8))
np.save(pform(P.data, "valid_tgt.npy"), encode(idx, tgt_valid, C.cap, np.uint8))
np.save(pform(P.data, "train_src.npy"), encode(idx, src_train, C.cap, np.uint8))
np.save(pform(P.data, "train_tgt.npy"), encode(idx, tgt_train, C.cap, np.uint8))
