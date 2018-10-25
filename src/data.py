#!/usr/bin/env python3

from collections import Counter
from trial import config as C
from util import PointedIndex
from util_io import *
from util_io import path as P
from util_np import np, encode

#########################
# build vocab and index #
#########################

# words = Counter()
# for _, (src, tgt) in zip(range(2**24), sieve(load_tab(pform(P.wmt, "corpus")), C.cap_src, C.cap_tgt)):
#     words.update(src.split())
#     words.update(tgt.split())

# save_txt(pform(P.data, P.voc), ("{}\t{}".format(w, n) for w, n in vocab.most_common()))

# words = Counter()
# for w, n in map(str.split, load_txt(pform(P.data, P.voc))):
#     words[w] = int(n)

# chars = Counter()
# for w, n in words.items():
#     chars.update({c: n for c in w})

# index = "".join(vocab(chars, specials= "\xa0\n "))
# save_pkl(pform(P.data, P.idx), index)

###########################
# prepare validation data #
###########################

src_tgt = list(sieve(
    ("{}\t{}\t".format(src, tgt) for src, tgt in zip(
        load_txt(pform(P.wmt, "dev/newstest2017.tc.de"))
        , load_txt(pform(P.wmt, "dev/newstest2017.tc.en"))))
    , C.cap_src
    , C.cap_tgt))

np.random.seed(C.seed)
np.random.shuffle(src_tgt)

src, tgt = zip(*src_tgt[:C.batch_valid * 5])
idx = PointedIndex(load_pkl(pform(P.data, P.idx)))
np.save(pform(P.data, P.src), encode(idx, src, dtype= np.uint8))
np.save(pform(P.data, P.tgt), encode(idx, tgt, dtype= np.uint8, length= C.cap_tgt))
