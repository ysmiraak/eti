#!/usr/bin/env python3

from trial import config as C
from util_io import path as P, pform, load_txt, save_txt
from util_np import np, vpack
from util_sp import spm

path_src = pform(P.raw, "europarl-v7.de-en.de")
path_tgt = pform(P.raw, "europarl-v7.de-en.en")

###############
# build vocab #
###############

vocab_src = spm(pform(P.data, "vocab_src"), path_src, C.dim_src, C.bos, C.eos, C.unk)
vocab_tgt = spm(pform(P.data, "vocab_tgt"), path_tgt, C.dim_tgt, C.bos, C.eos, C.unk)

#############
# load data #
#############

src_tgt = list(zip(load_txt(path_src), load_txt(path_tgt)))
np.random.seed(C.seed)
np.random.shuffle(src_tgt)

####################
# filter and split #
####################

train_src = []
train_tgt = []
valid_src = []
valid_tgt = []
valid_raw = []
for src, tgt in src_tgt:
    s = vocab_src.encode_as_ids(src)
    t = vocab_tgt.encode_as_ids(tgt)
    if 0 < len(s) <= C.cap and 0 < len(t) <= C.cap:
        if len(valid_raw) < C.total_valid:
            valid_src.append(s)
            valid_tgt.append(t)
            valid_raw.append(tgt)
        else:
            train_src.append(src)
            train_tgt.append(tgt)

#############
# save data #
#############

save_txt(pform(P.data, "train_src.txt"), train_src)
save_txt(pform(P.data, "train_tgt.txt"), train_tgt)
save_txt(pform(P.data, "valid_tgt.txt"), valid_raw)
np.save(pform(P.data, "valid_tgt.npy"), vpack(valid_tgt, (C.total_valid, C.cap), C.eos, np.uint16))
np.save(pform(P.data, "valid_src.npy"), vpack(valid_src, (C.total_valid, C.cap), C.eos, np.uint16))
