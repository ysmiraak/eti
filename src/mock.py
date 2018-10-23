#!/usr/bin/env python3

len_cap = 2**8
path  = "../data"
path2 = "../trial/data"
split = 0.01

# load data
from os.path import join
from util_io import load
src = list(load(join(path, "europarl-v7.de-en.de")))
tgt = list(load(join(path, "europarl-v7.de-en.en")))

# select data
src_tgt = []
for s, t in zip(src, tgt):
    # ignores empty sentences
    if not s or not t: continue
    # ignores long source sentences
    if len_cap < len(s): continue
    # ignores long target sentences
    # a target sentence will have to be padded once at the end
    if len_cap < len(t) + 1: continue
    src_tgt.append((s, t))

# train valid split
import random
random.seed(0)
random.shuffle(src_tgt)
i = int(len(src_tgt) * split)
valid_src, valid_tgt = zip(*src_tgt[:i])
train_src, train_tgt = zip(*src_tgt[i:])

# save data
from util_io import save
save(join(path2, "valid_src"), valid_src)
save(join(path2, "valid_tgt"), valid_tgt)
save(join(path2, "train_src"), train_src)
save(join(path2, "train_tgt"), train_tgt)
