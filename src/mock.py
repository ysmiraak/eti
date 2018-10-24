#!/usr/bin/env python3

split = 0.01

from trial import pform, path as P, config as C

# load data
from util_io import load
src = list(load('../data/europarl-v7.de-en.de'))
tgt = list(load('../data/europarl-v7.de-en.en'))

# select data
src_tgt = []
for s, t in zip(src, tgt):
    # ignores empty sentences
    if not s or not t: continue
    # ignores long source sentences
    if C.cap_src < len(s): continue
    # ignores long target sentences
    # a target sentence will have to be padded once at the end
    if C.cap_tgt < len(t) + 1: continue
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
save(pform(P.data, 'valid_src'), valid_src)
save(pform(P.data, 'valid_tgt'), valid_tgt)
save(pform(P.data, 'train_src'), train_src)
save(pform(P.data, 'train_tgt'), train_tgt)
