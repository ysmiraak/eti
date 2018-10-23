#!/usr/bin/env python3

from collections import Counter
from util import diter
from util_io import load
import matplotlib.pyplot as plt
import numpy as np

path = "../data/europarl-v7.de-en.de"
corp = list(load(path))

print(len(corp), "sentences")

lengths = np.fromiter(map(len, corp), np.float32)

# quantiles by lengths
for i in range(11):
    t = 50 * i
    print("len <= {:03d}:".format(t), np.sum(lengths <= t) / len(lengths))

alphabet = Counter(diter(corp))

print(len(alphabet), "characters")
