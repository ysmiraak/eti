from collections import Counter
from util import diter
from util_io import load_txt
import matplotlib.pyplot as plt
import numpy as np

path = "../data/europarl-v7.de-en.de"
corp = list(load_txt(path))

print(len(corp), "sentences")

lengths = np.fromiter(map(len, corp), np.float32)

# quantiles by lengths
for i in range(11):
    t = 50 * i
    print("len <= {:03d}:".format(t), np.sum(lengths <= t) / len(lengths))

alphabet = Counter(diter(corp))

print(len(alphabet), "characters")

#####################
# big training data #
#####################

from collections import Counter
from trial import config as C
from util_io import sieve
import numpy as np

vocab, count = Counter(), 0
for s, t in clean_load("/data/wmt/de-en/corpus", C.cap_src, C.cap_tgt):
    vocab.update(s.split())
    vocab.update(t.split())
    count += 1
    if count >= 2**24:
        break

with open("../trial/vocab", 'w') as lines:
    for w, n in vocab.most_common():
        print(w, n, sep= "\t", file= f)

vocab = Counter()
with open("../trial/vocab") as lines:
    for line in lines:
        w, n = line.split()
        vocab[w] = int(n)

from util_io import vocab as chars, save_pkl, load_pkl

alphabet = Counter()
for w, n in vocab.items():
    alphabet.update({c: n for c in w})

index = "".join(chars(alphabet, specials= "\xa0\n "))
save_pkl("../trial/data/index", index)
index = load_pkl("../trial/data/index")

###################
# validation data #
###################

from util_io import load_pkl
from util import PointedIndex
index = PointedIndex(load_pkl("../trial/data/index"))

with open("/data/wmt/de-en/dev/newstest2017.tc.de") as src, \
     open("/data/wmt/de-en/dev/newstest2017.tc.en") as tgt:
    src_tgt = list(sieve(zip(src, tgt), C.cap_src, C.cap_tgt))

import random
random.seed(C.seed)
random.shuffle(src_tgt)

src, tgt = zip(*src_tgt)

from util import partial
from util_np import encode

np.save("../trial/data/source.npy", encode(index, src, dtype= np.uint8))
np.save("../trial/data/target.npy", encode(index, tgt, C.cap_tgt, dtype= np.uint8))
