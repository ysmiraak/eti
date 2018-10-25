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




from collections import Counter
from trial import config as C
from util_io import clean_load
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
save_pkl("../trial/data/index.pickle", index)
index = load_pkl("../trial/data/index.pickle")
