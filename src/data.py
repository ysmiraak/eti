#!/usr/bin/env python3

from collections import defaultdict
from trial import config as C, paths as P, train as T
from util_io import pform, load_txt, save_txt
from util_np import np, vpack
from util_sp import spm, load_spm, encode

# train one spm for each lang
vocab_da = spm(pform(P.data, "vocab_da"), pform(P.raw, "europarl-v7.da-en.da"), C.dim_voc, C.bos, C.eos, C.unk)
vocab_de = spm(pform(P.data, "vocab_de"), pform(P.raw, "europarl-v7.de-en.de"), C.dim_voc, C.bos, C.eos, C.unk)
vocab_sv = spm(pform(P.data, "vocab_sv"), pform(P.raw, "europarl-v7.sv-en.sv"), C.dim_voc, C.bos, C.eos, C.unk)
vocab_nl = spm(pform(P.data, "vocab_nl"), pform(P.raw, "europarl-v7.nl-en.nl"), C.dim_voc, C.bos, C.eos, C.unk)
vocab_en = spm(pform(P.data, "vocab_en"), pform(P.raw, "europarl-v7.nl-en.en"), C.dim_voc, C.bos, C.eos, C.unk)

# remove long sentences
vocab_tgt = load_spm(pform(P.data, "vocab_en.model"))
for lang in 'da', 'de', 'sv', 'nl':
    src , tgt = [], []
    vocab_src = load_spm(pform(P.data, "vocab_{}.model".format(lang)))
    for sr, tr in zip(
              load_txt(pform(P.raw, "europarl-v7.{}-en.{}".format(lang, lang)))
            , load_txt(pform(P.raw, "europarl-v7.{}-en.en".format(lang)))):
        s = vocab_src.encode_as_ids(sr)
        t = vocab_tgt.encode_as_ids(tr)
        if 0 < len(s) <= C.cap and 0 < len(t) <= C.cap:
            src.append(sr)
            tgt.append(tr)
    save_txt(pform(P.data, "{}-{}".format(lang, lang)), src)
    save_txt(pform(P.data, "{}-en".format(lang      )), tgt)

# load filtered foreign sentences
da_da = list(load_txt(pform(P.data, "da-da")))
de_de = list(load_txt(pform(P.data, "de-de")))
sv_sv = list(load_txt(pform(P.data, "sv-sv")))
nl_nl = list(load_txt(pform(P.data, "nl-nl")))

# load filtered english sentences
da_en = list(load_txt(pform(P.data, "da-en")))
de_en = list(load_txt(pform(P.data, "de-en")))
sv_en = list(load_txt(pform(P.data, "sv-en")))
nl_en = list(load_txt(pform(P.data, "nl-en")))

# track unique instances on the english side
da = defaultdict(list)
de = defaultdict(list)
sv = defaultdict(list)
nl = defaultdict(list)
for i, s in enumerate(da_en): da[s].append(i)
for i, s in enumerate(de_en): de[s].append(i)
for i, s in enumerate(sv_en): sv[s].append(i)
for i, s in enumerate(nl_en): nl[s].append(i)
da = {s: i[0] for s, i in da.items() if 1 == len(i)}
de = {s: i[0] for s, i in de.items() if 1 == len(i)}
sv = {s: i[0] for s, i in sv.items() if 1 == len(i)}
nl = {s: i[0] for s, i in nl.items() if 1 == len(i)}

# shared unique instances
shared = []
for s in nl:
    if s in da and s in de and s in sv:
        shared.append((da[s], de[s], sv[s], nl[s]))
np.random.seed(0)
np.random.shuffle(shared)

# split test set
test = shared[:C.total_valid]
test_en, test_da, test_de, test_sv, test_nl = [], [], [], [], []
for da, de, sv, nl in test:
    assert da_en[da] == de_en[de] == sv_en[sv] == nl_en[nl]
    test_en.append(nl_en[nl])
    test_da.append(da_da[da])
    test_de.append(de_de[de])
    test_sv.append(sv_sv[sv])
    test_nl.append(nl_nl[nl])
save_txt(pform(P.data, "test_en"), test_en)
save_txt(pform(P.data, "test_da"), test_da)
save_txt(pform(P.data, "test_de"), test_de)
save_txt(pform(P.data, "test_sv"), test_sv)
save_txt(pform(P.data, "test_nl"), test_nl)

# remove test instances
da = {t[0] for t in test}
de = {t[1] for t in test}
sv = {t[2] for t in test}
nl = {t[3] for t in test}
da_da = [s for i, s in enumerate(da_da) if i not in da]
da_en = [s for i, s in enumerate(da_en) if i not in da]
de_de = [s for i, s in enumerate(de_de) if i not in de]
de_en = [s for i, s in enumerate(de_en) if i not in de]
sv_sv = [s for i, s in enumerate(sv_sv) if i not in sv]
sv_en = [s for i, s in enumerate(sv_en) if i not in sv]
nl_nl = [s for i, s in enumerate(nl_nl) if i not in nl]
nl_en = [s for i, s in enumerate(nl_en) if i not in nl]

np.random.seed(1)
np.random.shuffle(da_da)
np.random.seed(1)
np.random.shuffle(da_en)
np.random.seed(2)
np.random.shuffle(de_de)
np.random.seed(2)
np.random.shuffle(de_en)
np.random.seed(3)
np.random.shuffle(sv_sv)
np.random.seed(3)
np.random.shuffle(sv_en)
np.random.seed(4)
np.random.shuffle(nl_nl)
np.random.seed(4)
np.random.shuffle(nl_en)

valid_da_da = da_da[:1024]
valid_da_en = da_en[:1024]
valid_de_de = de_de[:1024]
valid_de_en = de_en[:1024]
valid_sv_sv = sv_sv[:1024]
valid_sv_en = sv_en[:1024]
valid_nl_nl = nl_nl[:2048]
valid_nl_en = nl_en[:2048]

train_da_da = da_da[1024:]
train_da_en = da_en[1024:]
train_de_de = de_de[1024:]
train_de_en = de_en[1024:]
train_sv_sv = sv_sv[1024:]
train_sv_en = sv_en[1024:]
train_nl_nl = nl_nl[2048:]
train_nl_en = nl_en[2048:]

vocab_en = load_spm(pform(P.data, "vocab_en.model"))
vocab_da = load_spm(pform(P.data, "vocab_da.model"))
vocab_de = load_spm(pform(P.data, "vocab_de.model"))
vocab_sv = load_spm(pform(P.data, "vocab_sv.model"))
vocab_nl = load_spm(pform(P.data, "vocab_nl.model"))

np.save(pform(P.data, "valid_da_da.npy"), encode(vocab_da, valid_da_da, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "valid_da_en.npy"), encode(vocab_en, valid_da_en, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "valid_de_de.npy"), encode(vocab_de, valid_de_de, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "valid_de_en.npy"), encode(vocab_en, valid_de_en, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "valid_sv_sv.npy"), encode(vocab_sv, valid_sv_sv, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "valid_sv_en.npy"), encode(vocab_en, valid_sv_en, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "valid_nl_nl.npy"), encode(vocab_nl, valid_nl_nl, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "valid_nl_en.npy"), encode(vocab_en, valid_nl_en, length= C.cap, dtype= np.uint16))

np.save(pform(P.data, "train_da_da.npy"), encode(vocab_da, train_da_da, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "train_da_en.npy"), encode(vocab_en, train_da_en, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "train_de_de.npy"), encode(vocab_de, train_de_de, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "train_de_en.npy"), encode(vocab_en, train_de_en, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "train_sv_sv.npy"), encode(vocab_sv, train_sv_sv, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "train_sv_en.npy"), encode(vocab_en, train_sv_en, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "train_nl_nl.npy"), encode(vocab_nl, train_nl_nl, length= C.cap, dtype= np.uint16))
np.save(pform(P.data, "train_nl_en.npy"), encode(vocab_en, train_nl_en, length= C.cap, dtype= np.uint16))
