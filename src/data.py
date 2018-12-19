#!/usr/bin/env python3

from trial import config as C, paths as P, train as T
from util_io import pform, load_txt, save_txt
from util_np import np, vpack
from util_sp import spm, load_spm

path_da = pform(P.raw, "europarl-v7.da-en.da")
path_nl = pform(P.raw, "europarl-v7.nl-en.nl")
path_sv = pform(P.raw, "europarl-v7.sv-en.sv")
path_en = pform(P.raw, "europarl-v7.nl-en.en")

vocab_da = spm(pform(P.data, "vocab_da"), path_da, C.dim_src, C.bos, C.eos, C.unk)
vocab_nl = spm(pform(P.data, "vocab_nl"), path_nl, C.dim_src, C.bos, C.eos, C.unk)
vocab_sv = spm(pform(P.data, "vocab_sv"), path_sv, C.dim_src, C.bos, C.eos, C.unk)
vocab_en = spm(pform(P.data, "vocab_en"), path_en, C.dim_src, C.bos, C.eos, C.unk)

vocab_tgt = load_spm(pform(P.data, "vocab_en.model"))
for lang in 'de', 'sv', 'nl':
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


da_en = list(load_txt(pform(P.data, "da-en")))
de_en = list(load_txt(pform(P.data, "de-en")))
sv_en = list(load_txt(pform(P.data, "sv-en")))
nl_en = list(load_txt(pform(P.data, "nl-en")))

da_da = list(load_txt(pform(P.data, "da-da")))
de_de = list(load_txt(pform(P.data, "de-de")))
sv_sv = list(load_txt(pform(P.data, "sv-sv")))
nl_nl = list(load_txt(pform(P.data, "nl-nl")))

assert len(da_en) == len(da_da)
assert len(de_en) == len(de_de)
assert len(sv_en) == len(sv_sv)
assert len(nl_en) == len(nl_nl)

da = {s: i for i, s in enumerate(da_en)}
de = {s: i for i, s in enumerate(de_en)}
sv = {s: i for i, s in enumerate(sv_en)}
nl = {s: i for i, s in enumerate(nl_en)}

shared = []
for s in nl:
    if s in da and s in de and s in sv:
        shared.append((da[s], de[s], sv[s], nl[s]))

# todo
# - split testing data from shared
# - merge training data from da de sv
# - merge training data from nl
# - split validation data for da de sv
# - split validation data for nl
