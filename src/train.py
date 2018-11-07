#!/usr/bin/env python3

from model import Transformer as T, infer, translate
from tqdm import tqdm
from trial import config as C
from util import partial, select, PointedIndex
from util_io import path as P, pform, sieve, load_txt, save_txt, load_pkl
from util_np import np, sample, encode, decode
from util_tf import tf, pipe
tf.set_random_seed(C.seed)

#############
# load data #
#############

src_tgt = list(sieve(
    zip(load_txt(pform(P.raw, "europarl-v7.de-en.de"))
        , load_txt(pform(P.raw, "europarl-v7.de-en.en")))
    , C.cap))
np.random.seed(C.seed)
np.random.shuffle(src_tgt)
src_valid, tgt_valid = zip(*src_tgt[:C.batch_valid * 5])
src_train, tgt_train = zip(*src_tgt[C.batch_valid * 5:])
del src_tgt

# from collections import Counter
# from itertools import chain
# from util import diter
# from util_io import save_pkl, vocab
# chars = Counter(chain(diter(src_train, 2), diter(tgt_train, 2)))
# index = "".join(vocab(chars, specials= "\xa0\n "))
# save_pkl(pform(P.data, P.idx), index)
# save_txt(pform(P.data, P.src), src_valid)
# save_txt(pform(P.data, P.tgt), tgt_valid)

idx = PointedIndex(load_pkl(pform(P.data, P.idx)))
enc = partial(encode, idx, length= C.cap, dtype= np.uint8)
dec = partial(decode, idx)

src_valid = enc(src_valid)
tgt_valid = enc(tgt_valid)
src_train = enc(src_train)
tgt_train = enc(tgt_train)

###############
# build model #
###############

model = T.new(**select(C, *T._new))
modat = model.data(**select(C, *T._data))
valid = modat.build(trainable= False)
infer = modat.infer(**select(C, *T._infer))
trans = partial(translate, model= infer, index= idx, batch= C.batch_valid)

# # for profiling
# m, src, tgt = modat, src_valid[:C.batch_valid], tgt_valid[:C.batch_valid]
# from util_tf import profile
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     with tf.summary.FileWriter(pform(P.log, C.trial), sess.graph) as wtr:
#         profile(sess, wtr, (infer.pred, valid.acc), feed_dict= {m.src_: src, m.tgt_: tgt})

src, tgt = pipe(
    lambda: ((src_train[bat], tgt_train[bat])
             for bat in sample(len(src_train), C.batch_train, C.seed))
    , (tf.uint8, tf.uint8)
    , prefetch= 16)

train = model.data(src= src, tgt= tgt, **select(C, *T._data)) \
             .build(**select(C, *T._build)) \
             .train(**select(C, *T._train))

############
# training #
############

sess = tf.InteractiveSession()
saver = tf.train.Saver()
if C.ckpt:
    saver.restore(sess, pform(P.ckpt, C.trial, C.ckpt))
else:
    tf.global_variables_initializer().run()

def summ(step, wtr = tf.summary.FileWriter(pform(P.log, C.trial))
         , summary = tf.summary.merge(
             (tf.summary.scalar('step_loss', valid.loss)
              , tf.summary.scalar('step_acc', valid.acc)))):
    loss, acc = map(np.mean, zip(*infer(
        sess= sess
        , model= valid
        , fetches= (valid.loss, valid.acc)
        , src= src_valid
        , tgt= tgt_valid
        , batch= C.batch_valid)))
    wtr.add_summary(sess.run(summary, {valid.loss: loss, valid.acc: acc}), step)

for _ in range(5):
    for _ in range(400):
        for _ in tqdm(range(250), ncols= 70):
            sess.run(train.up)
        step = sess.run(train.step)
        summ(step)
    saver.save(sess, pform(P.ckpt, C.trial, step // 100000), write_meta_graph= False)
    save_txt(pform(P.pred, C.trial, step // 100000), trans(sess, src_valid))
