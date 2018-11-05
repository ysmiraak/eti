#!/usr/bin/env python3

from model import Transformer as T, infer, translate
from tqdm import tqdm
from trial import config as C
from util import partial, select, PointedIndex
from util_io import path as P, pform, sieve, load_txt, load_pkl, save_txt
from util_np import np, batch, encode, decode
from util_tf import tf, pipe
tf.set_random_seed(C.seed)

#############
# load data #
#############

src_valid = np.load(pform(P.data, P.src))
tgt_valid = np.load(pform(P.data, P.tgt))
index = PointedIndex(load_pkl(pform(P.data, P.idx)))
enc = partial(encode, index, dtype= np.uint8)
dec = partial(decode, index)

def batch_load(path= pform(P.wmt, "corpus")
               , enc= enc
               , cap_src= C.cap_src
               , cap_tgt= C.cap_tgt
               , len_bat= C.batch_train
               , shuffle= C.shuffle
               , seed= C.seed):
    for src_tgt in batch(sieve(load_txt(path), cap_src, cap_tgt), len_bat, shuffle, seed):
        src, tgt = zip(*src_tgt)
        yield enc(src), enc(tgt, cap_tgt)

###############
# build model #
###############

model = T.new(**select(C, *T._new))
valid = model.data(**select(C, *T._data)).build(trainable= False)
trans = partial(translate, model= valid, index= index, batch= C.batch_valid)

# # for profiling
# m, src, tgt = valid, src_valid[:C.batch_valid], tgt_valid[:C.batch_valid]
# from util_tf import profile
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     with tf.summary.FileWriter(pform(P.log, C.trial), sess.graph) as wtr:
#         profile(sess, wtr, m.acc, feed_dict= {m.src_: src, m.tgt_: tgt})

src_train, tgt_train = pipe(batch_load, (tf.uint8, tf.uint8), prefetch= 16)
train = model.data(src= src_train, tgt= tgt_train, **select(C, *T._data)) \
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

for _ in range(9):
    for _ in range(200):
        for _ in tqdm(range(500), ncols= 70):
            sess.run(train.up)
        step = sess.run(train.step)
        summ(step)
    saver.save(sess, pform(P.ckpt, C.trial, step // 100000), write_meta_graph= False)
    save_txt(pform(P.pred, C.trial, step // 100000), trans(sess, src_valid))
