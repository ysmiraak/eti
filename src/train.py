#!/usr/bin/env python3

from model import Transformer as T, infer, translate
from tqdm import tqdm
from trial import pform, path as P, config as C
from util import partial, select, PointedIndex
from util_io import sieve, batch, load_wmt, load_pkl, save_txt
from util_np import np, partition, encode, decode
from util_tf import tf, pipe
tf.set_random_seed(C.seed)

#############
# load data #
#############

src_valid = np.load("../trial/data/source.npy")
tgt_valid = np.load("../trial/data/target.npy")
index = PointedIndex(load_pkl("../trial/data/index"))
enc = partial(encode, index, dtype= np.uint8)
dec = partial(decode, index)

def batch_load(path= "/data/wmt/de-en/corpus"
               , enc= enc
               , cap_src= C.cap_src
               , cap_tgt= C.cap_tgt
               , len_bat= C.train_batch
               , shuffle= C.shuffle
               , seed= C.seed):
    for src, tgt in batch(sieve(load_wmt(path), cap_src, cap_tgt), len_bat, shuffle, seed):
        # currently tgt must always be as long as cap_tgt but src can be shorter
        yield enc(src), enc(tgt, cap_tgt)

###############
# build model #
###############

model = T.new(**select(C, *T._new))
valid = model.data(**select(C, *T._data)).build(trainable= False)
trans = partial(translate, model= valid, index= index, batch= C.valid_batch)

# # for profiling
# m, src, tgt = valid, src_valid[:C.valid_batch], tgt_valid[:C.valid_batch]
# from util_tf import profile
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     with tf.summary.FileWriter(pform(P.log, "graph"), sess.graph) as wtr:
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

wtr = tf.summary.FileWriter(pform(P.log, C.trial))
summary = tf.summary.merge(
    (tf.summary.scalar('step_loss', valid.loss)
     , tf.summary.scalar('step_acc', valid.acc)))

def summ(step):
    loss, acc = map(np.mean, zip(*infer(
        model= valid
        , fetches= (valid.loss, valid.acc)
        , src= src_valid[:C.valid_total]
        , tgt= tgt_valid[:C.valid_total]
        , batch= C.valid_batch)))
    wtr.add_summary(sess.run(summary, {m.loss: loss, m.acc: acc}), step)

def save(step):
    saver.save(sess, pform(P.ckpt, C.trial, step), write_meta_graph= False)
    save_txt(pform(P.pred, C.trial, step), trans(src))

try:
    for _ in range(200):
        for _ in tqdm(range(500), ncols= 70):
            sess.run(train.up)
        step = sess.run(train.step)
        summ(step)
    save(step // 100000)
except tf.errors.OutOfRangeError:
    save(step)
