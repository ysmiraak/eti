#!/usr/bin/env python3

from model import Transformer as T
from tqdm import tqdm
from trial import pform, path as P, config as C
from util import select, PointedIndex
from util_io import encode, decode, save
from util_np import np, partition, sample
from util_tf import tf, pipe
tf.set_random_seed(0)

#############
# load data #
#############

src_train = np.load(pform(P.data, P.train_src, '.npy'))
tgt_train = np.load(pform(P.data, P.train_tgt, '.npy'))
src_valid = np.load(pform(P.data, P.valid_src, '.npy'))
tgt_valid = np.load(pform(P.data, P.valid_tgt, '.npy'))
assert src_train.shape[1] <= C.cap_src
assert tgt_train.shape[1] <= C.cap_tgt
assert src_valid.shape[1] <= C.cap_src
assert tgt_valid.shape[1] <= C.cap_tgt

###############
# build model #
###############

model = T.new(**select(C, *T._new))
valid = model.data(**select(C, *T._data)).build(trainable= False)

# # for profiling
# m, src, tgt = valid, src_valid[:C.valid_batch], tgt_valid[:C.valid_batch]
# from util_tf import profile
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     with tf.summary.FileWriter(pform(P.log, "graph"), sess.graph) as wtr:
#         profile(sess, wtr, m.acc, feed_dict= {m.src_: src, m.tgt_: tgt})

src, tgt = pipe(
    lambda: ((src_train[bat], tgt_train[bat]) for bat in sample(len(src_train), C.train_batch))
    , (tf.uint8, tf.uint8))

train = model.data(src= src, tgt= tgt, **select(C, *T._data)) \
             .build(**select(C, *T._build)) \
             .train(**select(C, *T._train))

##############
# validation #
##############

src_index = PointedIndex(np.load(pform(P.data, P.index_src, '.npy')).item())
tgt_index = PointedIndex(np.load(pform(P.data, P.index_tgt, '.npy')).item())

def trans(s, m= valid, idx_src= src_index, idx_tgt= tgt_index):
    src = np.array(encode(idx_src, s)).reshape(1, -1)
    return decode(idx_tgt, m.pred.eval({m.src_: src})[0])

def trans_valid(m= valid, src= src_valid, idx= tgt_index, bat= C.valid_batch):
    for i, j in partition(len(src), bat, discard= False):
        for p in m.pred.eval({m.src_: src[i:j]}):
            yield decode(idx, p)

def summ(m= valid
         , src= src_valid[:C.valid_total]
         , tgt= tgt_valid[:C.valid_total]
         , bat= C.valid_batch
         , summary= tf.summary.merge(
             (tf.summary.scalar('step_loss', valid.loss)
              , tf.summary.scalar('step_acc', valid.acc)))):
    loss, acc = zip(*(
        sess.run((m.loss, m.acc), {m.src_: src[i:j], m.tgt_: tgt[i:j]})
        for i, j in partition(len(src), bat, discard= False)))
    return sess.run(summary, {m.loss: np.mean(loss), m.acc: np.mean(acc)})

############
# training #
############

saver = tf.train.Saver()
sess = tf.InteractiveSession()
wtr = tf.summary.FileWriter(pform(P.log, C.trial))

if C.ckpt:
    saver.restore(sess, pform(P.ckpt, C.trial, C.ckpt))
else:
    tf.global_variables_initializer().run()

for _ in range(36):
    for _ in range(len(src_train) // C.train_batch // C.valid_inter):
        for _ in tqdm(range(C.valid_inter), ncols= 70):
            sess.run(train.up)
        step = sess.run(train.step)
        wtr.add_summary(summ(), step)
    saver.save(sess, pform(P.ckpt, C.trial, step), write_meta_graph= False)
    save(pform(P.pred, C.trial, step), trans_valid())
