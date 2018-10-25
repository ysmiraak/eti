#!/usr/bin/env python3

from model import Transformer as T
from tqdm import tqdm
from trial import pform, path as P, config as C
from util import partial, select, PointedIndex
from util_io import batch_load, load_pkl, save_txt
from util_np import np, partition, encode, decode
from util_tf import tf, pipe
tf.set_random_seed(C.seed)

#############
# load data #
#############

index = PointedIndex(load_pkl("../trial/data/index"))
assert 1 == index("\n")
enc = partial(encode, index, dtype= np.uint8)
dec = partial(decode, index)

def batch(enc= enc, kwargs= {
        filename= "/data/wmt/de-en/corpus"
        , cap_src= C.cap_src
        , cap_tgt= C.cap_tgt
        , len_bat= C.train_batch
        , shuffle= C.shuffle
        , seed= C.seed}):
    for src, tgt in batch_load(**kwargs):
        # currently tgt must always be as long as cap_tgt
        # but src can be shorter
        yield enc(src), enc(tgt, cap_tgt)

####################
# validation model #
####################

model = T.new(**select(C, *T._new))
valid = model.data(**select(C, *T._data)).build(trainable= False)

# # for profiling
# m, src, tgt = valid, src_valid[:C.valid_batch], tgt_valid[:C.valid_batch]
# from util_tf import profile
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     with tf.summary.FileWriter(pform(P.log, "graph"), sess.graph) as wtr:
#         profile(sess, wtr, m.acc, feed_dict= {m.src_: src, m.tgt_: tgt})

trans = lambda s: decode(index, valid.pred.eval({valid.src_: enc([s])})[0])

# todo validation data
def trans_valid(m= valid, src= src_valid, bat= C.valid_batch):
    for i, j in partition(len(src), bat, discard= False):
        for p in dec(m.pred.eval({m.src_: src[i:j]})):
            yield p

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

##################
# training model #
##################

src, tgt = pipe(batch, (tf.uint8, tf.uint8), prefetch= 13)
train = model.data(src= src, tgt= tgt, **select(C, *T._data)) \
             .build(**select(C, *T._build)) \
             .train(**select(C, *T._train))

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
    save_txt(pform(P.pred, C.trial, step), trans_valid())
