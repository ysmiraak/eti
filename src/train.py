#!/usr/bin/env python3

from model import Model, batch_run
from tqdm import tqdm
from trial import config as C, paths as P, train as T
from util import partial, select
from util_io import pform, load_txt, save_txt, load_pkl
from util_np import np, vpack, sample
from util_sp import load_spm, encode, decode
from util_tf import tf, pipe
tf.set_random_seed(C.seed)

#############
# load data #
#############

vocab_src = load_spm(pform(P.data, "vocab_src.model"))
vocab_tgt = load_spm(pform(P.data, "vocab_tgt.model"))
src_valid = np.load(pform(P.data, "valid_src.npy"))
tgt_valid = np.load(pform(P.data, "valid_tgt.npy"))

def batch(batch= C.batch_train
          , seed= C.seed
          , vocab_src= vocab_src, path_src= pform(P.data, "train_src.txt")
          , vocab_tgt= vocab_tgt, path_tgt= pform(P.data, "train_tgt.txt")
          , eos= C.eos
          , cap= C.cap):
    src, tgt = tuple(load_txt(path_src)), tuple(load_txt(path_tgt))
    bas, bat = [], []
    for i in sample(len(src), seed):
        if batch == len(bas):
            yield vpack(bas, (batch, cap), eos, np.int32) \
                , vpack(bat, (batch, cap), eos, np.int32)
            bas, bat = [], []
        # s = vocab_src.sample_encode_as_ids(src[i], -1, 0.1)
        # t = vocab_tgt.sample_encode_as_ids(tgt[i], -1, 0.1)
        s = vocab_src.encode_as_ids(src[i])
        t = vocab_tgt.encode_as_ids(tgt[i])
        if 0 < len(s) <= cap and 0 < len(t) <= cap:
            bas.append(s)
            bat.append(t)

###############
# build model #
###############

model = Model.new(**select(C, *Model._new))
modat = model.data()
valid = modat.valid()
infer = modat.infer()

# # for profiling
# m, src, tgt = modat, src_valid[:32], tgt_valid[:32]
# from util_tf import profile
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     with tf.summary.FileWriter(pform(P.log, C.trial), sess.graph) as wtr:
#         profile(sess, wtr, (infer.pred, valid.errt), feed_dict= {m.src_: src, m.tgt_: tgt})

train = model.data(*pipe(batch, (tf.int32, tf.int32), prefetch= 16)).train(**T)

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
             ( tf.summary.scalar('step_loss', valid.loss)
             , tf.summary.scalar('step_errt', valid.errt)))):
    loss, errt = map(np.mean, zip(*batch_run(
        sess= sess
        , model= valid
        , fetch= (valid.loss, valid.errt)
        , src= src_valid
        , tgt= tgt_valid
        , batch= C.batch_valid)))
    wtr.add_summary(sess.run(summary, {valid.loss: loss, valid.errt: errt}), step)
    wtr.flush()

def trans(sents, model= infer):
    if not isinstance(sents, np.ndarray):
        sents = encode(vocab_src, sents, C.cap, np.int32)
    for preds in batch_run(sess, model, model.pred, sents, batch= C.batch_infer):
        yield from decode(vocab_tgt, preds)

for _ in range(1):
    for _ in range(400):
        for _ in tqdm(range(250), ncols= 70):
            sess.run(train.up)
        step = sess.run(train.step)
        summ(step)
    saver.save(sess, pform(P.ckpt, C.trial, step // 100000), write_meta_graph= False)
    save_txt(pform(P.pred, C.trial, step // 100000), trans(src_valid))
