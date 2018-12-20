#!/usr/bin/env python3

from model import Model, batch_run
from tqdm import tqdm
from trial import config as C, paths as P, train as T
from util import partial, select
from util_io import pform, load_txt, save_txt
from util_np import np, partition, batch_sample
from util_sp import load_spm, encode, decode
from util_tf import tf, pipe
tf.set_random_seed(C.seed)

#############
# load data #
#############

valid_da_da = np.load(pform(P.data, "valid_da_da.npy"))
valid_da_en = np.load(pform(P.data, "valid_da_en.npy"))
valid_de_de = np.load(pform(P.data, "valid_de_de.npy"))
valid_de_en = np.load(pform(P.data, "valid_de_en.npy"))
valid_sv_en = np.load(pform(P.data, "valid_sv_en.npy"))
valid_sv_sv = np.load(pform(P.data, "valid_sv_sv.npy"))

train_da_da = np.load(pform(P.data, "train_da_da.npy"))
train_da_en = np.load(pform(P.data, "train_da_en.npy"))
train_de_de = np.load(pform(P.data, "train_de_de.npy"))
train_de_en = np.load(pform(P.data, "train_de_en.npy"))
train_sv_en = np.load(pform(P.data, "train_sv_en.npy"))
train_sv_sv = np.load(pform(P.data, "train_sv_sv.npy"))

def batch_valid(src, tgt, size= 256):
    assert 1024 == len(src) == len(tgt)
    while True:
        for i, j in partition(1024, size):
            yield src[i:j], tgt[i:j]

def batch_train(src, tgt, size= 18):
    assert len(src) == len(tgt)
    for i in batch_sample(len(src), size):
        yield src[i], tgt[i]

###############
# build model #
###############

model = Model.new(**select(C, *Model._new))

da_da, da_en = pipe(partial(batch_valid, valid_da_da, valid_da_en), (tf.int32, tf.int32), prefetch= 4)
de_de, de_en = pipe(partial(batch_valid, valid_de_de, valid_de_en), (tf.int32, tf.int32), prefetch= 4)
sv_sv, sv_en = pipe(partial(batch_valid, valid_sv_sv, valid_sv_en), (tf.int32, tf.int32), prefetch= 4)

valid = model.data(1, 0, da_da, da_en).valid() \
    ,   model.data(2, 0, de_de, de_en).valid() \
    ,   model.data(3, 0, sv_sv, sv_en).valid() \
    ,   model.data(0, 3, sv_en, sv_sv).valid() \
    ,   model.data(0, 2, de_en, de_de).valid() \
    ,   model.data(0, 1, da_en, da_da).valid()

da_da, da_en = pipe(partial(batch_train, train_da_da, train_da_en), (tf.int32, tf.int32), prefetch= 16)
de_de, de_en = pipe(partial(batch_train, train_de_de, train_de_en), (tf.int32, tf.int32), prefetch= 16)
sv_sv, sv_en = pipe(partial(batch_train, train_sv_sv, train_sv_en), (tf.int32, tf.int32), prefetch= 16)

train = model.data(1, 0, da_da, da_en).train(**T) \
    ,   model.data(2, 0, de_de, de_en).train(**T) \
    ,   model.data(3, 0, sv_sv, sv_en).train(**T) \
    ,   model.data(0, 3, sv_en, sv_sv).train(**T) \
    ,   model.data(0, 2, de_en, de_de).train(**T) \
    ,   model.data(0, 1, da_en, da_da).train(**T)

model.lr   = train[0].lr
model.step = train[0].step
model.errt = train[0].errt
model.loss = tf.add_n([t.loss for t in train])
model.down = tf.train.AdamOptimizer(model.lr, T.beta1, T.beta2, T.epsilon).minimize(model.loss, model.step)

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
             ( tf.summary.scalar('step_errt', model.errt)
             , tf.summary.scalar('step_loss', model.loss)))):
    errt_loss = []
    for m in valid:
        for _ in range(4): # 4 * 256 = 1024
            errt_loss.append(sess.run((m.errt, m.loss)))
    errt, loss = map(np.mean, zip(*errt_loss))
    wtr.add_summary(sess.run(summary, {model.errt: errt, model.loss: loss}), step)
    wtr.flush()

for _ in range(9): # 1 epoch per round
    for _ in range(200):
        for _ in tqdm(range(500), ncols= 70):
            sess.run(model.down)
        step = sess.run(model.step)
        summ(step)
    saver.save(sess, pform(P.ckpt, C.trial, step // 100000), write_meta_graph= False)
