#!/usr/bin/env python3


trial      = 'm'
len_cap    = 2**8
batch_size = 2**6
valid_size = batch_size * 4
summ_size  = valid_size * 4
step_summ  = 2**8
ckpt       = None


from model import Transformer
from os.path import expanduser, join
from tqdm import tqdm
from util import PointedIndex
from util_io import encode, decode, save
from util_np import np, partition, sample
from util_tf import tf, pipe

logdir = expanduser("~/cache/tensorboard-logdir/eti")
tf.set_random_seed(0)

###############
# preparation #
###############

src_train = np.load("../trial/data/train_src.npy")
tgt_train = np.load("../trial/data/train_tgt.npy")
src_valid = np.load("../trial/data/valid_src.npy")
tgt_valid = np.load("../trial/data/valid_tgt.npy")
assert src_train.shape[1] <= len_cap
assert tgt_train.shape[1] <= len_cap
assert src_valid.shape[1] <= len_cap
assert tgt_valid.shape[1] <= len_cap

# # for profiling
# from util_tf import profile
# m = Transformer.new().data().build(trainable= False)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     with tf.summary.FileWriter(join(logdir, "graph"), sess.graph) as wtr:
#         profile(sess, wtr, m.acc, {m.src_: src_valid[:batch_size], m.tgt_: tgt_valid[:batch_size]})

###############
# build model #
###############

def batch(src, tgt, size= batch_size, dtype= tf.uint8):
    return pipe(
        lambda: ((src[bat], tgt[bat]) for bat in sample(len(src), size))
        , (dtype, dtype))

model = Transformer.new()
valid = model.data(src_cap= len_cap).build(trainable= False)
train = model.data(*batch(src_train, tgt_train), len_cap).build().train()

idx_src = PointedIndex(np.load("../trial/data/index_src.npy").item())
idx_tgt = PointedIndex(np.load("../trial/data/index_tgt.npy").item())

def trans(s, m= valid, idx_src= idx_src, idx_tgt= idx_tgt):
    src = np.array(encode(idx_src, s)).reshape(1, -1)
    return decode(idx_tgt, m.pred.eval({m.src: src})[0])

def trans_valid(m= valid, src= src_valid, idx= idx_tgt, batch_size= valid_size):
    for i, j in partition(len(src), batch_size, discard= False):
        for p in m.pred.eval({m.src_: src[i:j]}):
            yield decode(idx, p)

############
# training #
############

saver = tf.train.Saver()
sess = tf.InteractiveSession()
wtr = tf.summary.FileWriter(join(logdir, "{}".format(trial)))
if ckpt:
    saver.restore(sess, "../trial/model/{}{}".format(trial, ckpt))
else:
    tf.global_variables_initializer().run()

def summ(m= valid
         , src= src_valid[:summ_size]
         , tgt= tgt_valid[:summ_size]
         , batch_size= valid_size
         , summary= tf.summary.merge(
             (tf.summary.scalar('step_loss', valid.loss)
              , tf.summary.scalar('step_acc', valid.acc)))):
    loss, acc = zip(*(
        sess.run((m.loss, m.acc), {m.src_: src[i:j], m.tgt_: tgt[i:j]})
        for i, j in partition(len(src), batch_size, discard= False)))
    return sess.run(summary, {m.loss: np.mean(loss), m.acc: np.mean(acc)})

while True:
    for _ in range(len(src_train) // batch_size // step_summ):
        for _ in tqdm(range(step_summ), ncols= 70):
            sess.run(train.up)
        step = sess.run(train.step)
        wtr.add_summary(summ(), step)
    saver.save(sess, "../trial/model/{}{}".format(trial, step), write_meta_graph= False)
    save("../trial/pred/{}{}".format(trial, step), trans_valid())
