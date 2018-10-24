#!/usr/bin/env python3

from model import Transformer as T
from trial import pform, path as P, config as C
from util import partial, select, PointedIndex
from util_io import encode, decode
from util_np import np
from util_tf import tf
import matplotlib.pyplot as plt

# load data
src = np.load(pform(P.data, P.valid_src, '.npy'))
tgt = np.load(pform(P.data, P.valid_tgt, '.npy'))
assert src.shape[1] <= C.cap_src
assert tgt.shape[1] <= C.cap_tgt

idx_src = PointedIndex(np.load(pform(P.data, P.index_src, '.npy')).item())
idx_tgt = PointedIndex(np.load(pform(P.data, P.index_tgt, '.npy')).item())
enc = partial(encode, idx_src)
dec = partial(decode, idx_tgt)

# load model
m = T.new(**select(C, *T._new)).data(**select(C, *T._data)).build(trainable= False)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, pform(P.ckpt, C.trial, C.ckpt))

trans = lambda s: dec(m.pred.eval({m.src_: np.array(enc(s)).reshape(1, -1)})[0])




# trained position embedding
emb_pos = m.emb_pos.kern.eval()

# how positions correlate with each other
plt.imshow(emb_pos @ emb_pos.T, cmap= 'gray')
plt.show()




# trained character embedding
emb_src = m.emb_src.kern.eval()

# how characters correlate with each other by frequencies
plt.imshow(emb_src @ emb_src.T, cmap= 'gray')
plt.show()




g = tf.get_default_graph()
# on the server, change Softmax to Reshape_1
e1 = g.get_tensor_by_name('.encode/layer1/att/attention/Softmax:0')
e2 = g.get_tensor_by_name('.encode/layer2/att/attention/Softmax:0')
d1 = g.get_tensor_by_name('.decode/layer1/att/attention/Softmax:0')
d2 = g.get_tensor_by_name('.decode/layer2/att/attention/Softmax:0')
b0 = g.get_tensor_by_name('.bridge/att/attention/Softmax:0')
att_tensors = e1, e2, d1, d2, b0




s = src[0:1]
att, prob, pred = sess.run((att_tensors, m.prob, m.pred), {m.src_: s})
ae1, ae2, ad1, ad2, ab0 = (a[0] for a in att)
prob = prob[0]
pred = pred[0]
print(dec(pred))

# encoder self-attention
plt.subplot(1, 2, 1)
plt.imshow(ae1, cmap= 'gray')
plt.subplot(1, 2, 2)
plt.imshow(ae2, cmap= 'gray')
plt.show()

# decoder self-attention
plt.subplot(1, 2, 1)
plt.imshow(ad1, cmap= 'gray')
plt.subplot(1, 2, 2)
plt.imshow(ad2, cmap= 'gray')
plt.show()

# bridge attention
plt.imshow(ab0, cmap= 'gray')
plt.show()

# highest probs
plt.plot(np.max(prob, -1))
plt.show()
