from model import Transformer as T
from trial import config as C
from util import partial, select, PointedIndex
from util_io import path as P, pform, load_pkl
from util_np import np, encode, decode
from util_tf import tf
import matplotlib.pyplot as plt

# load data
src = np.load(pform(P.data, P.src))
tgt = np.load(pform(P.data, P.tgt))
assert src.shape[1] <= C.cap_src
assert tgt.shape[1] <= C.cap_tgt
index = PointedIndex(load_pkl(pform(P.data, P.idx)))
enc = partial(encode, index, dtype= np.uint8)
dec = partial(decode, index)

# load model
m = T.new(**select(C, *T._new)).data(**select(C, *T._data)).build(trainable= False)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, pform(P.ckpt, C.trial, C.ckpt))

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
e1 = g.get_tensor_by_name('encode_/layer1/att/attention/Softmax:0')
e2 = g.get_tensor_by_name('encode_/layer2/att/attention/Softmax:0')
d1 = g.get_tensor_by_name('decode_/layer1/att/attention/Softmax:0')
d2 = g.get_tensor_by_name('decode_/layer2/att/attention/Softmax:0')
b0 = g.get_tensor_by_name('bridge_/att/attention/Softmax:0')
att_tensors = e1, e2, d1, d2, b0

s = src[7:8]
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
