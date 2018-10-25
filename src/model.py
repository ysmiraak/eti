from util import Record, identity
from util_tf import QueryAttention as Attention
from util_tf import tf, placeholder, Normalize, Smooth, Dropout, Linear, Affine, Multilayer
import numpy as np


def sinusoid(time, dim, freq= 1e-4, scale= True, array= False):
    """returns a rank-2 tensor of shape `time, dim`, where each row
    corresponds to a time step and each column a sinusoid, with
    frequencies in a geometric progression from 1 to `freq`.

    """
    assert not dim % 2
    if array:
        a = (freq ** ((2 / dim) * np.arange(dim // 2))).reshape(-1, 1) @ (1 + np.arange(time).reshape(1, -1))
        s = np.concatenate((np.sin(a), np.cos(a)), -1).reshape(dim, time)
        if scale: s *= dim ** -0.5
        return s.T
    else:
        a = tf.reshape(
            freq ** ((2 / dim) * tf.range(dim // 2, dtype= tf.float32))
            , (-1, 1)) @ tf.reshape(
                1 + tf.range(tf.cast(time, tf.float32), dtype= tf.float32)
                , (1, -1))
        s = tf.reshape(tf.concat((tf.sin(a), tf.cos(a)), -1), (dim, time))
        if scale: s *= dim ** -0.5
        return tf.transpose(s)


class Sinusoid(Record):

    def __init__(self, dim, cap= None, name= 'sinusoid'):
        self.dim, self.name = dim, name
        with tf.variable_scope(name):
            self.pos = tf.constant(sinusoid(cap, dim, array= True), tf.float32) if cap else None

    def __call__(self, time, name= None):
        with tf.variable_scope(name or self.name):
            return sinusoid(time, self.dim) if self.pos is None else self.pos[:time]


class AttentionBlock(Record):

    def __init__(self, dim, dim_mid, act, name):
        with tf.variable_scope(name):
            self.name = name
            with tf.variable_scope('att'):
                self.att = Attention(dim, layer= Multilayer, mid= dim_mid, act= act)
                self.norm_att = Normalize(dim)
            with tf.variable_scope('mlp'):
                self.mlp = Multilayer(dim, dim, dim_mid, act)
                self.norm_mlp = Normalize(dim)

    def __call__(self, x, mask, dropout, w= None, name= None):
        if w is None: w = x
        with tf.variable_scope(name or self.name):
            with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, w, mask)))
            with tf.variable_scope('mlp'): x = self.norm_mlp(x + dropout(self.mlp(x)))
            return x


class Transformer(Record):
    """-> Record

    model = Transformer.new( ... )
    train = model.data( ... ).build().train()
    valid = model.data( ... ).build(trainable= False)

    """
    _new   = 'dim', 'dim_mid', 'depth', 'dim_src', 'dim_tgt', 'cap_tgt'
    _data  = 'cap_src',
    _build = 'dropout', 'smooth'
    _train = 'warmup', 'beta1', 'beta2', 'epsilon'

    @staticmethod
    def new(dim, dim_mid, depth, dim_src, dim_tgt, cap_tgt, act= tf.nn.relu):
        """-> Transformer with fields

        mask_tgt : f32 (1, t, t)
         emb_src : Linear
         emb_pos : Linear
          encode : tuple AttentionBlock
          decode : tuple AttentionBlock
          bridge : AttentionBlock
           logit : Affine

        """
        assert not dim % 2
        with tf.variable_scope('encode'):
            encode = tuple(AttentionBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(depth))
        with tf.variable_scope('decode'):
            mask_tgt = tf.log(tf.expand_dims(1 - tf.eye(cap_tgt), 0))
            decode = tuple(AttentionBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(depth))
        return Transformer(
            mask_tgt= mask_tgt
            , emb_src= Linear(dim, dim_src, 'emb_src')
            , emb_pos= Linear(dim, cap_tgt, 'emb_pos')
            , encode= encode
            , decode= decode
            , bridge= AttentionBlock(dim, dim_mid, act, "bridge")
            , logit= Affine(dim_tgt, dim, 'logit'))

    def data(self, src= None, tgt= None, cap_src= None, end= 1):
        """-> Transformer with new fields

             src : i32 (b, s)    source with `end` trimmed among the batch
            src_ : i32 (b, ?)    source feed, in range `[0, dim_src)`
            tgt_ : i32 (b, ?)    target feed, in range `[0, dim_tgt)`
            mask : f32 (b, 1, s) bridge mask
        mask_src : f32 (b, s, s) source mask
        position : Sinusoid

        setting `cap_src` makes it more efficient for training.  you
        won't be able to feed it longer sequences, but it doesn't
        affect any model parameters.

        """
        with tf.variable_scope('tgt'):
            tgt_ = placeholder(tf.int32, (None, None), tgt)
        with tf.variable_scope('src'):
            src_ = placeholder(tf.int32, (None, None), src)
            not_end = tf.to_float(tf.not_equal(src_, end))
            len_src = tf.reduce_sum(tf.to_int32(0 < tf.reduce_sum(not_end, 0)))
            not_end = tf.expand_dims(not_end[:,:len_src], 1)
            mask_src = tf.log(not_end + tf.expand_dims(1 - tf.eye(len_src), 0))
            mask = tf.log(not_end)
            src = src_[:,:len_src]
        return Transformer(
            src= src
            , src_= src_
            , tgt_= tgt_
            , mask= mask
            , mask_src= mask_src
            , position= Sinusoid(int(self.logit.kern.shape[0]), cap_src)
            , **self)

    def build(self, trainable= True, dropout= 0.1, smooth= 0.1):
        """-> Transformer with new fields

          output : f32 (b, t, dim_tgt) prediction on logit scale
            prob : f32 (b, t, dim_tgt) prediction, soft
            pred : i32 (b, t)          prediction, hard
            loss : f32 ()              prediction loss
             acc : f32 ()              accuracy

        and when `trainable`

         dropout : Dropout
          smooth : Smooth

        """
        if trainable:
            dim, dim_tgt = map(int, self.logit.kern.shape)
            dropout = Dropout(dropout, (None, None, dim))
            smooth  = Smooth(smooth, dim_tgt)
        else:
            dropout = smooth = identity
        # encoder input is source embedding + position encoding
        # dropout only the trained embedding
        with tf.variable_scope('.emb_src'):
            shape = tf.shape(self.src)
            w = self.position(shape[1]) + dropout(self.emb_src.embed(self.src))
        # decoder input is trained position embedding only
        with tf.variable_scope('.emb_pos'):
            x = dropout(tf.tile(tf.expand_dims(self.emb_pos.kern, 0), (shape[0], 1, 1)))
        # source mask disables current step and padding steps
        with tf.variable_scope('.encode'):
            for enc in self.encode: w = enc(w, self.mask_src, dropout)
        # bridge mask disables padding steps in source
        x = self.bridge(x, self.mask, dropout, w, '.bridge')
        # target mask disables current step
        with tf.variable_scope('.decode'):
            for dec in self.decode: x = dec(x, self.mask_tgt, dropout)
        y = self.logit(x, '.logit')
        with tf.variable_scope('.prob'): prob = tf.nn.softmax(y)
        with tf.variable_scope('.pred'): pred = tf.argmax(y, -1, output_type= tf.int32)
        with tf.variable_scope('.acc'): acc = tf.reduce_mean(tf.to_float(tf.equal(self.tgt_, pred)))
        with tf.variable_scope('.loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels= smooth(self.tgt_), logits= y)
                if trainable else
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels= self.tgt_, logits= y))
        self = Transformer(output= y, prob= prob, pred= pred, loss= loss, acc= acc, **self)
        if trainable: self.dropout, self.smooth = dropout, smooth
        return self

    def train(self, warmup= 4e3, beta1= 0.9, beta2= 0.98, epsilon= 1e-9):
        """-> Transformer with new fields

        step : i64 () global update step
          lr : f32 () learning rate for the current step
          up :        update operation

        """
        d = int(self.logit.kern.shape[0])
        with tf.variable_scope('lr'):
            s = tf.train.get_or_create_global_step()
            t = tf.to_float(s + 1)
            lr = (d ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(self.loss, s)
        return Transformer(step= s, lr= lr, up= up, **self)


# def trans(s, m= valid, idx= index):
#     return decode(idx, m.pred.eval({m.src_: enc([s])})[0])

# def trans_valid(m= valid, src= src_valid, bat= C.valid_batch):
#     for i, j in partition(len(src), bat, discard= False):
#         for p in dec(m.pred.eval({m.src_: src[i:j]})):
#             yield p
