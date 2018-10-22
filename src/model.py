from util import Record, identity
from util_tf import QueryAttention as Attention
from util_tf import tf, placeholder, Normalize, Smooth, Dropout, Linear, Affine, Multilayer
import numpy as np


def sinusoid(time, dim, freq= 1e-4, name= 'sinusoid', scale= True, array= False):
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
    with tf.variable_scope(name):
        a = tf.reshape(
            freq ** ((2 / dim) * tf.range(dim // 2, dtype= tf.float32))
            , (-1, 1)) @ tf.reshape(
                1 + tf.range(tf.cast(time, tf.float32), dtype= tf.float32)
                , (1, -1))
        s = tf.reshape(tf.concat((tf.sin(a), tf.cos(a)), -1), (dim, time))
        if scale: s *= dim ** -0.5
        return tf.transpose(s)


class Sinusoid(Record):

    def __init__(self, dim, len_cap= None, name= 'sinusoid'):
        self.dim, self.name = dim, name
        self.pos = tf.constant(
            sinusoid(len_cap, dim, array= True), tf.float32, name= name
        ) if len_cap else None

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
            with tf.variable_scope('fwd'):
                self.fwd = Multilayer(dim, dim, dim_mid, act)
                self.norm_fwd = Normalize(dim)

    def __call__(self, x, mask, dropout, w= None, name= None):
        if w is None: w = x
        with tf.variable_scope(name or self.name):
            with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, w, mask)))
            with tf.variable_scope('fwd'): x = self.norm_fwd(x + dropout(self.fwd(x)))
            return x


class Transformer(Record):
    """-> Record

    model = Transformer.new()
    train = model.data(src_train, tgt_train, src_cap).build().train()
    valid = model.data(src_valid, tgt_valid).build(trainable= False)

    """

    @staticmethod
    def new(end= 1
            , tgt_cap= 256
            , dim_src= 256, dim= 256
            , dim_tgt= 256, dim_mid= 512, num_layer= 2
            , act= tf.nn.relu
            , smooth= 0.1
            , dropout= 0.1):
        """-> Transformer with fields

            end : i32 ()
        emb_src : Linear
        emb_pos : Linear
         encode : tuple AttentionBlock
         decode : tuple AttentionBlock
         bridge : AttentionBlock
          logit : Affine
         smooth : Smooth
        dropout : Dropout

        `end` is treated as the padding for both source and target.

        """
        assert not dim % 2
        with tf.variable_scope('encode'):
            encode = tuple(AttentionBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(num_layer))
        with tf.variable_scope('decode'):
            decode = tuple(AttentionBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(num_layer))
        return Transformer(
            dim= dim, end= tf.constant(end, tf.int32, (), 'end')
            , encode= encode, emb_src= Linear(dim, dim_src, 'emb_src')
            , decode= decode, emb_pos= Linear(dim, tgt_cap, 'emb_pos')
            , bridge= AttentionBlock(dim, dim_mid, act, "bridge")
            , logit= Affine(dim_tgt, dim, 'logit')
            , smooth= Smooth(smooth, dim_tgt)
            , dropout= Dropout(dropout, (None, None, dim)))

    def data(self, src= None, tgt= None, src_cap= None):
        """-> Transformer with new fields

            tgt_ : i32 (b, ?) target feed, in range `[0, dim_tgt)`
            src_ : i32 (b, ?) source feed, in range `[0, dim_src)`
             src : i32 (b, s) source with `end` trimmed among the batch
            mask : f32 (b, s) source mask
        position : Sinusoid

        setting `src_cap` makes it more efficient for training.  you
        won't be able to feed it longer sequences, but it doesn't
        affect any model parameters.

        """
        end, dim = self.end, self.dim
        with tf.variable_scope('tgt'):
            tgt_ = placeholder(tf.int32, (None, None), tgt)
        with tf.variable_scope('src'):
            src = src_ = placeholder(tf.int32, (None, None), src)
            len_src = tf.reduce_sum(tf.to_int32(~ tf.reduce_all(tf.equal(src, end), 0)))
            src = src[:,:len_src]
        with tf.variable_scope('mask'):
            mask = tf.to_float(tf.expand_dims(tf.not_equal(src, end), 1))
        return Transformer(
            position= Sinusoid(dim, src_cap)
            , src_= src_, src= src, mask= mask
            , tgt_= tgt_
            , **self)

    def build(self, trainable= True):
        """-> Transformer with new fields

          output : f32 (b, t, dim_tgt) prediction on logit scale
            prob : f32 (b, t, dim_tgt) prediction, soft
            pred : i32 (b, t)          prediction, hard
            loss : f32 ()              prediction loss
             acc : f32 ()              accuracy

        must be called after `data`.

        """
        logit, dropout = self.logit, self.dropout if trainable else identity
        src, mask, position = self.src, self.mask, self.position
        encode, emb_src = self.encode, self.emb_src
        decode, emb_pos = self.decode, self.emb_pos
        bridge = self.bridge
        # todo mask current step
        with tf.variable_scope('emb_src'): w = position(tf.shape(src)[1]) + dropout(emb_src.embed(src))
        with tf.variable_scope('emb_pos'): x = tf.tile(tf.expand_dims(emb_pos.kern, 0), (tf.shape(src)[0], 1, 1))
        with tf.variable_scope('encode'):
            for enc in encode: w = enc(w, mask, dropout)
        x = bridge(x, None, dropout, w)
        with tf.variable_scope('decode'):
            for dec in decode: x = dec(x, None, dropout)
        y = logit(x)
        p = tf.argmax(y, -1, output_type= tf.int32, name= 'pred')
        return Transformer(output= y, pred= p, **self)._eval()

    def _eval(self):
        gold, pred, output, smooth = self.tgt_, self.pred, self.output, self.smooth
        with tf.variable_scope('acc'):
            acc = tf.reduce_mean(tf.to_float(tf.equal(gold, pred)))
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= output, labels= smooth(gold)))
        with tf.variable_scope('prob'):
            prob = tf.nn.softmax(output, name= 'prob')
        return Transformer(prob= prob, loss= loss, acc= acc, **self)

    def train(self, warmup= 4e3, beta1= 0.9, beta2= 0.98, epsilon= 1e-9):
        """-> Transformer with new fields

        step : i64 () global update step
          lr : f32 () learning rate for the current step
          up :        update operation

        """
        dim, loss = self.dim, self.loss
        with tf.variable_scope('lr'):
            s = tf.train.get_or_create_global_step()
            t = tf.to_float(s + 1)
            lr = (dim ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(loss, s)
        return Transformer(step= s, lr= lr, up= up, **self)
