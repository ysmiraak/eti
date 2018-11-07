from util import Record, identity
from util_np import np, partition, encode, decode
from util_tf import QueryAttention as Attention
from util_tf import tf, placeholder, Normalize, Smooth, Dropout, Linear, Affine, Multilayer


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
                1 + tf.range(tf.to_float(time), dtype= tf.float32)
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


class EncodeBlock(Record):

    def __init__(self, dim, dim_mid, act, name):
        self.name = name
        with tf.variable_scope(name):
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


class DecodeBlock(Record):

    def __init__(self, dim, dim_mid, act, name):
        self.name = name
        with tf.variable_scope(name):
            with tf.variable_scope('csl'):
                self.csl = Attention(dim, layer= Multilayer, mid= dim_mid, act= act)
                self.norm_csl = Normalize(dim)
            with tf.variable_scope('att'):
                self.att = Attention(dim, layer= Multilayer, mid= dim_mid, act= act)
                self.norm_att = Normalize(dim)
            with tf.variable_scope('mlp'):
                self.mlp = Multilayer(dim, dim, dim_mid, act)
                self.norm_mlp = Normalize(dim)

    def __call__(self, x, v, w, m, dropout, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            with tf.variable_scope('csl'): x = self.norm_csl(x + dropout(self.csl(x, v, mask)))
            with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, w, m)))
            with tf.variable_scope('mlp'): x = self.norm_mlp(x + dropout(self.mlp(x)))
            return x


class Transformer(Record):
    """-> Record

    model = Transformer.new( ... )
    train = model.data( ... ).build( ... ).train( ... )
    valid = model.data( ... ).build(trainable= False)
    infer = model.data( ... ).infer( ... )

    """
    _new   = 'dim', 'dim_mid', 'depth', 'dim_src', 'dim_tgt'
    _data  = 'cap',
    _infer = 'cap',
    _build = 'dropout', 'smooth'
    _train = 'warmup', 'beta1', 'beta2', 'epsilon'

    @staticmethod
    def new(dim, dim_mid, depth, dim_src, dim_tgt, act= tf.nn.relu, end= 1, begin= 2):
        """-> Transformer with fields

         emb_src : Linear
         emb_tgt : Linear
          encode : tuple EncodeBlock
          decode : tuple EncodeBlock
           logit : Affine

        """
        assert not dim % 2
        with tf.variable_scope('encode'):
            encode = tuple(EncodeBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(depth))
        with tf.variable_scope('decode'):
            decode = tuple(DecodeBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(depth))
        return Transformer(
            logit= Affine(dim_tgt, dim, 'logit')
            , emb_src= Linear(dim, dim_src, 'emb_src')
            , emb_tgt= Linear(dim, dim_tgt, 'emb_tgt')
            , encode= encode
            , decode= decode
            , end= tf.constant(end, tf.int32, (), 'end')
            , begin= tf.constant(begin, tf.int32, (), 'begin'))

    def data(self, src= None, tgt= None, cap= None):
        """-> Transformer with new fields

            src_ : i32 (b, ?)    source feed, in range `[0, dim_src)`
            tgt_ : i32 (b, ?)    target feed, in range `[0, dim_tgt)`
             src : i32 (b, s)    source with `end` trimmed among the batch
             tgt : i32 (b, t)    target with `end` trimmed among the batch, padded with `begin`
            init : i32 (b, 1)    target initial step
            gold : i32 (b, t)    target one step ahead
            mask : f32 (b, 1, s) bridge mask
        mask_src : f32 (b, s, s) source mask
        position : Sinusoid

        setting `cap` makes it more efficient for training.  you won't
        be able to feed it longer sequences, but it doesn't affect any
        model parameters.

        """
        with tf.variable_scope('src'):
            src_ = placeholder(tf.int32, (None, None), src)
            not_end = tf.to_float(tf.not_equal(src_, self.end))
            len_src = tf.reduce_sum(tf.to_int32(0 < tf.reduce_sum(not_end, 0)))
            not_end = tf.expand_dims(not_end[:,:len_src], 1)
            mask_src = tf.log(not_end + tf.expand_dims(1 - tf.eye(len_src), 0))
            mask = tf.log(not_end)
            src = src_[:,:len_src]
            init = self.begin + tf.zeros_like(src_[:,:1])
        with tf.variable_scope('tgt'):
            tgt_ = placeholder(tf.int32, (None, None), tgt)
            not_end = tf.to_float(tf.not_equal(tgt_, self.end))
            len_tgt = tf.reduce_sum(tf.to_int32(0 < tf.reduce_sum(not_end, 0)))
            gold = tgt_[:,:len_tgt] # maybe add end padding here as well
            tgt = tf.concat((init, gold[:,:-1]), 1)
        return Transformer(
            position= Sinusoid(int(self.logit.kern.shape[0]), cap)
            , src= src, src_= src_
            , tgt= tgt, tgt_= tgt_
            , init= init
            , gold= gold
            , mask= mask
            , mask_src= mask_src
            , **self)

    def infer(self, cap= 256, random= False, minimal= True):
        """-> Transformer with new fields, autoregressive

        len_tgt : i32 ()              steps to unfold aka t
         output : f32 (b, t, dim_tgt) prediction on logit scale
           prob : f32 (b, t, dim_tgt) prediction, soft
           pred : i32 (b, t)          prediction, hard

        """
        dim, dim_tgt = map(int, self.logit.kern.shape)
        dropout = identity
        with tf.variable_scope('emb_src_infer'):
            w = self.position(tf.shape(self.src)[1]) + dropout(self.emb_src.embed(self.src))
        with tf.variable_scope('encode_infer'):
            for enc in self.encode: w = enc(w, self.mask_src, dropout)
        with tf.variable_scope('decode_infer'):
            with tf.variable_scope('init'):
                t = placeholder(tf.int32, (), cap, name= 'cap')
                pos = self.position(t)
                i = tf.constant(0)
                x = self.init
                v = w[:,:0]
                y = tf.reshape(v, (tf.shape(v)[0], 0, dim_tgt))
                p = x[:,1:]
            def autoreg(i, x, vs, y, p):
                # i : ()              time step from 0 to t
                # x : (b, 1)          x_i
                # v : (b, t, dim)     attention values
                # y : (b, t, dim_tgt) logit over x one step ahead
                # p : (b, t)          predictions
                with tf.variable_scope('emb_tgt'): x = pos[i] + dropout(self.emb_tgt.embed(x))
                us = []
                for dec, v in zip(self.decode, vs):
                    with tf.variable_scope('cache_v'):
                        v = tf.concat((v, x), 1)
                        us.append(v)
                    x = dec(x, v, w, self.mask, dropout)
                x = self.logit(x)
                with tf.variable_scope('cache_y'): y = tf.concat((y, x), 1)
                if random:
                    with tf.variable_scope('sample'):
                        x = tf.multinomial(tf.squeeze(x, 1), 1, output_dtype= tf.int32)
                else:
                    x = tf.argmax(x, -1, output_type= tf.int32, name= 'argmax')
                with tf.variable_scope('cache_p'): p = tf.concat((p, x), 1)
                return i + 1, x, tuple(us), y, p
            _, _, _, y, p = tf.while_loop(
                lambda i, x, *_: ((i < t) & ~ tf.reduce_all(tf.equal(x, self.end))) if minimal else (i < t)
                , autoreg
                , (i, x, (v,)*len(self.decode), y, p)
                , (i.shape, x.shape, (v.shape,)*len(self.decode), tf.TensorShape((None, None, dim_tgt)), p.shape)
                , back_prop= False
                , swap_memory= True
                , name= 'autoreg')
        with tf.variable_scope('prob_infer'): prob = tf.nn.softmax(y)
        return Transformer(len_tgt= t, output= y, pred= p, prob= prob, **self)

    def build(self, trainable= True, dropout= 0.1, smooth= 0.1):
        """-> Transformer with new fields, teacher forcing

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
        with tf.variable_scope('emb_src_'):
            w = self.position(tf.shape(self.src)[1]) + dropout(self.emb_src.embed(self.src))
        with tf.variable_scope('emb_tgt_'):
            x = self.position(tf.shape(self.tgt)[1]) + dropout(self.emb_tgt.embed(self.tgt))
        with tf.variable_scope('encode_'):
            for enc in self.encode: w = enc(w, self.mask_src, dropout)
        with tf.variable_scope('decode_'):
            with tf.variable_scope('mask'):
                len_tgt = tf.shape(x)[1]
                causal_mask = tf.log(
                    tf.linalg.LinearOperatorLowerTriangular(tf.ones((len_tgt, len_tgt))).to_dense()
                    # - tf.eye(len_tgt) # cannot mask current step lest the first step becomes empty
                )
            for dec in self.decode: x = dec(x, x, w, self.mask, dropout, causal_mask)
        y = self.logit(x, 'logit_')
        with tf.variable_scope('prob_'): prob = tf.nn.softmax(y)
        with tf.variable_scope('pred_'): pred = tf.argmax(y, -1, output_type= tf.int32)
        with tf.variable_scope('acc_'): acc = tf.reduce_mean(tf.to_float(tf.equal(self.gold, pred)))
        with tf.variable_scope('loss_'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels= smooth(self.gold), logits= y)
                if trainable else
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels= self.gold, logits= y))
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


def batch_run(sess, model, fetches, src, tgt= None, batch= None):
    if batch is None: batch = len(src)
    for i, j in partition(len(src), batch, discard= False):
        feed = {model.src_: src[i:j]}
        if tgt is not None:
            feed[model.tgt_] = tgt[i:j]
        yield sess.run(fetches, feed)


def translate(sess, sents, index, model, dtype= np.uint8, batch= None):
    if not isinstance(sents, np.ndarray):
        sents = encode(index, sents, dtype= dtype)
    for preds in batch_run(sess, model, model.pred, src= sents, batch= batch):
        yield from decode(index, preds)
