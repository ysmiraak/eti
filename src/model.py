from util import Record, identity
from util_np import np, partition
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

    def __call__(self, x, v, m, dropout, name= None):
        with tf.variable_scope(name or self.name):
            with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, v, m)))
            with tf.variable_scope('mlp'): x = self.norm_mlp(x + dropout(self.mlp(x)))
            return x


class DecodeBlock(Record):

    def __init__(self, dim, dim_mid, act, name):
        self.name = name
        with tf.variable_scope(name):
            self.internal = EncodeBlock(dim, dim_mid, act, 'internal')
            self.external = EncodeBlock(dim, dim_mid, act, 'external')

    def __call__(self, x, v, m, w, n, dropout, name= None):
        with tf.variable_scope(name or self.name):
            x = self.internal(x, v, m, dropout)
            x = self.external(x, w, n, dropout)
            return x


class Transformer(Record):
    """-> Record

    model = Transformer.new( ... )
    train = model.data( ... ).build( ... ).train( ... )
    valid = model.data( ... ).build(trainable= False)

    """
    _new   = 'dim_emb', 'dim_mid', 'depth', 'dim_src', 'dim_tgt', 'cap', 'eos'
    _build = 'dropout', 'smooth'
    _train = 'warmup', 'beta1', 'beta2', 'epsilon'

    @staticmethod
    def new(dim_emb, dim_mid, depth, dim_src, dim_tgt, cap, eos, act= tf.nn.relu):
        """-> Transformer with fields

           logit : Affine
          decode : tuple EncodeBlock
          encode : tuple DecodeBlock
        mask_tgt : f32 (1, t, t)
         emb_pos : Linear
         emb_src : Linear

        """
        assert not dim_emb % 2
        with tf.variable_scope('encode'):
            encode = tuple(EncodeBlock(dim_emb, dim_mid, act, "layer{}".format(1+i)) for i in range(depth))
        with tf.variable_scope('decode'):
            ante   = EncodeBlock(dim_emb, dim_mid, act, 'ante')
            decode = DecodeBlock(dim_emb, dim_mid, act, 'decode')
            post   = EncodeBlock(dim_emb, dim_mid, act, 'post')
        with tf.variable_scope('mask_tgt'):
            mask_tgt = tf.log(1 - tf.eye(cap))
        logit = Linear(dim_tgt, dim_emb, 'logit')
        return Transformer(
            logit= logit, step= tf.train.get_or_create_global_step()
            , ante= ante
            , post= post
            , decode= decode
            , encode= encode
            , mask_tgt= mask_tgt
            , emb_pos= Linear(dim_emb, cap, 'emb_pos')
            , emb_tgt= logit.transpose('emb_tgt')
            , emb_src= Linear(dim_emb, dim_src, 'emb_src')
            , dim_emb= dim_emb
            , dim_tgt= dim_tgt
            , eos= eos
            , cap= cap)

    def data(self, src= None, tgt= None):
        """-> Transformer with new fields

        position : Sinusoid
            src_ : i32 (b, ?)    source feed, in range `[0, dim_src)`
            tgt_ : i32 (b, ?)    target feed, in range `[0, dim_tgt)`
             src : i32 (b, s)    source with `eos` trimmed among the batch
            gold : i32 (b, t)    target
            mask : f32 (b, 1, s) bridge mask
        mask_src : f32 (b, s, s) source mask

        """
        src_ = placeholder(tf.int32, (None, None), src, 'src_')
        tgt_ = placeholder(tf.int32, (None, None), tgt, 'tgt_')
        with tf.variable_scope('src'):
            with tf.variable_scope('not_eos'):
                not_eos = tf.to_float(tf.not_equal(src_, self.eos))
            with tf.variable_scope('len_src'):
                len_src = tf.reduce_sum(tf.to_int32(0 < tf.reduce_sum(not_eos, 0)))
            not_eos = tf.expand_dims(not_eos[:,:len_src], 1)
            src = src_[:,:len_src]
        with tf.variable_scope('mask_src'):
            mask_src = tf.log(not_eos + (1 - tf.eye(len_src)))
        with tf.variable_scope('mask'):
            mask = tf.log(not_eos)
        return Transformer(
            position= Sinusoid(self.dim_emb, self.cap)
            , src_= src_, src= src
            , tgt_= tgt_
            , gold= tgt_
            , mask= mask
            , mask_src= mask_src
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
            dropout = Dropout(dropout, (None, None, self.dim_emb))
            smooth  = Smooth(smooth, self.dim_tgt)
        else:
            dropout = smooth = identity
        with tf.variable_scope('emb_src_'):
            shape = tf.shape(self.src)
            w = self.position(shape[1]) + dropout(self.emb_src.embed(self.src))
        with tf.variable_scope('encode_'):
            for enc in self.encode: w = enc(w, w, self.mask_src, dropout)
        with tf.variable_scope('emb_pos_'):
            x = dropout(tf.tile(tf.expand_dims(self.emb_pos.kern, 0), (shape[0], 1, 1)))

        with tf.variable_scope('emb_tgt_'):
            x += self.emb_tgt.embed(self.gold) \
                * tf.to_float(
                    tf.nn.sigmoid(tf.to_float(self.step) / 1e5)
                    < tf.random_uniform((shape[0], self.cap, self.dim_emb)))

        with tf.variable_scope('decode_'):
            x = self.ante(x, w, self.mask, dropout)
            x = self.decode(x, x, self.mask_tgt, w, self.mask, dropout)
            x = self.post(x, x, self.mask_tgt, dropout)
        y = self.logit(x, 'logit_')
        with tf.variable_scope('prob_'): prob = tf.nn.softmax(y)
        with tf.variable_scope('pred_'): pred = tf.argmax(y, -1, output_type= tf.int32)
        with tf.variable_scope('acc_'): acc = tf.reduce_mean(tf.to_float(tf.equal(self.gold, pred)))
        with tf.variable_scope('loss_'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels= smooth(self.gold), logits= y)
                # * (tf.range(tf.to_float(self.cap), dtype= tf.float32)
                #    * tf.to_float(tf.not_equal(self.gold, self.eos))
                #    + 1.0)
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
        s = self.step
        with tf.variable_scope('lr'):
            t = tf.to_float(s + 1)
            lr = (self.dim_emb ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(self.loss, s)
        return Transformer(lr= lr, up= up, **self)


def batch_run(sess, model, fetch, src, tgt= None, batch= None):
    if batch is None: batch = len(src)
    for i, j in partition(len(src), batch, discard= False):
        feed = {model.src_: src[i:j]}
        if tgt is not None:
            feed[model.tgt_] = tgt[i:j]
        yield sess.run(fetch, feed)
