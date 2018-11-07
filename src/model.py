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


class EncodeBlock(Record):

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
    train = model.data( ... ).build( ... ).train( ... )
    valid = model.data( ... ).build(trainable= False)

    """
    _new   = 'dim', 'dim_mid', 'depth', 'dim_src', 'dim_tgt'
    _data  = 'cap',
    _build = 'dropout', 'smooth'
    _train = 'warmup', 'beta1', 'beta2', 'epsilon'

    @staticmethod
    def new(dim, dim_mid, depth, dim_src, dim_tgt, act= tf.nn.relu, end= 1):
        """-> Transformer with fields

         emb_src : Linear
          encode : tuple EncodeBlock
           logit : Affine

        """
        assert not dim % 2
        with tf.variable_scope('encode'):
            encode = tuple(EncodeBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(depth))
        return Transformer(
            logit= Affine(dim_tgt, dim, 'logit')
            , emb_src= Linear(dim, dim_src, 'emb_src')
            , encode= encode
            , end= tf.constant(end, tf.int32, (), 'end'))

    def data(self, src= None, tgt= None, cap= None):
        """-> Transformer with new fields

            src_ : i32 (b, ?)    source feed, in range `[0, dim_src)`
            tgt_ : i32 (b, ?)    target feed, in range `[0, dim_tgt)`
             src : i32 (b, s)    source with `end` trimmed among the batch
            gold : i32 (b, t)    target one step ahead
            mask : f32 (1, s, s) source mask
        position : Sinusoid

        setting `cap` makes it more efficient for training.  you won't
        be able to feed it longer sequences, but it doesn't affect any
        model parameters.

        """
        with tf.variable_scope('src'):
            src_ = placeholder(tf.int32, (None, None), src)
            mask = tf.log(tf.expand_dims(1 - tf.eye(cap), 0))
        with tf.variable_scope('tgt'):
            tgt_ = placeholder(tf.int32, (None, None), tgt)
        return Transformer(
            position= Sinusoid(int(self.logit.kern.shape[0]), cap)
            , src_= src_, src= src_
            , tgt_= tgt_
            , gold= tgt_
            , mask= mask
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
        with tf.variable_scope('emb_src_'):
            shape = tf.shape(self.src)
            w = self.position(shape[1]) + dropout(self.emb_src.embed(self.src))
        with tf.variable_scope('encode_'):
            for enc in self.encode: w = enc(w, self.mask, dropout)
        y = self.logit(w, 'logit_')
        with tf.variable_scope('prob_'): prob = tf.nn.softmax(y)
        with tf.variable_scope('pred_'): pred = tf.argmax(y, -1, output_type= tf.int32)
        with tf.variable_scope('acc_'): acc = tf.reduce_mean(tf.to_float(tf.equal(self.gold, pred)))
        with tf.variable_scope('loss_'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels= smooth(self.gold), logits= y)
                * (1 + tf.range(cap, dtype= tf.float32) * tf.to_float(tf.not_equal(self.gold, self.end)))
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
