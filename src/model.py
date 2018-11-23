from util import Record, identity
from util_np import np, partition
from util_tf import TransformerAttention as Attention
from util_tf import tf, placeholder, Normalize, Smooth, Dropout, Linear, Multilayer, Conv, Affine


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
            with tf.variable_scope('csl'):
                self.csl = Attention(dim, layer= Multilayer, mid= dim_mid, act= act)
                self.norm_csl = Normalize(dim)
            with tf.variable_scope('att'):
                self.att = Attention(dim, layer= Multilayer, mid= dim_mid, act= act)
                self.norm_att = Normalize(dim)
            with tf.variable_scope('mlp'):
                self.mlp = Multilayer(dim, dim, dim_mid, act)
                self.norm_mlp = Normalize(dim)

    def __call__(self, x, v, m, w, n, dropout, name= None):
        with tf.variable_scope(name or self.name):
            with tf.variable_scope('csl'): x = self.norm_csl(x + dropout(self.csl(x, v, m)))
            with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, w, n)))
            with tf.variable_scope('mlp'): x = self.norm_mlp(x + dropout(self.mlp(x)))
            return x


class ConvBlock(Record):

    def __init__(self, dim, dim_mid, depth, act, name):
        self.name = name
        with tf.variable_scope(name):
            self.ante = Affine(dim_mid, dim, 'ante')
            self.conv = tuple(Conv(dim, name= "conv{}".format(1+i)) for i in range(depth))
            self.post = Affine(dim, dim_mid, 'post')
            self.norm = Normalize(dim, name= 'norm')

    def __call__(self, x, dropout, name= None):
        with tf.variable_scope(name or self.name):
            y = self.ante(x) # act here ????
            for conv in self.conv:
                y = self.act(self.conv(y))
            return self.norm(x + dropout(self.post(y)))


class Transformer(Record):
    """-> Record

    model = Transformer.new( ... )
    train = model.data( ... ).build( ... ).train( ... )
    valid = model.data( ... ).build(trainable= False)
    infer = model.data( ... ).infer( ... )

    """
    _new   = 'dim_emb', 'dim_mid', 'depth', 'dim_src', 'dim_tgt', 'cap', 'eos', 'bos'
    _build = 'dropout', 'smooth'
    _train = 'warmup', 'beta1', 'beta2', 'epsilon'

    @staticmethod
    def new(dim_emb, dim_mid, depth, dim_src, dim_tgt, cap, eos, bos, act= tf.nn.relu):
        """-> Transformer with fields

           logit : Linear
          decode : tuple DecodeBlock
        enc_satt : EncodeBlock
        enc_conv : tuple ConvBlock
         emb_tgt : Linear
         emb_src : Linear

        """
        assert not dim_emb % 2
        with tf.variable_scope('encode'):
            enc_conv = tuple(ConvBlock(dim_emb, 128, 2, act, "layer{}".format(1+i)) for i in range(6))
            enc_satt = EncodeBlock(dim_emb, dim_mid, act, "satt")
        with tf.variable_scope('decode'):
            decode = tuple(DecodeBlock(dim_emb, dim_mid, act, "layer{}".format(1+i)) for i in range(depth))
        return Transformer(
            logit= Linear(dim_tgt, dim_emb, 'logit')
            , decode= decode
            , enc_satt= enc_satt
            , enc_conv= enc_conv
            , emb_tgt= Linear(dim_emb, dim_tgt, 'emb_tgt')
            , emb_src= Linear(dim_emb, dim_src, 'emb_src')
            , dim_emb= dim_emb
            , dim_tgt= dim_tgt
            , bos= bos
            , eos= eos
            , cap= cap + 1)

    def data(self, src= None, tgt= None):
        """-> Transformer with new fields

        position : Sinusoid
            src_ : i32 (b, ?)    source feed, in range `[0, dim_src)`
            tgt_ : i32 (b, ?)    target feed, in range `[0, dim_tgt)`
             src : i32 (b, s)    source with `eos` trimmed among the batch
             tgt : i32 (b, t)    target with `eos` trimmed among the batch, padded with `bos`
            gold : i32 (b, t)    target one step ahead, padded with `eos`
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
        with tf.variable_scope('tgt'):
            with tf.variable_scope('not_eos'):
                not_eos = tf.to_float(tf.not_equal(tgt_, self.eos))
            with tf.variable_scope('len_tgt'):
                len_tgt = tf.reduce_sum(tf.to_int32(0 < tf.reduce_sum(not_eos, 0)))
            tgt = tgt_[:,:len_tgt]
            gold = tf.pad(tgt, ((0,0),(0,1)), constant_values= self.eos)
            tgt  = tf.pad(tgt, ((0,0),(1,0)), constant_values= self.bos)
        return Transformer(
            position= Sinusoid(self.dim_emb, self.cap)
            , src_= src_, src= src
            , tgt_= tgt_, tgt= tgt
            , gold= gold
            , mask= mask
            , mask_src= mask_src
            , **self)

    def infer(self, random= False, minimal= True):
        """-> Transformer with new fields, autoregressive

        len_tgt : i32 ()              steps to unfold aka t
         output : f32 (b, t, dim_tgt) prediction on logit scale
           prob : f32 (b, t, dim_tgt) prediction, soft
           pred : i32 (b, t)          prediction, hard

        """
        dropout = identity
        with tf.variable_scope('emb_src_infer'):
            w = self.position(tf.shape(self.src)[1]) + dropout(self.emb_src.embed(self.src))
        with tf.variable_scope('encode_infer'):
            for conv in self.enc_conv: w = conv(w, dropout)
            w = self.enc_satt(w, w, self.mask_src, dropout)
        with tf.variable_scope('decode_infer'):
            with tf.variable_scope('init'):
                b,t = tf.shape(w)[0], placeholder(tf.int32, (), self.cap, 't')
                pos = self.position(t)
                i = tf.constant(0)
                x = tf.fill((b, 1), self.bos, 'x')
                v, v_shape = tf.zeros((b, 1, self.dim_emb), name= 'v'), tf.TensorShape((None, None, self.dim_emb))
                y, y_shape = tf.zeros((b, 0, self.dim_tgt), name= 'y'), tf.TensorShape((None, None, self.dim_tgt))
                p, p_shape = tf.zeros((b, 0),     tf.int32, name= 'p'), tf.TensorShape((None, None))
            def body(i, x, vs, y, p):
                # i : ()                time step from 0 to t
                # x : (b, 1)            x_i
                # v : (b, 1+i, dim_emb) attention values
                # y : (b, i,   dim_tgt) logit over x one step ahead
                # p : (b, i)            predictions
                with tf.variable_scope('emb_tgt'): x = pos[i] + dropout(self.emb_tgt.embed(x))
                us = []
                for v, dec in zip(vs, self.decode):
                    with tf.variable_scope('cache_v'): us.append(tf.concat((v, x), 1))
                    x = dec(x, v, None, w, self.mask, dropout)
                x = self.logit(x)
                with tf.variable_scope('cache_y'): y = tf.concat((y, x), 1)
                if random:
                    with tf.variable_scope('sample'):
                        x = tf.multinomial(tf.squeeze(x, 1), 1, output_dtype= tf.int32)
                else:
                    with tf.variable_scope('argmax'):
                        x = tf.argmax(x, -1, output_type= tf.int32)
                with tf.variable_scope('cache_p'): p = tf.concat((p, x), 1)
                return i + 1, x, tuple(us), y, p
            def cond(i, x, *_):
                with tf.variable_scope('cond'):
                    return ((i < t) & ~ tf.reduce_all(tf.equal(x, self.eos))) if minimal else (i < t)
            _, _, _, y, p = tf.while_loop(
                cond, body
                , (i      , x      , (v      ,)*len(self.decode), y      , p)
                , (i.shape, x.shape, (v_shape,)*len(self.decode), y_shape, p_shape)
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
            dropout = Dropout(dropout, (None, None, self.dim_emb))
            smooth  = Smooth(smooth, self.dim_tgt)
        else:
            dropout = smooth = identity
        with tf.variable_scope('emb_src_'):
            w = self.position(tf.shape(self.src)[1]) + dropout(self.emb_src.embed(self.src))
        with tf.variable_scope('emb_tgt_'):
            x = self.position(tf.shape(self.tgt)[1]) + dropout(self.emb_tgt.embed(self.tgt))
        with tf.variable_scope('encode_'):
            for conv in self.enc_conv: w = conv(w, dropout)
            w = self.enc_satt(w, w, self.mask_src, dropout)
        with tf.variable_scope('decode_'):
            with tf.variable_scope('mask'):
                t = tf.shape(x)[1]
                m = tf.log(tf.linalg.LinearOperatorLowerTriangular(tf.ones((t, t))).to_dense())
            for i, dec in enumerate(self.decode):
                with tf.variable_scope("pad{}".format(1+i)):
                    v = tf.pad(x[:,:-1], ((0,0),(1,0),(0,0)))
                x = dec(x, v, m, w, self.mask, dropout)
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
        with tf.variable_scope('lr'):
            s = tf.train.get_or_create_global_step()
            t = tf.to_float(s + 1)
            lr = (self.dim_emb ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(self.loss, s)
        return Transformer(step= s, lr= lr, up= up, **self)


def batch_run(sess, model, fetch, src, tgt= None, batch= None):
    if batch is None: batch = len(src)
    for i, j in partition(len(src), batch, discard= False):
        feed = {model.src_: src[i:j]}
        if tgt is not None:
            feed[model.tgt_] = tgt[i:j]
        yield sess.run(fetch, feed)
