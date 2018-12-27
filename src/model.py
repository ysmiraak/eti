from util import Record, identity
from util_np import np, partition
from util_tf import tf, scope, placeholder, Normalize, Smooth, Dropout, Embed, Conv, SepConv, Attention


def sinusoid(dim, time, freq= 1e-4, array= False):
    """returns a rank-2 tensor of shape `dim, time`, where each column
    corresponds to a time step and each row a sinusoid, with
    frequencies in a geometric progression from 1 to `freq`.

    """
    assert not dim % 2
    if array:
        a = (freq ** ((2 / dim) * np.arange(dim // 2))).reshape(-1, 1) @ (1 + np.arange(time).reshape(1, -1))
        return np.concatenate((np.sin(a), np.cos(a)), -1).reshape(dim, time)
    else:
        assert False # figure out a better way to do this
        a = tf.reshape(
            freq ** ((2 / dim) * tf.range(dim // 2, dtype= tf.float32))
            , (-1, 1)) @ tf.reshape(
                1 + tf.range(tf.to_float(time), dtype= tf.float32)
                , (1, -1))
        return tf.reshape(tf.concat((tf.sin(a), tf.cos(a)), axis= -1), (dim, time))


class Sinusoid(Record):

    def __init__(self, dim, cap= None, name= 'sinusoid'):
        self.dim = dim
        self.name = name
        with scope(name):
            self.pos = tf.constant(sinusoid(dim, cap, array= True), tf.float32) if cap else None

    def __call__(self, time, name= None):
        with scope(name or self.name):
            return sinusoid(self.dim, time) if self.pos is None else self.pos[:,:time]


class MlpBlock(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.lin  = Conv(4*dim, dim, name= 'lin')
            self.lex  = Conv(dim, 4*dim, name= 'lex')
            self.norm = Normalize(dim)

    def __call__(self, x, dropout, name= None):
        with scope(name or self.name):
            return self.norm(x + dropout(self.lex(tf.nn.relu(self.lin(x)))))


class AttBlock(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.att  = Attention(dim)
            self.norm = Normalize(dim)

    def __call__(self, x, v, m, dropout, name= None):
        with scope(name or self.name):
            return self.norm(x + dropout(self.att(x, v, m)))


class BiattBlock(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.latt = Attention(dim, name= 'latt')
            self.ratt = Attention(dim, name= 'ratt')
            self.norm = Normalize(dim)

    def __call__(self, x, v, m, w, n, dropout, name= None):
        with scope(name or self.name):
            return self.norm(tf.add_n(((dropout(self.latt(x, v, m)), x, dropout(self.ratt(x, w, n))))))


class GluBlock(Record):

    def __init__(self, dim, name, mid= 128, depth= 2):
        self.name = name
        with scope(name):
            self.ante =       Conv(mid, dim, size= 1, name= 'ante')
            self.gate = tuple(Conv(mid, mid, size= 2, name= "gate{}".format(1+i)) for i in range(depth))
            self.conv = tuple(Conv(mid, mid, size= 2, name= "conv{}".format(1+i)) for i in range(depth))
            self.post =       Conv(dim, mid, size= 1, name= 'post')
            self.norm = Normalize(dim, name= 'norm')

    def __call__(self, x, dropout, name= None):
        with scope(name or self.name):
            y = self.ante(x)
            for gate, conv in zip(self.gate, self.conv):
                y = tf.pad(y, ((0,0),(0,0),(conv.shape()[0]-1,0)))
                y = tf.sigmoid(gate(y)) * conv(y)
            return self.norm(x + dropout(self.post(y)))


class SepBlock(Record):

    def __init__(self, dim, name, size= 5, depth= 2):
        self.name = name
        with scope(name):
            self.ante =          Conv(dim, dim, size=    1, name= 'ante')
            self.gate = tuple(SepConv(dim, dim, size= size, name= "gate{}".format(1+i)) for i in range(depth))
            self.conv = tuple(SepConv(dim, dim, size= size, name= "conv{}".format(1+i)) for i in range(depth))
            self.post =          Conv(dim, dim, size=    1, name= 'post')
            self.norm = Normalize(dim, name= 'norm')

    def __call__(self, x, dropout, name= None):
        with scope(name or self.name):
            y = self.ante(x)
            for gate, conv in zip(self.gate, self.conv):
                y = tf.pad(y, ((0,0),(0,0),(conv.shape()[0]-1,0)))
                y = tf.sigmoid(gate(y)) * conv(y)
            return self.norm(x + dropout(self.post(y)))


class Encode(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.blocks = AttBlock(dim, 's1') \
                ,         MlpBlock(dim, 'm1') \
                ,         AttBlock(dim, 's2') \
                ,         MlpBlock(dim, 'm2') \
                ,         AttBlock(dim, 's3') \
                ,         MlpBlock(dim, 'm3') \
                # ,         AttBlock(dim, 's4') \
                # ,         MlpBlock(dim, 'm4') \
                # ,         AttBlock(dim, 's5') \
                # ,         MlpBlock(dim, 'm5') \
                # ,         AttBlock(dim, 's6') \
                # ,         MlpBlock(dim, 'm6')

    def __call__(self, x, m, dropout, name= None):
        with scope(name or self.name):
            for block in self.blocks:
                btype = block.name[0]
                if   'c' == btype: x = block(x, dropout)
                elif 's' == btype: x = block(x, x, m, dropout)
                elif 'm' == btype: x = block(x, dropout)
                else: raise TypeError('unknown encode block')
            return x


class Decode(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.blocks = AttBlock(dim, 's1') \
                ,         AttBlock(dim, 'a1') \
                ,         MlpBlock(dim, 'm1') \
                ,         AttBlock(dim, 's2') \
                ,         AttBlock(dim, 'a2') \
                ,         MlpBlock(dim, 'm2') \
                ,         AttBlock(dim, 's3') \
                ,         AttBlock(dim, 'a3') \
                ,         MlpBlock(dim, 'm3') \
                # ,         AttBlock(dim, 's4') \
                # ,         AttBlock(dim, 'a4') \
                # ,         MlpBlock(dim, 'm4') \
                # ,         AttBlock(dim, 's5') \
                # ,         AttBlock(dim, 'a5') \
                # ,         MlpBlock(dim, 'm5') \
                # ,         AttBlock(dim, 's6') \
                # ,         AttBlock(dim, 'a6') \
                # ,         MlpBlock(dim, 'm6')

    def __call__(self, x, m, w, n, dropout, name= None):
        with scope(name or self.name):
            for block in self.blocks:
                btype = block.name[0]
                if   'c' == btype: x = block(x, dropout)
                elif 'b' == btype: x = block(x, x, m, w, n, dropout)
                elif 's' == btype: x = block(x, x, m, dropout)
                elif 'a' == btype: x = block(x, w, n, dropout)
                elif 'm' == btype: x = block(x, dropout)
                else: raise TypeError('unknown decode block')
            return x

    def cache_init(self, b, d):
        sn2cs, cs, cs_shape, j = {}, [], [], 0
        for block in self.blocks:
            btype = block.name[0]
            if 'c' == btype:
                for conv in block.conv:
                    s, n, _ = conv.shape()
                    try:
                        c, c_shape = sn2cs[(s, n)]
                    except KeyError:
                        c, c_shape = tf.zeros((b, n, s-1), name= 'c'), tf.TensorShape((None, n, s-1))
                        sn2cs[(s, d)] = c, c_shape
                    cs.append(c)
                    cs_shape.append(c_shape)
            elif 'b' == btype: j += 1
            elif 's' == btype: j += 1
            else: pass
        return tuple(cs), tuple(cs_shape), (tf.zeros((b, d, 0), name= 'v'),)*j, (tf.TensorShape((None, d, None)),)*j

    def autoreg(self, x, cs, vs, w, n, dropout, name= None):
        with scope(name or self.name):
            i, ds, j, us = 0, [], 0, []
            for block in self.blocks:
                btype = block.name[0]
                if 'c' == btype:
                    with scope(block.name):
                        d = block.ante(x)
                        if hasattr(block, 'gate'):
                            for gate, conv in zip(block.gate, block.conv):
                                c, i = cs[i], i + 1
                                d = tf.concat((c, d), axis= -1, name= 'cache_c')
                                ds.append(d[:,:,1:])
                                d = tf.sigmoid(gate(d)) * conv(d)
                        else:
                            for conv in block.conv:
                                c, i = cs[i], i + 1
                                d = tf.concat((c, d), axis= -1, name= 'cache_c')
                                ds.append(d[:,:,1:])
                                d = tf.nn.relu(conv(d))
                        x = block.norm(x + dropout(block.post(d)))
                elif 'b' == btype:
                    v, j = vs[j], j + 1
                    us.append(tf.concat((v, x), axis= -1, name= 'cache_v'))
                    x = block(x, v, None, w, n, dropout)
                elif 's' == btype:
                    v, j = vs[j], j + 1
                    v = tf.concat((v, x), axis= -1, name= 'cache_v')
                    us.append(v)
                    x = block(x, v, None, dropout)
                elif 'a' == btype: x = block(x, w, n, dropout)
                elif 'm' == btype: x = block(x, dropout)
                else: raise TypeError('unknown decode block')
            return x, tuple(ds), tuple(us)


class Model(Record):
    """-> Record

    model = Model.new( ... )
    train = model.data( ... ).train( ... )
    valid = model.data( ... ).valid( ... )
    infer = model.data( ... ).infer( ... )

    """
    _new = 'dim_emb', 'dim_mid', 'dim_src', 'dim_tgt', 'cap', 'eos', 'bos'

    @staticmethod
    def new(dim_emb, dim_mid, dim_src, dim_tgt, cap, eos, bos):
        """-> Model with fields

          decode : Decode
          encode : Encode
         emb_tgt : Embed
         emb_src : Embed

        """
        assert not dim_emb % 2
        return Model(
              decode= Decode(dim_emb, name= 'decode')
            , encode= Encode(dim_emb, name= 'encode')
            , emb_tgt= Embed(dim_emb, dim_tgt, name= 'emb_tgt')
            , emb_src= Embed(dim_emb, dim_src, name= 'emb_src')
            , dim_emb= dim_emb
            , dim_tgt= dim_tgt
            , bos= bos
            , eos= eos
            , cap= cap + 1)

    def data(self, src= None, tgt= None):
        """-> Model with new fields

        position : Sinusoid
            src_ : i32 (b, ?)    source feed, in range `[0, dim_src)`
            tgt_ : i32 (b, ?)    target feed, in range `[0, dim_tgt)`
             src : i32 (b, s)    source with `eos` trimmed among the batch
             tgt : i32 (b, t)    target with `eos` trimmed among the batch, padded with `bos`
            gold : i32 (b, t)    target one step ahead, padded with `eos`
        mask_tgt : f32 (1, t, t) target mask
        mask_arr : f32 (b, 1, s) bridge mask
        mask_src : f32 (b, s, s) source mask

        """
        src_ = placeholder(tf.int32, (None, None), src, 'src_')
        tgt_ = placeholder(tf.int32, (None, None), tgt, 'tgt_')
        with scope('src'):
            with scope('not_eos'): not_eos = tf.not_equal(src_, self.eos)
            with scope('len_src'): len_src = tf.reduce_sum(tf.to_int32(not_eos), axis= 1)
            with scope('max_src'): max_src = tf.reduce_max(len_src)
            src = src_[:,:max_src]
        with scope('mask_arr'): mask_arr = tf.log(tf.expand_dims(tf.to_float(not_eos[:,:max_src]), axis= 1))
        with scope('mask_src'): mask_src = mask_arr
        with scope('tgt'):
            with scope('not_eos'): not_eos = tf.not_equal(tgt_, self.eos)
            with scope('len_tgt'): len_tgt = tf.reduce_sum(tf.to_int32(not_eos), axis= 1)
            with scope('max_tgt'): max_tgt = tf.reduce_max(len_tgt)
            tgt = tgt_[:,:max_tgt]
            gold = tf.pad(tgt, ((0,0),(0,1)), constant_values= self.eos)
            tgt  = tf.pad(tgt, ((0,0),(1,0)), constant_values= self.bos)
        with scope('mask_tgt'): mask_tgt = tf.log(tf.expand_dims(
                tf.linalg.LinearOperatorLowerTriangular(tf.ones((max_tgt + 1,) * 2)).to_dense()
                , axis= 0))
        return Model(
            position= Sinusoid(self.dim_emb, self.cap)
            , src_= src_, mask_src= mask_src, src= src
            , tgt_= tgt_, mask_tgt= mask_tgt, tgt= tgt
            , gold= gold, mask_arr= mask_arr
            , **self)

    def infer(self, minimal= True):
        """-> Model with new fields, autoregressive

        len_tgt : i32 ()     steps to unfold aka t
           pred : i32 (b, t) prediction, hard

        """
        dropout = identity
        with scope('emb_src_infer'): w = self.position(tf.shape(self.src)[1]) + dropout(self.emb_src(self.src))
        w = self.encode(w, self.mask_src, dropout, name= 'encode_infer')
        with scope('decode_infer'):
            with scope('init'):
                b,t = tf.shape(w)[0], placeholder(tf.int32, (), self.cap, 't')
                pos = self.position(t)
                i = tf.constant(0)
                x = tf.fill((b, 1), self.bos, 'x')
                p, p_shape = x[:,1:], tf.TensorShape((None, None))
                cs, cs_shape, vs, vs_shape = self.decode.cache_init(b, self.dim_emb)
            def body(i, x, p, cs, vs):
                # i : ()              time step from 0 to t
                # x : (b,          1) x_i
                # p : (b,          i) predictions
                # c : (b,     128, 1) convolution value
                # v : (b, dim_emb, i) attention values
                with scope('emb_tgt'): x = tf.expand_dims(pos[:,i], axis= -1) + self.emb_tgt(x)
                x, cs, vs = self.decode.autoreg(x, cs, vs, w, self.mask_arr, dropout)
                x = self.emb_tgt(x, name= 'logit')
                x = tf.argmax(x, axis= -1, output_type= tf.int32, name= 'pred')
                p = tf.concat((p, x), axis= -1, name= 'cache_p')
                return i + 1, x, p, cs, vs
            def cond(i, x, *_):
                with scope('cond'):
                    return ((i < t) & ~ tf.reduce_all(tf.equal(x, self.eos))) if minimal else (i < t)
            _, _, p, *_ = tf.while_loop(
                cond, body
                , (i      , x      , p      , cs      , vs      )
                , (i.shape, x.shape, p_shape, cs_shape, vs_shape)
                , back_prop= False
                , swap_memory= True
                , name= 'autoreg')
        return Model(self, len_tgt= t, pred= p)

    def valid(self, dropout= identity, smooth= None):
        """-> Model with new fields, teacher forcing

          output : f32 (b, t, dim_tgt) prediction on logit scale
            prob : f32 (b, t, dim_tgt) prediction, soft
            pred : i32 (b, t)          prediction, hard
            loss : f32 ()              prediction loss
            errt : f32 ()              error rate

        """
        with scope('emb_src_'): w = self.position(tf.shape(self.src)[1]) + dropout(self.emb_src(self.src))
        with scope('emb_tgt_'): x = self.position(tf.shape(self.tgt)[1]) + dropout(self.emb_tgt(self.tgt))
        w = self.encode(w, self.mask_src,                   dropout, name= 'encode_')
        x = self.decode(x, self.mask_tgt, w, self.mask_arr, dropout, name= 'decode_')
        y = self.emb_tgt(x, name= 'logit_')
        with scope('prob_'): prob = tf.nn.softmax(y, axis= -1)
        with scope('pred_'): pred = tf.argmax(y, axis= -1, output_type= tf.int32)
        with scope('errt_'): errt = tf.reduce_mean(tf.to_float(tf.not_equal(self.gold, pred)))
        with scope('loss_'): loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels= smooth(self.gold), logits= y)
                if smooth else
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels= self.gold, logits= y))
        return Model(self, output= y, prob= prob, pred= pred, loss= loss, errt= errt)

    def train(self, dropout= 0.1, smooth= 0.1, warmup= 4e3, beta1= 0.9, beta2= 0.98, epsilon= 1e-9):
        """-> Model with new fields, teacher forcing

        step : i64 () global update step
          lr : f32 () learning rate for the current step
          up :        update operation

        along with all the fields from `valid`

        """
        dropout, smooth = Dropout(dropout, (None, self.dim_emb, None)), Smooth(smooth, self.dim_tgt)
        self = self.valid(dropout= dropout, smooth= smooth)
        with scope('lr'):
            s = tf.train.get_or_create_global_step()
            t = tf.to_float(s + 1)
            lr = (self.dim_emb ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(self.loss, s)
        return Model(self, dropout= dropout, smooth= smooth, step= s, lr= lr, up= up)


def batch_run(sess, model, fetch, src, tgt= None, batch= None):
    if batch is None: batch = len(src)
    for i, j in partition(len(src), batch, discard= False):
        feed = {model.src_: src[i:j]}
        if tgt is not None:
            feed[model.tgt_] = tgt[i:j]
        yield sess.run(fetch, feed)
