from util import Record
import tensorflow as tf


scope = tf.variable_scope


def profile(sess, wtr, run, feed_dict= None, prerun= 3, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wtr.add_run_metadata(meta, tag)


def pipe(*args, prefetch= 1, repeat= -1, name= 'pipe', **kwargs):
    """see `tf.data.Dataset.from_generator`"""
    with scope(name):
        return tf.data.Dataset.from_generator(*args, **kwargs) \
                              .repeat(repeat) \
                              .prefetch(prefetch) \
                              .make_one_shot_iterator() \
                              .get_next()


def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`

    if tensor `x` is given, converts and uses it as default

    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)


def variable(name, shape, init= 'rand', initializers=
             {  'zero': tf.initializers.zeros()
              , 'unit': tf.initializers.ones()
              , 'rand': tf.glorot_uniform_initializer()
             }):
    """wraps `tf.get_variable` to provide initializer based on usage"""
    return tf.get_variable(name, shape, initializer= initializers.get(init, init))


class Normalize(Record):
    """layer normalization"""

    def __init__(self, dim, name= 'normalize'):
        self.name = name
        with scope(name):
            self.gain = variable('gain', (1, dim, 1), init= 'unit')
            self.bias = variable('bias', (1, dim, 1), init= 'zero')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            mean, var = tf.nn.moments(x, 1, keep_dims= True)
            return (x - mean) * tf.rsqrt(var + 1e-12) * self.gain + self.bias


class Smooth(Record):
    """binary smoothing if dim is None or channel-last one-hot smoothing"""

    def __init__(self, rate, dim= None, name= 'smooth'):
        self.dim = dim
        self.name = name
        with scope(name):
            self.rate = placeholder(tf.float32, (), rate, 'rate')
            self.shared = self.rate / (dim or 2)
            self.smooth = 1.0 - self.rate

    def __call__(self, x, name= None):
        with scope(name or self.name):
            if self.dim:
                return tf.one_hot(x, self.dim, self.smooth + self.shared, self.shared)
            else:
                return x * self.smooth + self.shared


class Dropout(Record):
    """dropout shape may contain None (to be dynamically filled) or 1 (to
    be broadcasted) or some fixed dimension, such as `(None, 256, 1)`

    """

    def __init__(self, rate, shape= None, name= 'dropout'):
        self.shape = shape
        self.name = name
        with scope(name):
            self.rate = placeholder(tf.float32, (), rate, 'rate')
            self.keep = 1.0 - self.rate

    def __call__(self, x, name= None):
        with scope(name or self.name):
            if self.shape is not None:
                shape = tf.shape(x)
                shape = [s or shape[i] for i, s in enumerate(self.shape)]
            return tf.nn.dropout(x, self.keep, shape)


class Embed(Record):
    """input and output embedding

    i32 (b, t)    -> f32 (b, n, t)
    f32 (b, n, t) -> f32 (b, t, m)

    """

    def __init__(self, n, m, name= 'embed'):
        self.name = name
        with scope(name):
            self.logit = variable('kern', (n, m))
            self.embed = tf.transpose(self.logit) * (n ** 0.5)

    def __call__(self, x, name= None):
        with scope(name or self.name):
            if x.dtype.is_integer:
                return tf.transpose(tf.gather(self.embed, x), (0, 2, 1))
            else:
                n , m = self.logit.shape.as_list()
                shape = tf.shape(x)
                b,d,t = (d.value or shape[i] for i, d in enumerate(x.shape))
                assert n == d
                return tf.reshape(tf.reshape(tf.transpose(x, (0, 2, 1)), (b * t, n)) @ self.logit, (b, t, m))


class Conv(Record):
    """convolution from `m` to `n` channels

    the default parameters make a position-wise linear layer

    """

    def __init__(self, n, m= None, size= 1, name= 'conv'):
        if m is None: m = n
        self.name = name
        with scope(name):
            self.kern = variable('kern', (size, m, n))

    def __call__(self, x, name= None):
        with scope(name or self.name):
            return tf.nn.convolution(x, self.kern, padding= 'VALID', data_format= 'NCW')

    def shape(self):
        return tuple(self.kern.shape.as_list())


class SepConv(Record):
    """separable convolution from `m` to `n` channels"""

    def __init__(self, n, m= None, size= 2, name= 'conv'):
        if m is None: m = n
        self.name = name
        with scope(name):
            self.kern_depthwise = variable('kern_depthwise', (1, size, m, 1))
            self.kern_pointwise = variable('kern_pointwise', (1,    1, m, n))

    def __call__(self, x, name= None):
        with scope(name or self.name):
            return tf.squeeze( # bdt
                tf.nn.separable_conv2d(
                    tf.expand_dims(x, axis= 2) # bd1t
                    , depthwise_filter= self.kern_depthwise  # 1sm1
                    , pointwise_filter= self.kern_pointwise  # 11mn
                    , strides= (1, 1, 1, 1)
                    , padding= 'VALID'
                    , data_format= 'NCHW')
                , axis= 2)

    def shape(self):
        _, s, _, _ = self.kern_depthwise.shape.as_list()
        _, _, m, n = self.kern_pointwise.shape.as_list()
        return (s, m, n)


class Attention(Record):
    """computes multi-head scaled dot-product attention

    query : tensor f32 (b, d_q, t)
    value : tensor f32 (b, d_v, s)
     mask : tensor f32 (b,   t, s)
         -> tensor f32 (b, d_q, t)

    `dim` must be divisible by `head`

    `mask` has on-values 0 and off-values -inf

    """

    def __init__(self, dim, d_q= None, d_v= None, name= 'attention'):
        if d_q is None: d_q = dim
        if d_v is None: d_v = dim
        self.dim = dim
        self.name = name
        with scope(name):
            self.v = Conv(dim, d_v, name= 'v')
            self.k = Conv(dim, d_v, name= 'k')
            self.q = Conv(dim, d_q, name= 'q')
            self.p = Conv(d_q, dim, name= 'p')

    def __call__(self, query, value, mask= None, name= None, head= 8):
        assert not self.dim % head
        with scope(name or self.name):
            v = self.v(value) # bds <- bvs
            k = self.k(value) # bds <- bvs
            q = self.q(query) # bdt <- bqt
            if 1 < head:
                v = tf.stack(tf.split(v, head, axis= 1)) # hbcs <- bds
                k = tf.stack(tf.split(k, head, axis= 1)) # hbcs <- bds
                q = tf.stack(tf.split(q, head, axis= 1)) # hbct <- bdt
            a = tf.matmul(q, k, transpose_a= True) # hbts <- (hbtc <- hbct) @ hbcs
            a *= ((self.dim // head) ** -0.5)
            if mask is not None: a += mask
            a = tf.nn.softmax(a, axis= -1)
            y = tf.matmul(v, a, transpose_b= True) # hbct <- hbcs @ (hbst <- hbts)
            if 1 < head: y = tf.concat(tf.unstack(y), axis= 1) # bdt <- hbct
            return self.p(y)
