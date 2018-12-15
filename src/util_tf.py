from copy import copy
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


def variable(name, shape, init, initializers=
             {  'zero': tf.initializers.zeros()
              , 'unit': tf.initializers.ones()
              , 'vso1': tf.variance_scaling_initializer(1.0, 'fan_out', 'uniform')
              , 'vso2': tf.variance_scaling_initializer(2.0, 'fan_out', 'uniform')
             }):
    """wraps `tf.get_variable` to provide initializer based on usage"""
    return tf.get_variable(name, shape, initializer= initializers.get(init, init))


class Normalize(Record):
    """layer normalization"""

    def __init__(self, dim, name= 'normalize'):
        self.name = name
        with scope(name):
            self.gain = variable('gain', (1, dim, 1), 'unit')
            self.bias = variable('bias', (1, dim, 1), 'zero')

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
    f32 (b, m, t) -> f32 (b, t, n)

    """

    def __init__(self, n, m, name= 'embed'):
        self.name = name
        with scope(name): self.kern = variable('kern', (m, n), 'vso1')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            if x.dtype.is_integer:
                return tf.transpose(tf.gather(self.kern, x), (0, 2, 1))
            else:
                m , n = self.kern.shape.as_list()
                shape = tf.shape(x)
                b,d,t = (d.value or shape[i] for i, d in enumerate(x.shape))
                assert m == d
                return tf.reshape(tf.reshape(tf.transpose(x, (0, 2, 1)), (b * t, m)) @ self.kern, (b, t, n))

    def transpose(self, name= None):
        m, n = self.kern.shape.as_list()
        self = copy(self)
        self.name = name or self.name
        with scope(self.name): self.kern = tf.transpose(self.kern) * ((n / m) ** 0.5)
        return self


class Conv(Record):
    """convolution from `m` to `n` channels

    the default parameters make a position-wise affine layer

    """

    def __init__(self, n, m= None, shape= (1,), bias= False, act= None, init= None, name= 'conv'):
        if m is None: m = n
        self.act = act
        self.form = ('NCW', 'NCHW', 'NCDHW')[len(shape) - 1]
        self.name = name
        with scope(name):
            self.kern = variable('kern', shape + (m, n), init or 'vso2' if tf.nn.relu == act else 'vso1')
            self.bias = variable('bias', (1, n) + (1,) * len(shape), 'zero') if bias else None

    def __call__(self, x, padding= 'VALID', stride= None, dilation= None, name= None):
        with scope(name or self.name):
            x = tf.nn.convolution(
                x, self.kern
                , padding= padding
                , strides= stride
                , dilation_rate= dilation
                , data_format= self.form)
            if self.bias is not None: x += self.bias
            if self.act is not None: x = self.act(x)
            return x


class Multilayer(Record):
    """position-wise mlp from `m` to `n`, with `dim` type units between

    if dim is None,          m -> n

    if isinstance(dim, int), m -> dim -> act -> n

    if dim == (d1, ..., dx), m -> d1 -> act -> ... -> dx -> act -> n

    """

    def __init__(self, n, m= None, dim= None, act= tf.nn.relu, name= 'multilayer'):
        if m is None: m = n
        if dim is None: dim = ()
        if isinstance(dim, int): dim = dim,
        dim = (m,) + dim
        self.name = name
        with scope(name):
            self.layers = tuple(
                Conv(i, j, act= act, name= "layer{}".format(k)) for k, (j, i) in enumerate(zip(dim, dim[1:]), 1)
            ) + (Conv(n, dim[-1], name= 'layer'),)

    def __call__(self, x, name= None):
        with scope(name or self.name):
            for layer in self.layers: x = layer(x)
            return x


class Attention(Record):
    """computes multi-head scaled dot-product attention

    query : tensor f32 (b, d_q, t)
    value : tensor f32 (b, d_v, s)
     mask : tensor f32 (b,   t, s)
         -> tensor f32 (b, dim, t)

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
            return y
