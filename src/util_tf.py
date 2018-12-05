from util import Record, identity
import tensorflow as tf


def profile(sess, wtr, run, feed_dict= None, prerun= 3, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wtr.add_run_metadata(meta, tag)


def pipe(*args, prefetch= 1, repeat= -1, name= 'pipe', **kwargs):
    """see `tf.data.Dataset.from_generator`."""
    with tf.variable_scope(name):
        return tf.data.Dataset.from_generator(*args, **kwargs) \
                              .repeat(repeat) \
                              .prefetch(prefetch) \
                              .make_one_shot_iterator() \
                              .get_next()


def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`.

    if tensor `x` is given, converts and uses it as default.

    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)


def variable(name, shape, init):
    # todo doc
    if 'zero' == init:
        init = tf.initializers.zeros()
    elif 'one' == init:
        init = tf.initializers.ones()
    elif 'vs1' == init:
        init = tf.variance_scaling_initializer(1.0, 'fan_avg', 'uniform')
    elif 'vs2' == init:
        init = tf.variance_scaling_initializer(2.0, 'fan_avg', 'uniform')
    else:
        assert 0.0 < init
        init = tf.random_uniform_initializer(-init, init)
    return tf.get_variable(name, shape, initializer= init)


class Normalize(Record):
    """layer normalization"""

    def __init__(self, dim, name= 'normalize'):
        self.name = name
        with tf.variable_scope(name):
            self.gain = variable('gain', (1, dim, 1), 'one')
            self.bias = variable('bias', (1, dim, 1), 'zero')

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            mean, var = tf.nn.moments(x, 1, keep_dims= True)
            return (x - mean) * tf.rsqrt(var + 1e-12) * self.gain + self.bias


class Smooth(Record):
    """binary smoothing if dim is None or one-hot smoothing"""

    def __init__(self, rate, dim= None, name= 'smooth'):
        self.dim = dim
        self.name = name
        with tf.variable_scope(name):
            self.rate = placeholder(tf.float32, (), rate, 'rate')
            self.shared = self.rate / (dim or 2)
            self.smooth = 1.0 - self.rate

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            if self.dim:
                return tf.one_hot(x, self.dim, self.smooth + self.shared, self.shared)
            else:
                return x * self.smooth + self.shared


class Dropout(Record):
    """dropout shape must be a tuple of None or 1 or a fixed known
    dimension, such as `(None, 256, 1)`.  when applied to a tensor,
    None will be filled, and the whole shape broadcast to fit.

    """

    def __init__(self, rate, shape= None, name= 'dropout'):
        self.shape = shape
        self.name = name
        with tf.variable_scope(name):
            self.rate = placeholder(tf.float32, (), rate, 'rate')
            self.keep = 1.0 - self.rate

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            if self.shape is not None:
                shape = tf.shape(x)
                shape = [s or shape[i] for i, s in enumerate(self.shape)]
            return tf.nn.dropout(x, self.keep, shape)


class Embed(Record):

    def __init__(self, n, m, name= 'embed'):
        self.name = name
        with tf.variable_scope(name):
            init = (6 / (m / n + 1)) ** 0.5
            self.kern = variable('kern', (m, n), init)

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            return tf.transpose(tf.gather(self.kern, x), (0, 2, 1))


class Linear(Record):
    """linear transformation from `m` to `n`"""

    def __init__(self, n, m= None, name= 'linear'):
        if m is None: m = n
        self.name = name
        with tf.variable_scope(name):
            self.kern = variable('kern', (m, n), 'vs1')

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            shape = tf.shape(x)
            shape = [s.value or shape[i] for i, s in enumerate(x.shape)]
            return tf.reshape(tf.reshape(x, (-1, shape[-1])) @ self.kern, (shape[0], shape[1], -1))


class Conv(Record):
    """convolution from `m` to `n` channels.

    the default parameters make a position-wise affine layer.

    """

    def __init__(self, n, m= None, shape= (1,), act= None, name= 'conv'):
        if m is None: m = n
        self.act = act or identity
        self.form = ('NCW', 'NCHW', 'NCDHW')[len(shape) - 1]
        self.name = name
        with tf.variable_scope(name):
            self.kern = variable('kern', shape + (m, n), 'vs2' if tf.nn.relu == act else 'vs1')
            self.bias = variable('bias', (1, n) + (1,) * len(shape), 'zero')

    def __call__(self, x, padding= 'VALID', stride= None, dilation= None, name= None):
        with tf.variable_scope(name or self.name):
            return self.act(self.bias + tf.nn.convolution(
                x, self.kern
                , padding= padding
                , strides= stride
                , dilation_rate= dilation
                , data_format= self.form))


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
        with tf.variable_scope(name):
            self.layers = tuple(
                Conv(i, j, act= act, name= "layer{}".format(k)) for k, (j, i) in enumerate(zip(dim, dim[1:]), 1)
            ) + (Conv(n, dim[-1], name= 'layer'),)

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            for layer in self.layers: x = layer(x)
            return x


class Attention(Record):
    """computes multi-head scaled dot-product attention

    query : tensor f32 (b, d_q, t)
    value : tensor f32 (b, d_v, s)
     mask : tensor f32 (b,   s, t)
         -> tensor f32 (b, dim, t)

    `dim` must be divisible by `head`

    `mask` has on-values 0 and off-values -inf

    """

    def __init__(self, dim, d_q= None, d_v= None, name= 'attention'):
        if d_q is None: d_q = dim
        if d_v is None: d_v = dim
        self.dim = dim
        self.name = name
        with tf.variable_scope(name):
            self.q = Conv(dim, d_q, name= 'q')
            self.k = Conv(dim, d_v, name= 'k')
            self.v = Conv(dim, d_v, name= 'v')

    def __call__(self, query, value, mask= None, name= None, head= 8):
        assert not self.dim % head
        if 1 < head:
            fork = lambda x: tf.stack(tf.split(x, head, axis= 1))
            join = lambda x: tf.concat(tf.unstack(x), axis= 1)
        else:
            fork = join = identity
        with tf.variable_scope(name or self.name):
            k = fork(self.k(value)) # hbcs <- bds <- bvs
            v = fork(self.v(value)) # hbcs <- bds <- bvs
            q = fork(self.q(query)) # hbct <- bdt <- bqt
            a = tf.matmul(k, q, transpose_a= True) # hbst <- (hbsc <- hbcs) @ hbct
            a *= ((self.dim // head) ** -0.5)
            if mask is not None: a += mask
            a = tf.nn.softmax(a, axis= -2)
            return join(v @ a) # bdt <- bhct <- hbcs @ hbst
