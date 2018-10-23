from copy import copy
from util import Record
import tensorflow as tf


def profile(sess, wtr, run, feed_dict= None, prerun= 3, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wtr.add_run_metadata(meta, tag)


def pipe(*args, **kwargs):
    """see `tf.data.Dataset.from_generator`."""
    return tf.data.Dataset.from_generator(*args, **kwargs) \
                          .prefetch(1) \
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


def normalize(x, axis= -1, eps= 1e-8, name= 'normalize'):
    """returns a tensor from `x` scaled and centered across `axis`."""
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, axis, keep_dims=True)
        return (x - mean) * tf.rsqrt(var + eps * eps)


class Normalize(Record):
    """layer normalization"""

    def __init__(self, dim
                 , gain_initializer= tf.ones_initializer()
                 , bias_initializer= tf.zeros_initializer()
                 , name= 'normalize'):
        with tf.variable_scope(name):
            self.name = name
            self.gain = tf.get_variable('gain', dim, initializer= gain_initializer)
            self.bias = tf.get_variable('bias', dim, initializer= bias_initializer)

    def __call__(self, x, axis= -1, eps= 1e-8, name= None):
        with tf.variable_scope(name or self.name):
            return normalize(x, axis, eps) * self.gain + self.bias


class Smooth(Record):
    """binary smoothing if dim is None or one-hot smoothing"""

    def __init__(self, rate, dim= None, name= 'smooth'):
        self.dim = dim
        with tf.variable_scope(name):
            self.name = name
            self.rate = placeholder(tf.float32, (), rate, 'rate')
            self.shared = self.rate / (dim or 2)
            self.smooth = 1.0 - self.rate
            if dim: self.smooth += self.shared

    def __call__(self, x, name= None):
        if self.dim:
            return tf.one_hot(x, self.dim, self.smooth, self.shared, name= name or self.name)
        else:
            with tf.variable_scope(name or self.name):
                return x * self.smooth + self.shared


class Dropout(Record):
    """dropout shape must be a tuple of None or 1 or a fixed known
    dimension, such as `(None, 1, 256)`.  when applied to a tensor,
    None will be filled, and the whole shape broadcast to fit.

    """

    def __init__(self, rate, shape= None, name= 'dropout'):
        self.shape = shape
        with tf.variable_scope(name):
            self.name = name
            self.rate = placeholder(tf.float32, (), rate, 'rate')
            self.keep = 1.0 - self.rate

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            if self.shape is not None:
                shape = tf.shape(x)
                shape = [s or shape[i] for i, s in enumerate(self.shape)]
            return tf.nn.dropout(x, self.keep, shape)


class Maxout(Record):

    def __init__(self, k, name= 'maxout'):
        self.k, self.name = k, name

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            slist, shape = x.shape.as_list(), tf.shape(x)
            for i, d in enumerate(slist):
                if d is None: slist[i] = shape[i]
            slist[-1] = slist[-1] // self.k
            slist.append(self.k)
            return tf.reduce_max(tf.reshape(x, slist), -1)


class Linear(Record):
    """linear transformation from `m` to `n`"""

    def __init__(self, n, m= None, name= 'linear'):
        if m is None: m = n
        self.name = name
        self.kern = tf.get_variable(name, (m, n))

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            return tf.tensordot(x, self.kern, 1)

    def embed(self, x, name= 'embed'):
        return tf.gather(self.kern, x, name= name or self.name)

    def transpose(self, name= None):
        self = copy(self)
        self.name = name or self.name
        self.kern = tf.transpose(self.kern, name= self.name)


class Affine(Record):
    """affine transformation from `m` to `n`"""

    def __init__(self, n, m= None, name= 'affine'):
        if m is None: m = n
        with tf.variable_scope(name):
            self.name = name
            self.kern = tf.get_variable('kern', (m, n))
            self.bias = tf.get_variable('bias', n)

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            return tf.tensordot(x, self.kern, 1) + self.bias


class Multilayer(Record):
    """mlp from `m` to `n`, with `mid` dimension(s)"""

    def __init__(self, n, m= None, mid= None, act= Maxout(2), name= 'multilayer'):
        if m is None: m = n
        if mid is None: mid = m
        if isinstance(mid, int): mid = mid,
        if isinstance(act, Maxout): mid = [act.k * i for i in mid]
        self.act = act
        with tf.variable_scope(name):
            self.name = name
            self.out = Affine(n, mid[-1], 'out')
            if 1 == len(mid):
                self.mid = Affine(mid[0], m, 'mid'),
            else:
                self.mid, j = [], m
                for i in mid:
                    self.mid.append(Affine(i, j, 'm{}d'.format(i)))
                    j = i
                self.mid = tuple(self.mid)

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            for mid in self.mid:
                x = self.act(mid(x))
            return self.out(x)


class AdditiveAttention(Record):

    def __init__(self, n, m= None, name= 'attention', mid= 4, act= Maxout(2)):
        if m is None: m = n
        if mid is None: mid = m
        if isinstance(act, Maxout): mid *= act.k
        self.act = act
        with tf.variable_scope(name):
            self.name = name
            self.q = Affine(mid, m, 'q')
            self.k = Linear(mid, n, 'k')
            self.a = Linear(1, mid, 'a')

    def __call__(self, query, value, mask= None, name= None):
        # query:btq -> value:bsd -> btd
        with tf.variable_scope(name or self.name):
            # bts <- bts1 <- btsk <- (b1sk <- bsk <- bsd) + (bt1k <- btk <- btq)
            a = tf.squeeze(self.a(self.act(tf.expand_dims(self.k(value), 1) + tf.expand_dims(self.q(query), 2))), 3)
            if mask is not None: a += mask
            return tf.nn.softmax(a) @ value # btd <- bts @ bsd


class TransformerAttention(Record):
    """computes multi-head attention from `query` and `value` tensors.

    with batch size `b`, time steps `t,s`, dimensions `m,n`

    - query : b,t,m
    - value : b,s,n

    the returned tensor has shape `b,t,n`, and `mask` when supplied
    should have shape `t,s`.

    """

    def __init__(self, n, m= None, name= 'attention', num_head= None):
        if m is None: m = n
        if num_head is None: num_head = n // 64
        assert not n % num_head
        self.n, self.num_head = n, num_head
        with tf.variable_scope(name):
            self.name = name
            self.q = Linear(n, m, 'q')
            self.k = Linear(n, n, 'k')
            self.v = Linear(n, n, 'v')

    def __call__(self, query, value, mask= None, name= None):
        # query:btm -> value:bsn -> btn
        stack_split = lambda x: tf.stack(tf.split(x, self.num_head, -1)) # btn -> hbtc
        with tf.variable_scope(name or self.name):
            # hbts <- (hbtc <- btn <- btm) @ (hbcs <- hbsc <- btn <- btn)
            a = tf.matmul(stack_split(self.q(query)), stack_split(self.k(value)), transpose_b= True)
            a *= (self.n // self.num_head) ** -0.5
            if mask is not None: a += mask
            a = tf.nn.softmax(a)
            # btn <- hbtc <- hbts @ (hbsc <- bsn <- bsn)
            return tf.concat(tf.unstack(a @ stack_split(self.v(value))), -1)


class QueryAttention(Record):

    def __init__(self, n, m= None, name= 'attention', layer= Multilayer, **largs):
        if m is None: m = n
        self.n = n
        with tf.variable_scope(name):
            self.name = name
            self.q = layer(n, m, name= 'q', **largs)

    def __call__(self, query, value, mask= None, name= None, softmax= True, scale= True, head= 1):
        # query:btm -> value:bsn -> btn
        if 1 < head:
            assert not self.n % head
            stack_split = lambda x: tf.stack(tf.split(x, head, -1)) # btn -> hbtc
        with tf.variable_scope(name or self.name):
            query = self.q(query) # btn <- btm
            if 1 < head: query, value = stack_split(query), stack_split(value)
            a = tf.matmul(query, value, transpose_b= True) # bts <- btn @ (bns <- bsn)
            if softmax:
                if scale: a *= self.n ** -0.5
                if mask is not None: a += mask
                a = tf.nn.softmax(a)
            else:
                if mask is not None: a *= tf.exp(mask)
                a = tf.square(a)
                a /= tf.reduce_sum(a, -1, True) + 1e-8
            a @= value # btn <- bts @ bsn
            if 1 < head: a = tf.concat(tf.unstack(a), -1)
            return a