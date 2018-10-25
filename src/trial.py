from os.path import expanduser, join
from util import Record


path = Record(
    log = expanduser('~/cache/tensorboard-logdir/eti')
    , pred = '../trial/pred'
    , ckpt = '../trial/ckpt'
    , data = '../trial/data'
    , index_src = 'index_src'
    , index_tgt = 'index_tgt'
    , train_src = 'train_src'
    , train_tgt = 'train_tgt'
    , valid_src = 'valid_src'
    , valid_tgt = 'valid_tgt'
)


def pform(path, *names, sep= ''):
    """format a path as `path` followed by `names` joined with `sep`."""
    return join(path, sep.join(map(str, names)))


master = Record(
    trial  = 'm'
    , ckpt = 875520
    ### model spec
    , dim_src = 256
    , dim_tgt = 256
    , cap_tgt = 256
    , cap_src = 256
    # model dimension
    , dim     = 256
    # mlp middle layer dimension
    , dim_mid = 512
    # encoder and decoder depth
    , depth   = 2
    ### regularization
    , dropout = 0.1
    , smooth  = 0.1
    ### adam optimizer
    , warmup  = 4e3
    , beta1   = 0.9
    , beta2   = 0.98
    , epsilon = 1e-9
    ### training schedule
    # batch size for training
    , train_batch = 64
    # interval between training steps for validation
    , valid_inter = 258
)
# batch size for validation
master.valid_batch = master.train_batch * 8
# number of validation instances for summary
master.valid_total = master.valid_batch * 4


wide = Record(master, dim= 512, dim_mid= 1024)
deep = Record(master, depth= 4)


config = Record(master, trial= 'm')
