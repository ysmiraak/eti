from util import Record


config = Record(
    trial  = 'm'
    , ckpt = None
    , seed = 0
    ### data
    , unk = 0
    , eos = 1
    , bos = 2
    , cap = 64
    ### model
    , dim_src = 8192
    , dim_tgt = 8192
    , dim_emb = 512
    , dim_mid = 2048
    , depth   = 2
    ### batch
    , batch_train = 64
    , batch_valid = 256
    , total_valid = 4096
)


paths = Record(
    log = "~/cache/tensorboard-logdir/eti"
    , raw = "../data"
    , pred = "../trial/pred"
    , ckpt = "../trial/ckpt"
    , data = "../trial/data"
)


train = Record(
      dropout = 0.1
    , smooth  = 0.1
    , warmup  = 4e3
    , beta1   = 0.9
    , beta2   = 0.98
    , epsilon = 1e-9
)
