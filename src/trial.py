from util import Record


config = Record(
    trial  = None
    , ckpt = None
    , seed = 0
    ### data
    , unk = 0
    , eos = 1
    , bos = 2
    , cap = 64
    ### model
    , dim_voc = 8192
    , dim_emb = 512
    , dim_mid = 2048
    ### batch
    , batch_train = 128
    , batch_infer = 256
    , batch_valid = 512
)


paths = Record(
    log = "~/cache/tensorboard-logdir/tau"
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
