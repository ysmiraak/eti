from util import Record


master = Record(
    trial  = 'm'
    , ckpt = None
    , seed = 0
    ### data spec
    , unk = 0
    , eos = 1
    , bos = -1
    , cap = 64
    ### model spec
    , dim_src = 8192
    , dim_tgt = 8192
    , dim_emb = 512
    , dim_mid = 1024
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
    , batch_train = 64
    , batch_valid = 512
    , total_valid = 2560
)


wide = Record(master, dim_mid= 2048)
deep = Record(master, depth= 4)


config = Record(master, trial= 'j', ckpt= None)
