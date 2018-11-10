from util import Record


master = Record(
    trial  = 'm'
    , ckpt = None
    , seed = 0
    ### data spec
    , unk = 0
    , eos = 1
    , bos = 2
    , cap = 64
    ### model spec
    , dim_src = 8192
    , dim_tgt = 8192
    # model dimension
    , dim_emb = 512
    # mlp middle layer dimension
    , dim_mid = 1024
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
    , shuffle = 2**14
    # batch size for training
    , batch_train = 64
    # batch size for validation
    , batch_valid = 512
    # instances for validation
    , total_valid = 2560
)


wide = Record(master, dim_mid= 2048)
deep = Record(master, depth= 4)


config = Record(master, trial= 'a', ckpt= None)
