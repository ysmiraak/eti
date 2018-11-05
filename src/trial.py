from util import Record


master = Record(
    trial  = 'm'
    , ckpt = None
    , seed = 0
    ### model spec
    , dim_src = 256
    , dim_tgt = 256
    , cap     = 256
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
    , shuffle = 2**14
    # batch size for training
    , batch_train = 64
    # batch size for validation
    , batch_valid = 512
)


wide = Record(master, dim= 512, dim_mid= 1024)
deep = Record(master, depth= 4)


config = Record(master, trial= 'a', ckpt= None)
