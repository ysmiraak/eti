from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from util_np import np, vpack


def load_spm(path):
    """-> SentencePieceProcessor

    loads a sentence piece model.

    """
    spm = SentencePieceProcessor()
    spm.load(path)
    return spm


def spm(name, path, size, bos= -1, eos= -1, unk= 0, coverage= 0.9995):
    """-> SentencePieceProcessor

    trains a sentence piece model of `size` from text file on `path`
    and saves with `name`.

    """
    SentencePieceTrainer.train(
        "--model_prefix={name} \
        --input={path} \
        --vocab_size={size} \
        --bos_id={bos} \
        --eos_id={eos} \
        --unk_id={unk} \
        --character_coverage={coverage}".format(
            coverage= coverage
            , unk= unk
            , eos= eos
            , bos= bos
            , size= size
            , path= path
            , name= name))
    return load_spm(name + ".model")


def encode(vocab, sents, length= None, dtype= np.int32, eos= '</s>'):
    """-> array dtype

    encodes `sents : seq str` with `vocab : SentencePieceProcessor`.
    returns a rank 2 array whose second axis is padded to `length` or
    the maximum length.

    """
    sents = list(map(vocab.encode_as_ids, sents))
    if length is None: length = max(map(len, sents))
    return vpack(sents, (len(sents), length), vocab[eos], dtype)


def decode(vocab, array):
    """-> str

    decodes `array : array int` with `vocab : SentencePieceProcessor`.
    if `array` has a higher rank, generates the results instead.

    """
    if 1 < array.ndim: return (decode(vocab, arr) for arr in array)
    return vocab.decode_ids(list(map(int, array)))
