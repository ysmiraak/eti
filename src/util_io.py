from os.path import expanduser, join
from util import Record
import pickle


path = Record(
    log = expanduser("~/cache/tensorboard-logdir/eti")
    # , raw = expanduser("~/data/wmt/de-en")
    , raw = "../data"
    , pred = "../trial/pred"
    , ckpt = "../trial/ckpt"
    , data = "../trial/data"
)


def pform(path, *names, sep= ''):
    """formats a path as `path` followed by `names` joined with `sep`."""
    return join(path, sep.join(map(str, names)))


def load_txt(filename):
    """yields lines from text file."""
    with open(filename) as file:
        yield from (line[:-1] for line in file)


def save_txt(filename, lines):
    """writes lines to text file."""
    with open(filename, 'w') as file:
        for line in lines:
            print(line, file= file)


def load_pkl(filename):
    """loads pickle file."""
    with open(filename, 'rb') as dump:
        return pickle.load(dump)


def save_pkl(filename, obj):
    "saves to pickle file."
    with open(filename, 'wb') as dump:
        pickle.dump(obj, dump)
