
from docopt import docopt


import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils import data as data_utils
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
import numpy as np
from numba import jit


from utils import generate_cloned_samples, Speech_Dataset
import dv3

import sys
import os

# sys.path.append('./deepvoice3_pytorch')
from dv3 import build_deepvoice_3
from SpeechEmbedding import Encoder

# print(hparams)




if __name__=="main"

    args = docopt(__doc__)
    print("Cmmand line args:\n",args)

    checkpoint_dv3 = args["--checkpoints-dv3"]
    checkpoint_encoder = args["--checkpoint-encoder"]

    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(dirname(__file__), "data", "ljspeech")

    

    train_dv3 = args["--train-dv3"]
    train_encoder = args["--train-encoder"]

    if not train_dv3:
        train_dv3 = True
    if not train_encoder:
        train_encoder = True
    if train_dv3:
        print("Training seq2seq model")
    elif train_encoder:
        print("Training postnet model")
    else:
        assert False, "must be specified wrong args"
