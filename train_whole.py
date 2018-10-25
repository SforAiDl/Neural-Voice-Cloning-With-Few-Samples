
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
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser


# import requirements for dv3
from utils import generate_cloned_samples, Speech_Dataset
import dv3
from dv3 import build_deepvoice_3
from dv3.hparams import hparams, hparams_debug_string
from dv3.train import train as train_dv3
from dv3.train import TextDataSource,MelSpecDataSource,LinearSpecDataSource,\
                        PyTorchDataset,PartialyRandomizedSimilarTimeLengthSampler
from dv3.deepvoice3_pytorch import frontend


from utils import generate_cloned_samples, Speech_Dataset
from SpeechEmbedding import Encoder
from train_encoder import get_cloned_voices,build_encoder,get_speaker_embeddings
from train_encoder import load_checkpoint as load_checkpoint_encoder
# from train_encoder import save_checkpoint as save_checkpoint_encoder
from train_encoder import train as train_encoder


import sys
import os

# sys.path.append('./deepvoice3_pytorch')

# print(hparams)
batch_size_encoder = 64



use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


def train(dv3_model , encoder , dv3_optimizer , encoder_optimizer):
    # this training function is to train the combined model

    grad = {}
    def save_grad(name):
        def hook(grad):
            grads[name] = grad
        return hook

    # to remember the embeddings of the speakers
    dv3_model.embed_speakers.weight.register_hook(save_grad('embeddings'))

    dv3_model.zero_grad()
    encoder.zero_grad()
    #---------------ENCODER---------------------
    #foward pass on encoder
    encoder_out = encoder(inp)
    dv3_model.embed_speakers.weight.data = (encoder_out).data


    #----------------------DV3------------------
    #foward pass dv3
    _frontend = getattr(frontend, "en")

    dv3_model.eval()
    text = "hi bye"
    speaker_id = 0
    sequence = np.array(_frontend.text_to_sequence(text, p=0))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long()
    text_positions = Variable(text_positions)
    speaker_ids = None if speaker_id is None else Variable(torch.LongTensor([speaker_id]))

    if use_cuda:
        sequence = sequence.cuda()
        text_positions = text_positions.cuda()
        speaker_ids = None if speaker_ids is None else speaker_ids.cuda()

    mel_outputs, linear_outputs, alignments, done = dv3_model(
        sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    # dv3 loss function
    # backward on that
    mel_outputs.backward()
    # dv3_model.embed_speakers.weight.data = (encoder_out).data
    encoder_out.backward(grads['embeddings'])

    dv3_optimizer.step()
    encoder_optimizer.step()









if __name__=="main"

    args = docopt(__doc__)
    print("Command line args:\n",args)

    checkpoint_dv3 = args["--checkpoints-dv3"]
    checkpoint_encoder = args["--checkpoint-encoder"]
    speaker_id = None
    dv3_preset =None

    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(dirname(__file__), "data", "ljspeech")



    train_dv3_v = args["--train-dv3"]
    train_encoder_v = args["--train-encoder"]


    if not train_dv3_v and not train_encoder_v:
        print("Training whole model")
        train_dv3_v,train_encoder_v= True,True
    if train_dv3_v:
        print("Training deep voice 3 model")
    elif train_encoder_v:
        print("Training encoder model")
    else:
        assert False, "must be specified wrong args"

    os.makedirs(checkpoint_dir , exist_ok=True)

    # Input dataset definitions
    X = FileSourceDataset(TextDataSource(data_root, speaker_id))
    Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id))
    Y = FileSourceDataset(LinearSpecDataSource(data_root, speaker_id))

    # Prepare sampler
    frame_lengths = Mel.file_data_source.frame_lengths
    sampler = PartialyRandomizedSimilarTimeLengthSampler(
        frame_lengths, batch_size=hparams.batch_size)

    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X, Mel, Y)
    data_loader_dv3 = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, sampler=sampler,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)
    print("dataloader for dv3 prepared")


    dv3_model = build_deepvoice_3(dv3_preset , checkpoint_dv3)
    print("Built dv3!")

    if use_cuda:
        dv3_model = dv3_model.cuda()

    dv3_optimizer = optim.Adam((dv3_model.get_trainable_parameters(),
                           lr=hparams.initial_learning_rate, betas=(
        hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay)

    log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("Log event path for dv3: {}".format(log_event_path))
    writer_dv3 = SummaryWriter(log_dir=log_event_path)

    # ENCODER
    all_speakers = get_cloned_voices(dv3_model)
    print("Cloning Texts are produced")

    speaker_embed = get_speaker_embeddings(dv3_model)

    encoder = build_encoder()

    print("Encoder is built!")

    speech_data_encoder = Speech_Dataset(all_speakers, speaker_embed)

    criterion_encoder = nn.L1Loss()

    optimizer_encoder = torch.optim.SGD(encoder.parameters(),lr=0.0006)

    lambda1_encoder = lambda epoch: 0.6 if epoch%8000==7999 else 1#???????????
    scheduler_encoder = torch.optim.lr_scheduler.LambdaLR(optimizer_encoder, lr_lambda=lambda1_encoder)

    data_loader_encoder = DataLoader(speech_data_encoder, batch_size=batch_size_encoder, shuffle=True, drop_last=True)
    # Training The Encoder
    dataiter_encoder = iter(data_loader_encoder)

    if use_cuda:
        encoder = encoder.cuda()

    if os.path.isfile(checkpoint_encoder):
        encoder, optimizer_encoder = load_checkpoint_encoder(encoder, optimizer_encoder)

    if train_encoder_v and train_dv3_v:
        try:
            train()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
    elif train_encoder_v:
        try:
            train_encoder(encoder , data_loader_encoder , optimizer_encoder,scheduler_encoder,criterion_encoder,epochs=100000)
        except KeyboardInterrupt:

            print("KeyboardInterrupt")

    elif train_dv3_v:
        try:
            train_dv3(dv3_model ,data_loader_dv3, dv3_optimizer, writer_dv3,
                  init_lr=hparams.initial_learning_rate,
                  checkpoint_dir=checkpoint_dv3,
                  checkpoint_interval=hparams.checkpoint_interval,
                  nepochs=hparams.nepochs,
                  clip_thresh=hparams.clip_thresh,
                  train_seq2seq=True, train_postnet=True)
        except KeyboardInterrupt:

            print("KeyboardInterrupt")
    else:
        assert False , "Wrongs arguments specified"

    print("Finished")
    sys.exit(0)
