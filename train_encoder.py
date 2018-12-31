from docopt import docopt
import sys
from os.path import dirname, join
from tqdm import tqdm, trange
from datetime import datetime

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
from Encoder import Encoder

# print(hparams)
batch_size_encoder = 16


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

def get_cloned_voices(model,no_speakers = 108,no_cloned_texts = 23):
    try:
        with open("./Cloning_Audio/speakers_cloned_voices_mel.p" , "rb") as fp:
            cloned_voices = pickle.load(fp)
    except:
        cloned_voices = generate_cloned_samples(model)
    if(np.array(cloned_voices).shape != (no_speakers , no_cloned_texts)):
        cloned_voices = generate_cloned_samples(model,"./Cloning_Audio/cloning_text.txt" ,no_speakers,True,0)
    print("Cloned_voices Loaded!")
    return cloned_voices

# Assumes that only Deep Voice 3 is given
def get_speaker_embeddings(model):
    '''
        return the speaker embeddings and its shape from deep voice 3
    '''
    embed = model.embed_speakers.weight.data
    # shape = embed.shape
    return embed

def build_encoder():
    encoder = Encoder()
    return encoder


def save_checkpoint(model, optimizer, checkpoint_path, epoch):

    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_epoch": epoch,
        "epoch":epoch+1,

    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def load_checkpoint(encoder, optimizer, path='checkpoints/encoder_checkpoint.pth'):

    checkpoint = torch.load(path)

    encoder.load_state_dict(checkpoint["state_dict"])

    print('Encoder state restored')

    optimizer.load_state_dict(checkpoint["optimizer"])

    print('Optimizer state restored')

    return encoder, optimizer

def my_collate(batch):
    data = [item[0] for item in batch]
    samples = [text.shape[0] for text in data]
    max_size = data[0].shape[1]
    max_samples = np.amax(np.array(samples))
    for i, i_element in enumerate(data):
        final = torch.zeros(int(max_samples), max_size, 80)
        final[:data[i].shape[0], :, :] += torch.from_numpy(i_element).type(torch.FloatTensor)
        data[i]=torch.unsqueeze(final, 0)
    data = torch.cat(data, 0)
    target = np.stack([item[1] for item in batch], 0)
    target = torch.from_numpy(target)
    return [data, target]

def train_encoder(encoder, data, optimizer, scheduler, criterion, epochs=100000, after_epoch_download=1000):

    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

    for i in range(epochs):

        epoch_loss=0.0

        for i_element, element in enumerate(data):

            voice, embed = element[0], element[1]

            input_to_encoder = Variable(voice.type(torch.cuda.FloatTensor))

            optimizer.zero_grad()

            output_from_encoder = encoder(input_to_encoder)

            embeddings = Variable(embed.type(torch.cuda.FloatTensor))

            loss = criterion(output_from_encoder,embeddings)

            loss.backward()

            scheduler.step()
            optimizer.step()

            epoch_loss+=loss


        if i%100==99:
            save_checkpoint(encoder,optimizer,"encoder_checkpoint.pth",i)
            print(i, ' done')
            print('Loss for epoch ', i, ' is ', loss)

def download_file(file_name=None):
    from google.colab import files
    files.download(file_name)


batch_size=64

if __name__ == "__main__":

    #Load Deep Voice 3
    # Pre Trained Model
    print("start")
    dv3_model = build_deepvoice_3(True)
    print("dv3 built")
    all_speakers = get_cloned_voices(dv3_model)
    print("Cloning Texts are produced")

    speaker_embed = get_speaker_embeddings(dv3_model)

    encoder = build_encoder()

    print("Encoder is built!")


    speech_data = Speech_Dataset(all_speakers, speaker_embed, sampler=True)

    criterion = nn.L1Loss()

    optimizer = torch.optim.SGD(encoder.parameters(),lr=0.0006)

    lambda1 = lambda epoch: 0.6 if epoch%8000==7999 else 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


    data_loader = DataLoader(speech_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn = my_collate)
    # Training The Encoder

    encoder = encoder.cuda()

    if os.path.isfile('checkpoints/encoder_checkpoint.pth'):
        encoder, optimizer = load_checkpoint(encoder, optimizer)

    try:
        train_encoder(encoder, data_loader, optimizer, scheduler, criterion, epochs=100000)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    print("Finished")
    sys.exit(0)
