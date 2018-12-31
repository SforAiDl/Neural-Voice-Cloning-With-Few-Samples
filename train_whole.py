
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
from dv3.train import collate_fn
from dv3.deepvoice3_pytorch import frontend
from dv3.train import sequence_mask
from dv3.train import save_checkpoint as save_checkpoint_dv3
from dv3.train import save_states as save_states_dv3
from tensorboardX import SummaryWriter

# requirements for encoder
from utils import generate_cloned_samples, Speech_Dataset
from Encoder import Encoder
from train_encoder import get_cloned_voices,build_encoder,get_speaker_embeddings
from train_encoder import load_checkpoint as load_checkpoint_encoder
from train_encoder import save_checkpoint as save_checkpoint_encoder
from train_encoder import train as train_encoder


import sys
import os

# sys.path.append('./deepvoice3_pytorch')

# print(hparams)
batch_size_encoder = 16


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


def train(model_dv3,model_encoder,
            data_loader_dv3,
            optimizer_dv3,
            init_lr_dv3=0.002,
            checkpoint_dir_dv3=None,
            clip_thresh = 1.0,
            data_loader_encoder=None,
            optimizer_encoder=None,
            scheduler_encoder=None,
            checkpoint_interval=None,
            nepochs=None):
    # this training function is to train the combined model

    grad = {}
    def save_grad(name):
        def hook(grad):
            grads[name] = grad
        return hook

    # to remember the embeddings of the speakers
    model_dv3.embed_speakers.weight.register_hook(save_grad('embeddings'))

    if use_cuda:
        model_dv3 = model_dv3.cuda()
        model_encoder = model_encoder.cuda()
    linear_dim = model_dv3.linear_dim
    r = hparams.outputs_per_step
    downsample_step = hparams.downsample_step
    current_lr = init_lr_dv3

    binary_criterion_dv3 = nn.BCELoss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.0
        for step, (x, input_lengths, mel, y, positions, done, target_lengths,
                   speaker_ids) \
                in tqdm(enumerate(data_loader_dv3)):


            model_dv3.zero_grad()
            encoder.zero_grad()

            #Declaring Requirements
            model_dv3.train()
            ismultispeaker = speaker_ids is not None
            # Learning rate schedule
            if hparams.lr_schedule is not None:
                lr_schedule_f = getattr(dv3.lrschedule, hparams.lr_schedule)
                current_lr = lr_schedule_f(
                    init_lr, global_step, **hparams.lr_schedule_kwargs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            optimizer_dv3.zero_grad()

            # Used for Position encoding
            text_positions, frame_positions = positions

            # Downsample mel spectrogram
            if downsample_step > 1:
                mel = mel[:, 0::downsample_step, :].contiguous()

            # Lengths
            input_lengths = input_lengths.long().numpy()
            decoder_lengths = target_lengths.long().numpy() // r // downsample_step

            voice_encoder = mel.view(mel.shape[0],1,mel.shape[1],mel.shape[2])
            # Feed data
            x, mel, y = Variable(x), Variable(mel), Variable(y)
            voice_encoder = Variable(voice_encoder)
            text_positions = Variable(text_positions)
            frame_positions = Variable(frame_positions)
            done = Variable(done)
            target_lengths = Variable(target_lengths)
            speaker_ids = Variable(speaker_ids) if ismultispeaker else None
            if use_cuda:
                x = x.cuda()
                text_positions = text_positions.cuda()
                frame_positions = frame_positions.cuda()
                y = y.cuda()
                mel = mel.cuda()
                voice_encoder = voice_encoder.cuda()
                done, target_lengths = done.cuda(), target_lengths.cuda()
                speaker_ids = speaker_ids.cuda() if ismultispeaker else None

            # Create mask if we use masked loss
            if hparams.masked_loss_weight > 0:
                # decoder output domain mask
                decoder_target_mask = sequence_mask(
                    target_lengths / (r * downsample_step),
                    max_len=mel.size(1)).unsqueeze(-1)
                if downsample_step > 1:
                    # spectrogram-domain mask
                    target_mask = sequence_mask(
                        target_lengths, max_len=y.size(1)).unsqueeze(-1)
                else:
                    target_mask = decoder_target_mask
                # shift mask
                decoder_target_mask = decoder_target_mask[:, r:, :]
                target_mask = target_mask[:, r:, :]
            else:
                decoder_target_mask, target_mask = None, None

            #apply encoder model



            model_dv3.embed_speakers.weight.data = (encoder_out).data
            # Apply dv3 model
            mel_outputs, linear_outputs, attn, done_hat = model_dv3(
                    x, mel, speaker_ids=speaker_ids,
                    text_positions=text_positions, frame_positions=frame_positions,
                    input_lengths=input_lengths)



            # Losses
            w = hparams.binary_divergence_weight

            # mel:
            mel_l1_loss, mel_binary_div = spec_loss(
                    mel_outputs[:, :-r, :], mel[:, r:, :], decoder_target_mask)
                mel_loss = (1 - w) * mel_l1_loss + w * mel_binary_div

            # done:
            done_loss = binary_criterion(done_hat, done)

            # linear:
            n_priority_freq = int(hparams.priority_freq / (fs * 0.5) * linear_dim)
                linear_l1_loss, linear_binary_div = spec_loss(
                    linear_outputs[:, :-r, :], y[:, r:, :], target_mask,
                    priority_bin=n_priority_freq,
                    priority_w=hparams.priority_freq_weight)
                linear_loss = (1 - w) * linear_l1_loss + w * linear_binary_div

            # Combine losses
            loss_dv3 = mel_loss + linear_loss + done_loss
            loss_dv3 = mel_loss + done_loss
            loss_dv3 = linear_loss

            # attention
            if hparams.use_guided_attention:
                soft_mask = guided_attentions(input_lengths, decoder_lengths,
                                              attn.size(-2),
                                              g=hparams.guided_attention_sigma)
                soft_mask = Variable(torch.from_numpy(soft_mask))
                soft_mask = soft_mask.cuda() if use_cuda else soft_mask
                attn_loss = (attn * soft_mask).mean()
                loss_dv3 += attn_loss

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_states_dv3(
                    global_step, writer, mel_outputs, linear_outputs, attn,
                    mel, y, input_lengths, checkpoint_dir)
                save_checkpoint_dv3(
                    model, optimizer, global_step, checkpoint_dir, global_epoch,
                    train_seq2seq, train_postnet)

            if global_step > 0 and global_step % hparams.eval_interval == 0:
                eval_model(global_step, writer, model, checkpoint_dir, ismultispeaker)

            # Update
            loss_dv3.backward()
            encoder_out.backward(grads['embeddings'])

            optimizer_dv3.step()
            optimizer_encoder.step()

            # if clip_thresh> 0:
            #     grad_norm = torch.nn.utils.clip_grad_norm(
            #         model.get_trainable_parameters(), clip_thresh)
            global_step += 1
            running_loss += loss.data[0]

        averaged_loss = running_loss / (len(data_loader))

        print("Loss: {}".format(running_loss / (len(data_loader))))

        global_epoch += 1


    # dv3 loss function
    # backward on that
    mel_outputs.backward()
    # dv3_model.embed_speakers.weight.data = (encoder_out).data


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

    dv3.train._frontend = getattr(frontend, hparams.frontend)
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

    data_loader_encoder = data_utils.DataLoader(speech_data_encoder, batch_size=batch_size_encoder, shuffle=True, drop_last=True)
    # Training The Encoder
    dataiter_encoder = iter(data_loader_encoder)

    if use_cuda:
        encoder = encoder.cuda()

    if checkpoint_encoder!=None and os.path.isfile(checkpoint_encoder):
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
