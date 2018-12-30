import torch
import numpy as np
import librosa
import librosa.display
# import IPython
# from IPython.display import Audio
# need this for English text processing frontend
import nltk

import dv3.train
import dv3.synthesis
# print(os.getcwd())

import dv3.hparams
import json

from dv3.train import build_model
from dv3.train import restore_parts, load_checkpoint
from dv3.synthesis import tts as _tts


from dv3.deepvoice3_pytorch import frontend

# print(os.getcwd())


def build_deepvoice_3(pretrained = True , preset = None ,checkpoint_path = None):
    if preset is None:
        preset = "./dv3/deepvoice3_vctk.json"

    if checkpoint_path is None:
        checkpoint_path = "./checkpoint_step000090000.pth"

    # Newly added params. Need to inject dummy values
    for dummy, v in [("fmin", 0), ("fmax", 0),
                    ("rescaling", False),
                    ("rescaling_max", 0.999),
                    ("allow_clipping_in_normalization", False)]:

        if dv3.hparams.hparams.get(dummy) is None:
            dv3.hparams.hparams.add_hparam(dummy, v)
    # Load parameters from preset
    with open(preset) as f:
        dv3.hparams.hparams.parse_json(f.read())

    # Tell we are using multi-speaker DeepVoice3
    dv3.hparams.hparams.builder = "deepvoice3_multispeaker"

    # Inject frontend text processor
    dv3.synthesis._frontend = getattr(frontend, "en")
    dv3.train._frontend =  getattr(frontend, "en")

    # alises
    fs = dv3.hparams.hparams.sample_rate
    hop_length = dv3.hparams.hparams.hop_size
    model = build_model()
    if(pretrained):
        model = load_checkpoint(checkpoint_path, model, None, True)


    return model
    # model = build_deepvoice_3()
