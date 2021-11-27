import matplotlib
import matplotlib.pylab as plt

import sys
import numpy as np
import torch

# from hparams import create_hparams
from model import Tacotron2
# from layers import TacotronSTFT, STFT
# from audio_processing import griffin_lim
# from train import load_model
from text import text_to_sequence
# from denoiser import Denoiser

import config_hparams as hparams
hparams.sampling_rate = 22050

checkpoint_path = "./checkpoint/tacotron2_statedict.pt"
model = Tacotron2(hparams)
model.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu')['state_dict'])
model.eval()

text = "Waveglow is really awesome! error"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

print(mel_outputs.shape, mel_outputs_postnet.shape, type(mel_outputs_postnet))