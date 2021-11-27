import numpy as np
import torch

from tacotron2_mini.model import Tacotron2
from tacotron2_mini.text import text_to_sequence

from tacotron2_mini import config_hparams as hparams

def load_model(checkpoint_path, device = 'cpu'):
    model = Tacotron2(hparams)
    model.load_state_dict(torch.load(checkpoint_path, map_location = device)['state_dict'])
    model.eval()
    return model

def generate_mel(model, text, device):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
    sequence.to(device = device)

    # mel_outputs, mel_outputs_postnet, file, alignments 
    return model.inference(sequence)

# print(mel_outputs.shape, mel_outputs_postnet.shape, type(mel_outputs_postnet))