import torch
import json
from hifi_gan.env import AttrDict
from hifi_gan.models import Generator

def load_wave_config(file_path):
    with open(file_path) as f:
        data = f.read()
    json_config = json.loads(data)
    cf = AttrDict(json_config)
    return cf

def load_model(config_file, checkpoint_path, device):
    cf = load_wave_config(config_file)
    generator = Generator(cf)
    generator.to(device = device)

    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint_dict['generator'])

    generator.eval()
    generator.remove_weight_norm()
    return generator

def generate_wave(model, mel_input, max_wave_value, device = 'cpu'):
    mel_input = mel_input.to(device = device)
    wave = model(mel_input).squeeze()
    audio = wave * max_wave_value
    audio = audio.numpy().astype('int16')
    return audio
