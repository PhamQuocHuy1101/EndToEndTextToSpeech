from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from scipy.io.wavfile import write

import tacotron2_inference as ti
import hifi_gan_inference as hi

MAX_WAV_VALUE = 32768.0

class EndToEndTTS():
    def __init__(self, mel_generator, wave_generator, audio_cf, device):
        self.mel_generator = mel_generator
        self.wave_generator = wave_generator
        self.audio_cf = audio_cf
        self.device = device

    def inference(self, text, output_file):
        with torch.no_grad():
            _, mel_outputs_postnet, _, _ = ti.generate_mel(self.mel_generator, text, self.device)
            wave = hi.generate_wave(self.wave_generator, mel_outputs_postnet, self.audio_cf['max_wave_value'], self.device)
            write(output_file, self.audio_cf['sampling_rate'], wave)
            return True