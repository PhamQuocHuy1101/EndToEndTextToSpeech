# EndToEndTextToSpeech
Combine Tacotron2 and Hifi GAN to generate speech from text 

## Download weights
- [Hifi GAN](https://drive.google.com/file/d/1uSqIE8zTrDigTUb0dbDKaXuFG0viL5XY/view?usp=sharing) -> hifi_gan/checkpoint/ : pretrain 2.5M step, traning 350K & 550K step
- [Tacotron2](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view) -> tacotron2_mini/checkpoint/

## Install
- `pip install -r requirements.txt`

## Run
- `python server.py`
- App run on localhost:5000
