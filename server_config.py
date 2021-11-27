device = 'cpu'

audio = {
    'max_wave_value': 32768.0,
    'sampling_rate': 22050 # JLSpeech
}

INPUT_FIELDS = {
    'pv1': {
        'id': 'pv1',
        'label': 'Config v1 : pretrain 2.5M step',
    },
    'pv2': {
        'id': 'pv2',
        'label': 'Config v2 : pretrain 2.5M step',
    },
    'tv1': {
        'id': 'tv1',
        'label': 'Config v1 : traning 350K step',
    },
    'tv2': {
        'id': 'tv2',
        'label': 'Config v2 : traning 550K step',
    },
}

wave_storage = './static'

tacotron2_checkpoint = 'tacotron2_mini/checkpoint/tacotron2_statedict.pt'

hifi_pretrain_v1 = {
    'config': 'hifi_gan/checkpoint/pretrain_v1/config.json',
    'checkpoint': 'hifi_gan/checkpoint/pretrain_v1/generator_v1.pt'
}
hifi_pretrain_v2 = {
    'config': 'hifi_gan/checkpoint/pretrain_v2/config.json',
    'checkpoint': 'hifi_gan/checkpoint/pretrain_v2/generator_v2.pt'
}

hifi_fine_tune_v1 = {
    'config': 'hifi_gan/checkpoint/fine_tune_v1/config.json',
    'checkpoint': 'hifi_gan/checkpoint/fine_tune_v1/g_00350000.pt'
}
hifi_fine_tune_v2 = {
    'config': 'hifi_gan/checkpoint/fine_tune_v2/config.json',
    'checkpoint': 'hifi_gan/checkpoint/fine_tune_v2/g_00550000.pt'
}

