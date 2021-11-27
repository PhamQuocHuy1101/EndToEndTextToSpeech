import os
import random

from flask import Flask, render_template, request, redirect
app = Flask(__name__)

def clear_storage():
    for f in os.listdir(cf.wave_storage):
        os.remove(os.path.join(cf.wave_storage, f))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', input_fields=cf.INPUT_FIELDS.values())

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        return redirect('/')
    
    clear_storage()
    form = request.form
    text = form.get('text' ,'').strip()

    audio = []
    if text != '':
        threshes = []
        for k in MODELS.keys():
            if form.get(k, None) != None:
                file_name = os.path.join(cf.wave_storage, str(random.random()) + '.wav')
                MODELS[k].inference(text, file_name)
                audio.append({
                    'label': cf.INPUT_FIELDS[k]['label'],
                    'src': file_name
                })
    
    return render_template('index.html', 
                            input_fields = cf.INPUT_FIELDS.values(), 
                            text = text, audio = audio)

if __name__ == '__main__':
    import tacotron2_inference as ti
    import hifi_gan_inference as hi
    from inference_e2e import EndToEndTTS
    import server_config as cf

    tacotron2_model = ti.load_model(cf.tacotron2_checkpoint, cf.device)
    hifi_pre_v1 = hi.load_model(cf.hifi_pretrain_v1['config'],
                                cf.hifi_pretrain_v1['checkpoint'],
                                cf.device)
    hifi_pre_v2 = hi.load_model(cf.hifi_pretrain_v2['config'],
                                cf.hifi_pretrain_v2['checkpoint'],
                                cf.device)
    hifi_ft_v1 = hi.load_model(cf.hifi_fine_tune_v1['config'],
                                cf.hifi_fine_tune_v1['checkpoint'],
                                cf.device)
    hifi_ft_v2 = hi.load_model(cf.hifi_fine_tune_v2['config'],
                                cf.hifi_fine_tune_v2['checkpoint'],
                                cf.device)

    pretrain_v1 = EndToEndTTS(tacotron2_model, hifi_pre_v1, cf.audio, cf.device)
    pretrain_v2 = EndToEndTTS(tacotron2_model, hifi_pre_v2, cf.audio, cf.device)
    fine_tune_v1 = EndToEndTTS(tacotron2_model, hifi_ft_v1, cf.audio, cf.device)
    fine_tune_v2 = EndToEndTTS(tacotron2_model, hifi_ft_v2, cf.audio, cf.device)

    MODELS = {
        'pv1': pretrain_v1,
        'pv2': pretrain_v2,
        'tv1': fine_tune_v1,
        'tv2': fine_tune_v2
    }

    app.run(host='0.0.0.0', port=5000)