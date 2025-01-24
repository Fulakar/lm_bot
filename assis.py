import sounddevice as sd
import pvporcupine
import pvcobra
import numpy as np
import soundfile as sf
from torchaudio import load as torch_load
from transformers import VitsModel, AutoTokenizer
import whisper
import requests
import json
import torch
import argparse
import keyboard
import json
from pvrecorder import PvRecorder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Аргументы
arg_parser = argparse.ArgumentParser(description = "Process some parameters.")
arg_parser.add_argument('--stt_model', default = 'small')
arg_parser.add_argument('--tts_model', default = 'utrobinmv/tts_ru_free_hf_vits_high_multispeaker')
arg_parser.add_argument('--picovoice_key', default = 'kszOncmfUz7CnIwqlzn/PcoRmhBSovtJtm/u7OaUyu9OO6784lM/9Q==')
arg_parser.add_argument('--keyword', default = 'bumblebee')
arg_parser.add_argument('--port', default='1234')
arg_parser.add_argument('--llm', default='llama-3.2-3b-instruct')
args = arg_parser.parse_args()




# HOTWORD, VOICE DETECT
porcupine = pvporcupine.create(access_key = args.picovoice_key, keywords=[args.keyword])
recoder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
cobra = pvcobra.create(access_key = args.picovoice_key)
# STT WHISPER
model_stt = whisper.load_model(args.stt_model, device=device)
# TTS MODEL
model_tts = VitsModel.from_pretrained(args.tts_model)
tokenizer_tts = AutoTokenizer.from_pretrained(args.tts_model)
model_tts = model_tts.to(device)
model_tts.eval()

url = f'http://localhost:{args.port}/v1/chat/completions'
head = {'Content-Type':'application/json'}

SAMPLE_RATE = 512 * 32
CHANNELS = 1 
DURATION_FRAME_HOTWORD = 0.5
DURATION_FRAME_VOID = 1

def get_default_audio_device():
    """Возвращает индекс устройства вывода по умолчанию."""
    default_device = sd.default.device['output']
    devices = sd.query_devices()
    return next((i for i, d in enumerate(devices) if i == default_device), None)

def get_next_audio_frame(DURATION):
    audio_frame = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    return audio_frame

def detect_phrase():
    try:
        recoder.start()
        while True:
            keyword_index = porcupine.process(recoder.read())
            if keyword_index >= 0:
                return True
    except Exception as e:
        print(e)
        return False
        
def detect_speech(audio_frame):
    for sample in range(len(audio_frame)):
        if cobra.process(audio_frame[sample]) > 0.95:
            return True
        
def STT(audio):
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).type(torch.float16).to(model_stt.device)
    options = whisper.DecodingOptions(language='ru')
    result = whisper.decode(model_stt, mel, options)
    result_text = result[0].text
    return result_text

def TTS(text, device_audio):
    text = text.lower()
    inputs = tokenizer_tts(text, return_tensors="pt")
    speaker = 1 # 0-woman, 1-man 
    with torch.no_grad():
        output = model_tts(**inputs.to(device), speaker_id=speaker).waveform
        output = output.detach().cpu().numpy()
    sd.play(output.reshape(-1), samplerate=model_tts.config.sampling_rate, device=device_audio)
    sd.wait()

def text_for_llm(req):
        instruct = """Ты голосовой ассистент вызывающий функции в python.
        Возвращай ответ в виде: os_music(code). Варианты для code: 1 - предыдущий трек, 2 - следующий трек, 3 - пауза или включение музыки, 4 - повышение громокости, 5 - понижение громкости.
        Пример:
        Запрос: сделай музыку тише
        Ответ: {"func":"os_music", "code":5}
        Запрос: поставь музыку на паузу
        Ответ: {"func":"os_music", "code":3}"""

        text = {'model':args.llm, 
        'messages':[
            {'role':'system', 'content':instruct},
            {'role':'user', 'content':f'{req}'}
            ],
        "temperature": 0.1, 
        "max_tokens": -1,
        "stream": False}
        resp = requests.post(url, json=text)
        resp = json.loads(resp.text)['choices'][0]['message']['content']
        return resp

def os_music(code):
    cmd = {1:"previous track",
       2:"next track", 
       3:"play/pause", 
       4:"volume up", 
       5:"volume down"}
    keyboard.send(cmd[code])

device_audio = get_default_audio_device()
TTS('     Готов     ', device_audio)
print('Готов')

while True:
    # Детекция ключевого слова
    if detect_phrase():
        print('Слушаю') #log
        detect_speech_ = True 
        VAW = np.array([])
        # Запись фразы до момента тишины, запись в вафку
        while detect_speech_:
            audio_frame = get_next_audio_frame(DURATION_FRAME_VOID).reshape(-1, 512)
            detect_speech_ = detect_speech(audio_frame)
            VAW = np.append(VAW, audio_frame)
        print('Понял ', f'Секунд голоса: {VAW.shape[0] / 512 / 32}') #log
        sf.write('for_llm.WAV', VAW.reshape(-1).astype(np.float32) / 32768.0, samplerate=SAMPLE_RATE)
        # Чтение вафки и перевод в текст
        VAW, _ = torch_load('for_llm.WAV')
        text = STT(VAW)
        print(f"USER: {text}") #log
        # Запрос в llm
        llm_resp = text_for_llm(text)
        try:
            llm_resp = json.loads(llm_resp)
            print(f'ASSISTENT: {llm_resp}')
            globals()[llm_resp['func']](llm_resp['code'])
        except:
            print(f'ASSISTENT: {llm_resp}') #log
            # Озвучивание ответа llm
            TTS(f'     {llm_resp}     ', device_audio)
        torch.cuda.empty_cache()
