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
import json
from pvrecorder import PvRecorder

import keyboard
from duckduckgo_search import DDGS
from call_func import os_music, duckduck_search
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Парсинг аргументов
arg_parser = argparse.ArgumentParser(description = "Process some parameters.")
arg_parser.add_argument('--whisper_size', default = 'small')
arg_parser.add_argument('--tts_model', default = 'utrobinmv/tts_ru_free_hf_vits_high_multispeaker')
arg_parser.add_argument('--picovoice_key', default = 'kszOncmfUz7CnIwqlzn/PcoRmhBSovtJtm/u7OaUyu9OO6784lM/9Q==')
arg_parser.add_argument('--keyword', default = 'bumblebee')
arg_parser.add_argument('--port', default='1234')
arg_parser.add_argument('--llm_name', default='llama-3.2-3b-instruct')
args = arg_parser.parse_args()

# HOTWORD, VOICE DETECT
porcupine = pvporcupine.create(access_key = args.picovoice_key, keywords=[args.keyword])
recoder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
cobra = pvcobra.create(access_key = args.picovoice_key)
# STT WHISPER
model_stt = whisper.load_model(args.whisper_size, device=device)
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
    """audioframe с микрофона"""
    audio_frame = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    return audio_frame

def detect_phrase():
    """Детекция горячего слова. Определяется в --keyword.
    ['alexa','americano','blueberry','bumblebee','computer','grapefruit','grasshopper','hey barista','hey google',
    'hey siri','jarvis','ok google','pico clock','picovoice','porcupine','terminator']
    """
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
        instruct = """
            Ты - голосовой ассистент, который управляет функциями через JSON-ответы.
            Действуй по схеме: распознай команду -> найди соответствующую функцию -> верни код.
            Доступные функции:
            1. os_music (управление музыкой):
                - code 1 = предыдущий трек
                - code 2 = следующий трек
                - code 3 = пауза/возобновление
                - code 4 = увеличить громкость
                - code 5 = уменьшить громкость
            2. duckduck_search (запрос в поисковик):
                - text = запрос пользователя
            Правила:
            - Отвечай только в формате JSON: {"func": "название", "args":{"аргумент": "значение"}}
            Примеры:
            Запрос: сделай музыку тише -> {"func":"os_music", "args":{"code":5}}
            Запрос: поставь музыку на паузу -> {"func":"os_music", "args":{"code":3}}
            Запрос: посмотри кто такой пушкин -> {"func":"duckduck_search", "args":{"text":"Кто такой пушкин?"}}
            Запрос: поищи что такое интеграл -> {"func":"duckduck_search", "args":{"text":"Что такое интеграл?"}}
            Запрос: найди мне кто был президентом в России в 2007 -> {"func":"duckduck_search", "args":{"text":"Кто был президентом России в 2007 году?"}}
            """
        text = {'model':args.llm_name, 
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
            globals()[llm_resp['func']](**llm_resp['args'])
        except:
            print(f'ASSISTENT: {llm_resp}') #log
            # Озвучивание ответа llm
            TTS(f'     {llm_resp}     ', device_audio)
        torch.cuda.empty_cache()
