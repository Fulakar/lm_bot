import sounddevice as sd
import pvporcupine
import pvcobra
import numpy as np
import soundfile as sf
from torchaudio import load as torch_load
from transformers import VitsModel, AutoTokenizer
import whisper
import torch
from pvrecorder import PvRecorder
from typing import List
from colorama import Fore, init as colorama_init
colorama_init()
# CALL FUNC
import keyboard
from duckduckgo_search import DDGS
import json
from call_func import os_music, duckduck_search, call_func_with_llm_resp
# UTILS
import requests
from utils import llm_request

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent():
    def __init__(self, picovoice_key:str, keyword:List[str], whisper_size:str, tts_model:str,
                 llm_name, llm_api_url,
                SAMPLE_RATE = 512 * 32,
                CHANNELS = 1,
                DURATION_FRAME_HOTWORD = 0.5,
                DURATION_FRAME_VOID = 1):
        self.SAMPLE_RATE = SAMPLE_RATE
        self.CHANNELS = CHANNELS
        self.DURATION_FRAME_HOTWORD = DURATION_FRAME_HOTWORD
        self.DURATION_FRAME_VOID = DURATION_FRAME_VOID
        self.llm_name = llm_name 
        self.llm_api_url = llm_api_url
        # HOTWORD, VOICE DETECT
        self.porcupine = pvporcupine.create(access_key=picovoice_key, keywords=[keyword])
        self.recoder = PvRecorder(device_index=-1, frame_length=self.porcupine.frame_length)
        self.cobra = pvcobra.create(access_key=picovoice_key)
        # STT WHISPER
        self.model_stt = whisper.load_model(whisper_size, device=device)
        # TTS MODEL
        self.model_tts = VitsModel.from_pretrained(tts_model).to(device).eval()
        self.tokenizer_tts = AutoTokenizer.from_pretrained(tts_model)
        # DEFAULT AUDIO DEVICE
        self.device_audio = self.get_default_audio_device()

    def get_default_audio_device(self):
        """Возвращает индекс устройства вывода по умолчанию."""
        default_device = sd.default.device['output']
        devices = sd.query_devices()
        return next((i for i, d in enumerate(devices) if i == default_device), None)

    def get_next_audio_frame(self, DURATION):
        """audioframe с микрофона"""
        audio_frame = sd.rec(int(self.SAMPLE_RATE * DURATION), samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype='int16')
        sd.wait()
        return audio_frame

    def detect_phrase(self):
        """Детекция горячего слова. Определяется в --keyword.
        ['alexa','americano','blueberry','bumblebee','computer','grapefruit','grasshopper','hey barista','hey google',
        'hey siri','jarvis','ok google','pico clock','picovoice','porcupine','terminator']
        """
        try:
            self.recoder.start()
            while True:
                keyword_index = self.porcupine.process(self.recoder.read())
                if keyword_index >= 0:
                    return True
        except Exception as e:
            print(e)
            return False
        
    def detect_speech(self, audio_frame):
        for sample in range(len(audio_frame)):
            if self.cobra.process(audio_frame[sample]) > 0.95:
                return True
            
    def STT(self, audio):
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).type(torch.float16).to(self.model_stt.device)
        options = whisper.DecodingOptions(language='ru')
        result = whisper.decode(self.model_stt, mel, options)
        result_text = result[0].text
        return result_text

    def TTS(self, text):
        text = text.lower()
        inputs = self.tokenizer_tts(text, return_tensors="pt")
        speaker = 1 # 0-woman, 1-man 
        with torch.no_grad():
            output = self.model_tts(**inputs.to(device), speaker_id=speaker).waveform
        output = output.cpu().numpy()
        sd.play(output.reshape(-1), samplerate=self.model_tts.config.sampling_rate, device=self.device_audio)
        sd.wait()

    def text_for_llm(self, user_request):
        instruct = """
Ты - голосовой ассистент. Возвращай ответы ТОЛЬКО в формате JSON.


Доступные функции:
1. os_music (управление музыкой)
Аргументы:
- code (число): 
    1 = предыдущий трек
    2 = следующий трек
    3 = пауза/воспроизведение
    4 = прибавить громкость
    5 = убавить громкость

2. duckduck_search (поиск в интернете)
   Аргументы:
   - text (строка): запрос пользователя    

   
Формат ответа:
{"func": "название функции","args": {"аргумент": значение}}


Примеры:
Запрос пользователя: сделай музыку тише
Твой ответ: {"func":"os_music", "args":{"code":5}}

Запрос пользователя: поставь музыку на паузу
Твой ответ: {"func":"os_music", "args":{"code":3}}

Запрос пользователя: посмотри кто такой пушкин
Твой ответ: {"func":"duckduck_search", "args":{"text":"Кто такой пушкин?"}}

Запрос пользователя: поищи что такое интеграл
Твой ответ: {"func":"duckduck_search", "args":{"text":"Что такое интеграл?"}}

Запрос пользователя: найди мне кто был президентом в России в 2007
Твой ответ: {"func":"duckduck_search", "args":{"text":"Кто был президентом России в 2007 году?"}}
"""
        resp = llm_request(user_request, instruct, self.llm_name, self.llm_api_url)
        return resp

    def start(self):
        while True:
            # Детекция ключевого слова
            if self.detect_phrase():
                print(Fore.BLUE + 'Слушаю') #log
                detect_speech_ = True 
                VAW = np.array([])
                # Запись фразы до момента тишины, запись в вафку
                while detect_speech_:
                    audio_frame = self.get_next_audio_frame(self.DURATION_FRAME_VOID).reshape(-1, 512)
                    detect_speech_ = self.detect_speech(audio_frame)
                    VAW = np.append(VAW, audio_frame)
                print(Fore.BLUE + 'Понял ', f'Секунд голоса: {VAW.shape[0] / 512 / 32}') #log
                sf.write('for_llm.WAV', VAW.reshape(-1).astype(np.float32) / 32768.0, samplerate=self.SAMPLE_RATE)
                # Чтение вафки и перевод в текст
                VAW, _ = torch_load('for_llm.WAV')
                text = self.STT(VAW)
                print(Fore.GREEN + f"USER: {text}") #log
                # Запрос в llm
                llm_resp = self.text_for_llm(text)
                try:
                    func_return = call_func_with_llm_resp(llm_resp, self.llm_name, self.llm_api_url)
                    if func_return is not None:
                        self.TTS(func_return)
                except Exception as e:
                    print(Fore.BLUE + f'AGENT ANSWER: {llm_resp}') #log
                    print(e)
                torch.cuda.empty_cache()
