from langchain.agents import AgentType, Tool, load_tools, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

import sounddevice as sd
import pvporcupine
import pvcobra
import numpy as np
from transformers import VitsModel, AutoTokenizer
from faster_whisper import WhisperModel
import torch
from pvrecorder import PvRecorder
from typing import List
from colorama import Fore, init as colorama_init
colorama_init()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent():
    def __init__(self, picovoice_key:str, keyword:List[str], whisper_size:str, tts_model:str,
                 llm_name, llm_api_url, tools:List[Tool],
                SAMPLE_RATE = 512 * 32,
                CHANNELS = 1,
                DURATION_FRAME_HOTWORD = 0.5,
                DURATION_FRAME_VOID = 1):
        self.TOOLS = tools
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
        self.model_stt = WhisperModel(whisper_size, 
                                      device=device, 
                                      compute_type="int8_float16" if device == "cuda" else "float32")
        # TTS MODEL
        self.model_tts = VitsModel.from_pretrained(tts_model).to(device).eval()
        self.tokenizer_tts = AutoTokenizer.from_pretrained(tts_model)
        # DEFAULT AUDIO DEVICE
        self.device_audio = self.get_default_audio_device()
        # AGENT
        self.agent = self.get_agent()
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
            
    def STT(self, audio:np.array):
        audio = audio / 2 ** 15
        segments, _ = self.model_stt.transcribe(audio, language="ru")
        text = ""
        for segment in segments:
            text += segment.text
        return text

    def TTS(self, text:str):
        text = text.lower()
        inputs = self.tokenizer_tts(text, return_tensors="pt")
        speaker = 1 # 0-woman, 1-man 
        with torch.no_grad():
            output = self.model_tts(**inputs.to(device), speaker_id=speaker).waveform
        output = output.cpu().numpy()
        sd.play(output.reshape(-1), samplerate=self.model_tts.config.sampling_rate, device=self.device_audio)
        sd.wait()

    def get_agent(self):
        # LLM
        self.llm = OpenAI(
            model_name=self.llm_name,
            openai_api_base=self.llm_api_url,
            temperature=0.5,
            max_tokens=1000,
            model_kwargs={"model": self.llm_name},
            openai_api_key="any_key"
        )
        self.system_prompt = PromptTemplate.from_template(
            template="""
        Ты ассистент, отвечающий только на текущий запрос пользователя. Игнорируй любые предыдущие вопросы, сообщения или контекст, если они не указаны явно. 
        Отвечай строго на русском языке.
        """
        )
        # TOOLS
        llm_math = load_tools(["llm-math"], self.llm)[0]
        llm_math.description = "Полезно, когда вам нужно ответить на вопросы по математике"
        llm_math.name = "Калькулятор"
        self.TOOLS.append(llm_math)
        # AGENT
        agent = initialize_agent(
            tools=self.TOOLS, 
            llm=self.llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=True,
            system_prompt=self.system_prompt
        )
        return agent

    def start(self):
        while True:
            # Детекция ключевого слова
            if self.detect_phrase():
                print(Fore.BLUE + 'Слушаю')
                detect_speech_ = True 
                VAW = np.array([])
                # Запись фразы до момента тишины, запись в вафку
                while detect_speech_:
                    audio_frame = self.get_next_audio_frame(self.DURATION_FRAME_VOID).reshape(-1, 512)
                    detect_speech_ = self.detect_speech(audio_frame)
                    VAW = np.append(VAW, audio_frame)
                print(Fore.BLUE + 'Понял ', f'Секунд голоса: {VAW.shape[0] / 512 / 32}') #log
                # Audio-to-text
                text = self.STT(VAW)
                print(Fore.GREEN + f"USER: {text}") #log
                # Запрос в llm
                agent_resp = self.agent.run(text)
                print(Fore.BLUE + f"AGENT: {agent_resp}") #log
                torch.cuda.empty_cache()
