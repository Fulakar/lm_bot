import fulakar_agent
from dotenv import load_dotenv
import os
from colorama import Fore, init as colorama_init
colorama_init()
from call_func import TOOLS

# Парсинг аргументов
load_dotenv()
WHISPER_SIZE = os.getenv("WHISPER_SIZE")
TTS_MODEL = os.getenv("TTS_MODEL")
PICOVOICE_KEY = os.getenv("PICOVOICE_KEY")
KEYWORD = os.getenv("KEYWORD")
PORT = os.getenv("PORT")
HOST = os.getenv("HOST")
LLM_NAME = os.getenv("LLM_NAME")
API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
ALLOWED_DIALOG = os.getenv("ALLOWED_DIALOG")
ALLOWED_DIALOG = [name.strip() for name in ALLOWED_DIALOG.split(";")]

LLM_API_URL = f'http://{HOST}:{PORT}/v1'

SAMPLE_RATE = 512 * 32
CHANNELS = 1 
DURATION_FRAME_HOTWORD = 0.5
DURATION_FRAME_VOID = 1

agent = fulakar_agent.Agent(PICOVOICE_KEY, KEYWORD, 
                            WHISPER_SIZE, 
                            TTS_MODEL, 
                            LLM_NAME, llm_api_url=LLM_API_URL, tools=TOOLS,
                            SAMPLE_RATE=SAMPLE_RATE, CHANNELS=CHANNELS, DURATION_FRAME_HOTWORD=DURATION_FRAME_HOTWORD, DURATION_FRAME_VOID=DURATION_FRAME_VOID)

print(Fore.RED + 'Агент создан')

agent.start()
