import fulakar_agent
import call_func
import argparse
from colorama import Fore, init as colorama_init
colorama_init()


# Парсинг аргументов
arg_parser = argparse.ArgumentParser(description = "Process some parameters.")
arg_parser.add_argument('--whisper_size', default = 'small')
arg_parser.add_argument('--tts_model', default = 'utrobinmv/tts_ru_free_hf_vits_high_multispeaker')
arg_parser.add_argument('--picovoice_key', default = 'kszOncmfUz7CnIwqlzn/PcoRmhBSovtJtm/u7OaUyu9OO6784lM/9Q==')
arg_parser.add_argument('--keyword', default = 'bumblebee')
arg_parser.add_argument('--port', default='1234')
arg_parser.add_argument('--host', default='localhost')
arg_parser.add_argument('--llm_name', default='llama-3.2-3b-instruct')
args = arg_parser.parse_args()

url = f'http://{args.host}:{args.port}/v1/chat/completions'

SAMPLE_RATE = 512 * 32
CHANNELS = 1 
DURATION_FRAME_HOTWORD = 0.5
DURATION_FRAME_VOID = 1

agent = fulakar_agent.Agent(args.picovoice_key, args.keyword, 
                            args.whisper_size, 
                            args.tts_model, 
                            args.llm_name, llm_api_url=url,
                            SAMPLE_RATE=SAMPLE_RATE, CHANNELS=CHANNELS, DURATION_FRAME_HOTWORD=DURATION_FRAME_HOTWORD, DURATION_FRAME_VOID=DURATION_FRAME_VOID)

print(Fore.RED + 'Агент создан')

agent.start()
