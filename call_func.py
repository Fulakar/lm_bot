import keyboard
from duckduckgo_search import DDGS
import json
# UTILS
import requests
from utils import llm_request

from colorama import Fore, init as colorama_init
colorama_init()

# Вызывает модель
def os_music(code):
    cmd = {1:"previous track",
       2:"next track", 
       3:"play/pause", 
       4:"volume up", 
       5:"volume down"}
    keyboard.send(cmd[code])

def duckduck_search(text, llm_name, llm_api_url, max_results:int=5):
    results_from_engine = DDGS().text(text, max_results=max_results)
    results_from_engine = " | ".join([row["body"] for row in results_from_engine])

    instruct = """
    Ты специалист по суммаризации текстов. 
    Пользователь тебе дает текст, ты выдаешь его суммаризацию текста на основе запроса пользователя.
    """
    request = f'Запрос пользователя: {text}. Текст для суммаризации: {results_from_engine}'

    resp = llm_request(request, instruct, llm_name, llm_api_url)
    return resp

def call_func_with_llm_resp(llm_resp, llm_name, llm_api_url):
    llm_resp = llm_resp.replace(" ", "").strip("`")
    llm_resp = json.loads(llm_resp)
    print(Fore.BLUE + f'AGENT FUNCTION CALLABLE: {llm_resp}')
    if llm_resp["func"] == "duckduck_search":
        llm_resp["args"]["llm_name"] = llm_name
        llm_resp["args"]["llm_api_url"] = llm_api_url
    return globals()[llm_resp["func"]](**llm_resp["args"])
