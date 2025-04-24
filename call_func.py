from langchain.agents import Tool
import keyboard
from duckduckgo_search import DDGS
from telethon import TelegramClient
from telethon.tl.types import User
import asyncio
from dotenv import load_dotenv
import os
import re
load_dotenv()
API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
ALLOWED_DIALOG = os.getenv("ALLOWED_DIALOG")
ALLOWED_DIALOG = [name.strip() for name in ALLOWED_DIALOG.split(";")]

def os_music(code):
    code = int(code)
    cmd = {1:"previous track",
       2:"next track", 
       3:"play/pause", 
       4:"volume up", 
       5:"volume down"}

    keyboard.send(code)
    return f"Команда '{cmd[code]}' выполнена."

def duckduck_search(text, max_results:int=5):
    results_from_engine = DDGS().text(text, max_results=max_results)
    results_from_engine = " | ".join([row["body"] for row in results_from_engine])
    return results_from_engine

def send_message(input_:str):
    dialog_name, message = input_.split("|")
    dialog_name, message = dialog_name.strip(), message.strip()
    async def basic_func(dialog_name: str, message: str):

        async with TelegramClient("TG_SESSION", API_ID, API_HASH, system_version="4.16.30-vxCUSTOM") as client:
            target_username = None
            async for dialog in client.iter_dialogs():
                if re.sub(r"[^a-zA-Z0-9а-яА-Я]", "", dialog.name) == dialog_name and isinstance(dialog.entity, User):
                # if dialog.name == dialog_name and isinstance(dialog.entity, User):
                    target_username = dialog.entity.username
                    await client.send_message(target_username, message)
                    return f"Пользователю @{target_username} отправлено сообщение : {message}. "
            # Если пользователь не найден
            if target_username is None:
                return (f"Диалог с именем '{dialog_name}' не найден")
        
    resp = asyncio.run(basic_func(dialog_name, message))
    return resp

def get_unread_dialogs(_):
        
    async def basic_func():
        """Получение информации о непрочитанных диалогах"""
        async with TelegramClient("TG_SESSION", API_ID, API_HASH, system_version="4.16.30-vxCUSTOM") as client:
            row = ""
            async for dialog in client.iter_dialogs():
                if isinstance(dialog.entity, User) and dialog.unread_count > 0:
                    unread = dialog.unread_count
                    row += f"{dialog.name}: {unread} непрочитанных сообщений. "
            if row == "":
                return "Нет непрочитанных сообщений"
            else:
                return row
    
    resp = asyncio.run(basic_func())

    return resp 

TOOLS = [
    Tool(
        name="Поиск в интернете",
        func=duckduck_search,
        description="Используй для поиска информации в интернете. Используй только если пользователь явно просит что-то найти. Ввод: строка с запросом."
    ),
    Tool(
        name="Непрочитанные сообщения",
        func=get_unread_dialogs,
        description="Вызывай функцию если трубется получить список непрочитанных сообщений. Ввод: None"
    ),
    Tool(
        name="Отправка сообщения",
        func=send_message,
        description=f"""f"Вызывай функцию если нужно отправить сообщение пользователю. Список доступных пользователей : {ALLOWED_DIALOG}.
        Вводить имя надо точно так же как оно указано в списке доступных пользователей.
        Если пользователя нету в списке доступных, напиши что пользователь не доступен и функцию вызывать не надо.
        Ввод: имя_пользователя | текст."""
    ),
    Tool(
        name="Регулировка музыки",
        func=os_music,
        description=f"""
        Вызывай когда необходимо регуировка музыки:
        code (число):
            1 - "Предыдущий трек"
            2 - "Следующий трек"
            3 - "Пазуа/Воспроизведение"
            4 - "Повышение громкости"
            5 - "Понижение громкости"
        Ввод: code
"""
    )
]
