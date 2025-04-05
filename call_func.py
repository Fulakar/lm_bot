import keyboard
from duckduckgo_search import DDGS

# Вызывает модель
def os_music(code):
    cmd = {1:"previous track",
       2:"next track", 
       3:"play/pause", 
       4:"volume up", 
       5:"volume down"}
    keyboard.send(cmd[code])

def duckduck_search(user_request):
    results = DDGS().text(user_request, max_results=5)
    results = " | ".join([row["body"] for row in results])

    instruct = "Ты специалист по суммаризации текстов. Пользователь тебе дает текст, ты выдаешь его суммаризацию."

    text = {'model':args.llm, 
        'messages':[
            {'role':'system', 'content':instruct},
            {'role':'user', 'content':f'Запрос пользователя: {user_request}. Текст для суммаризации: {results}'}
            ],
        "temperature": 0.1, 
        "max_tokens": -1,
        "stream": False}
    resp = requests.post(url, json=text)
    resp = json.loads(resp.text)['choices'][0]['message']['content']
    return resp
    return results