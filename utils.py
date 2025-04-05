import json
import requests

def llm_request(user_request, instruct, llm_name, llm_api_url, temperature:float=0.1, max_tokens:int=-1, stream:bool=False):
    text = {'model':llm_name, 
        'messages':[
            {'role':'system', 'content':instruct},
            {'role':'user', 'content':user_request}
            ],
        "temperature": temperature, 
        "max_tokens": max_tokens,
        "stream": stream}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(llm_api_url, json=text, headers=headers)
    resp = json.loads(resp.text)['choices'][0]['message']['content']
    return resp
