from typing import Union, Dict, List, Any,Optional

import argparse
import json
import logging

import fastapi
import httpx
import uvicorn


from pydantic import BaseModel

class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    
    def __init__(self,system, roles, messages):
        self.system = system
        self.roles = roles
        self.messages = messages
    
conv = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. ",
    roles=(["USER", "Assistant"]),
    prompt=(),
)


app = fastapi.FastAPI()

headers = {"User-Agent": "FastChat API Server"}



class Request(BaseModel):
    model: str
    prompt: List[Dict[str, str]]
    top_p: Optional[float] =0.7
    temperature: Optional[float] = 0.7
    

@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    """Creates a completion for the chat message"""
    conv.prompt =[]
    payload = generate_payload(request.prompt)    
    content = await invoke_example(request.model, payload)
    
    generate_payload(content)
        
        
    print('a',content)
    return content[0]['content']
    


import zhipuai
 
# your api key
zhipuai.api_key = "you_api_key"

async def invoke_example(model,prompt):

    response = zhipuai.model_api.invoke(
        model= model,
        prompt=prompt,
        top_p=0.7,
        temperature=0.7,
    )
    
    return response['data']['choices']



def generate_payload(messages: List[Dict[str, str]]):

    conv.prompt = list(conv.prompt)
    for message in prompt:
       
        msg_role = message["role"]
        
        if msg_role == "user":
            conv.messages.append({'role': conv.roles[0], 'content': message["content"]})
        elif msg_role == "assistant":
            conv.messages.append({'role': conv.roles[1], 'content': message["content"]})
        else:
            raise ValueError(f"Unknown role: {msg_role}")
    
    return conv.messages
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGLM-compatible Restful API server.")
    parser.add_argument("--host", type=str, default="10.109.116.3", help="host name")
    parser.add_argument("--port", type=int, default=6000, help="port number")
 
    args = parser.parse_args()
    uvicorn.run("chatglm-api:app", host=args.host, port=args.port, reload=False)
    
    
    
    
    
    