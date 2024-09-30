import datetime
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models['chat'] = AutoModel.from_pretrained(
        "THUDM/chatglm-6b", 
        trust_remote_code=True).half().cuda()
    models['chat'].eval()
    models['tokenizer'] = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", 
        trust_remote_code=True)
    yield
    for model in models.values():
        del model
        torch_gc()

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

class Item(BaseModel):
    prompt: str = "你好"
    history: list[tuple[str, str]] = []
    max_length: int = 2048
    top_p: float = 0.7
    temperature: float = 0.95 

class Answer(BaseModel):
    response: str
    history: list[tuple[str, str]]
    status: int
    time: str

@app.post("/")
async def create_item(item: Item):
    response, history = models['chat'].chat(
        models['tokenizer'],
        item.prompt,
        history=item.history,
        max_length=item.max_length,
        top_p=item.top_p,
        temperature=item.temperature)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{time}] prompt: '{item.prompt}', response: '{response}'")
    return Answer(response=response, history=history, status=200, time=time)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
