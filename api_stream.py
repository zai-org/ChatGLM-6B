from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn
import json
import datetime
import torch
import threading

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

stream_buffer = {}


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


def stream_item(prompt, history, max_length, top_p, temperature):
    global model, tokenizer
    global stream_buffer
    for response, history in model.stream_chat(tokenizer, prompt, history=history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        query, response = history[-1]
        now = datetime.datetime.now()
        stream_buffer[prompt] = {
            "response": response, "stop": False, "history": history,"time": now}
    stream_buffer[prompt]["stop"] = True
    torch_gc()


def removeTimeoutBuffer():
    global stream_buffer
    for key in stream_buffer.copy():
        diff = datetime.datetime.now() - stream_buffer[key]["time"]
        seconds = diff.total_seconds()
        print(key + ": 已存在" + str(seconds) + "秒")
        if seconds > 120:            
            if stream_buffer[key]["stop"]:
                del stream_buffer[key]
                print(key + "：已被从缓存中移除")
            else:
                stream_buffer[key]["stop"] = True
                print(key + "：已被标识为结束")


@app.post("/stream")
async def create_item(request: Request):
    # 删除过期的buffer
    removeTimeoutBuffer()
    # 全局变量buffer
    global stream_buffer
    # 获取入参
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    # 判断是否已在生成，只有首次才调stream_chat
    now = datetime.datetime.now()
    if stream_buffer.get(prompt) is None:
        stream_buffer[prompt] = {"response": "", "stop": False, "history": [],"time": now}
        # 在线程中调用stream_chat
        sub_thread = threading.Thread(target=stream_item, args=(prompt, history, max_length if max_length else 2048,
                                                                top_p if top_p else 0.7, temperature if temperature else 0.95))
        sub_thread.start()
    # 异步返回response
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    response = stream_buffer[prompt]["response"]
    history = stream_buffer[prompt]["history"]
    # 如果stream_chat调用完成，给返回加一个停止词[stop]
    if stream_buffer[prompt]["stop"]:
        response = response + '[stop]'
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + \
        prompt + '", response:"' + repr(response) + '"'
    print(log)

    return answer


if __name__ == '__main__':
    #tokenizer = AutoTokenizer.from_pretrained(
    #    "THUDM/chatglm-6b", trust_remote_code=True)
    #model = AutoModel.from_pretrained(
    #    "THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    # mkdir model
    # cp ~/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/658202d88ac4bb782b99e99ac3adff58b4d0b813 ./model
    model_path = "./model/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
