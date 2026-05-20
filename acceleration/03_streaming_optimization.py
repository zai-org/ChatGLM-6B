# ChatGLM3-6B 流式输出
# 很多人觉得慢其实是因为等全部生成完才能看到结果，流式输出能大幅改善体感
# 这里提供 CLI / FastAPI SSE / Gradio 三种方式

import os
import time
import json
import queue
import threading
import torch
from typing import Optional, Generator, List, Tuple
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
    return model, tokenizer


def cli_streaming_chat(model, tokenizer):
    """命令行流式对话，比原版 cli_demo 多了首字延迟统计"""
    print("ChatGLM3-6B 流式对话 (stop退出)")
    history = []
    past_key_values = None

    while True:
        query = input("\n用户: ").strip()
        if query == "stop":
            break
        if query == "clear":
            history, past_key_values = [], None
            print("已清空")
            continue
        if not query:
            continue

        print("ChatGLM: ", end="", flush=True)
        t0 = time.perf_counter()
        first = True
        pos = 0

        for response, history, past_key_values in model.stream_chat(
            tokenizer, query, history=history,
            past_key_values=past_key_values,
            return_past_key_values=True,
            max_length=2048, top_p=0.8, temperature=0.8,
        ):
            delta = response[pos:]
            if delta:
                if first:
                    ttft = time.perf_counter() - t0
                    first = False
                print(delta, end="", flush=True)
                pos = len(response)

        elapsed = time.perf_counter() - t0
        print(f"\n[首字: {ttft:.2f}s | 总: {elapsed:.2f}s | {pos/elapsed:.1f} chars/s]")


class StreamingGenerator:
    """简单封装，可以直接给 FastAPI StreamingResponse 用"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_stream(self, query, history=None, max_length=2048,
                        top_p=0.8, temperature=0.8):
        if history is None:
            history = []
        pos = 0
        for response, _, _ in self.model.stream_chat(
            self.tokenizer, query, history=history,
            max_length=max_length, top_p=top_p, temperature=temperature,
            past_key_values=None, return_past_key_values=True,
        ):
            delta = response[pos:]
            if delta:
                yield delta
                pos = len(response)


class AsyncStreamingGenerator:
    """把生成放到单独线程里跑，避免阻塞主线程"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _worker(self, q, query, history, **kwargs):
        try:
            pos = 0
            for response, _, _ in self.model.stream_chat(
                self.tokenizer, query, history=history, **kwargs
            ):
                delta = response[pos:]
                if delta:
                    q.put(("token", delta))
                    pos = len(response)
            q.put(("done", None))
        except Exception as e:
            q.put(("error", str(e)))

    def generate_async(self, query, history=None, **kwargs):
        q = queue.Queue()
        t = threading.Thread(target=self._worker, args=(q, query, history),
                             kwargs=kwargs, daemon=True)
        t.start()
        while True:
            typ, data = q.get()
            if typ == "token":
                yield data
            elif typ == "done":
                break
            elif typ == "error":
                raise RuntimeError(f"生成出错: {data}")


def create_streaming_api_app(model, tokenizer):
    """FastAPI 流式 API，SSE 格式输出"""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                      allow_methods=["*"], allow_headers=["*"])

    gen = AsyncStreamingGenerator(model, tokenizer)

    class Req(BaseModel):
        query: str
        history: list = []
        max_length: int = 2048
        top_p: float = 0.8
        temperature: float = 0.8

    @app.post("/chat/stream")
    async def chat_stream(req: Req):
        def events():
            for tok in gen.generate_async(req.query, req.history,
                                          max_length=req.max_length,
                                          top_p=req.top_p, temperature=req.temperature):
                yield f"data: {json.dumps({'token': tok}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/chat")
    async def chat(req: Req):
        text = ""
        for tok in gen.generate_async(req.query, req.history,
                                      max_length=req.max_length):
            text += tok
        return {"response": text}

    return app


def create_gradio_app(model, tokenizer):
    """Gradio 流式 Web 界面"""
    import gradio as gr

    def predict(message, history):
        chat_hist = [(m["content"], r["content"]) for m, r in history] if history else []
        pos = 0
        text = ""
        for response, chat_hist, _ in model.stream_chat(
            tokenizer, message, history=chat_hist,
            max_length=2048, top_p=0.8, temperature=0.8,
        ):
            text = response
            pos = len(response)
            yield text

    return gr.ChatInterface(fn=predict, title="ChatGLM3-6B",
                            retry_btn="重试", undo_btn="撤销", clear_btn="清空")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="cli", choices=["cli", "api", "gradio"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.mode == "cli":
        cli_streaming_chat(model, tokenizer)
    elif args.mode == "api":
        import uvicorn
        app = create_streaming_api_app(model, tokenizer)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.mode == "gradio":
        demo = create_gradio_app(model, tokenizer)
        demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
