# ChatGLM3-6B 低显存方案
# FP16 要 13GB 显存，很多人卡跑不动
# 这里提供自动 offload、INT4+offload、纯CPU多线程 三种方式

import argparse
import time
import torch
from transformers import AutoModel, AutoTokenizer


def print_mem(tag=""):
    """看一下内存和显存"""
    import psutil
    m = psutil.virtual_memory()
    s = f"[{tag}] RAM: {m.used/1024**3:.1f}/{m.total/1024**3:.1f}GB ({m.percent}%)"
    if torch.cuda.is_available():
        g = torch.cuda.memory_allocated() / 1024**3
        t = torch.cuda.get_device_properties(0).total_mem / 1024**3
        s += f" | GPU: {g:.1f}/{t:.1f}GB"
    print(s)


def load_auto_offload(model_path):
    """accelerate 自动分配，GPU放不下的层放CPU，适合4-8GB显存"""
    print(f"加载(auto offload): {model_path}")
    print_mem("加载前")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto",
        torch_dtype=torch.float16, offload_folder="offload",
        offload_state_dict=True,
    ).eval()

    print_mem("加载后")
    if hasattr(model, 'hf_device_map'):
        devs = set(str(v) for v in model.hf_device_map.values())
        print(f"  层分布: {', '.join(devs)}")
    return model, tokenizer


def load_int4_offload(model_path):
    """INT4量化后再offload，适合4GB显卡"""
    from transformers import BitsAndBytesConfig

    print(f"加载(INT4+offload): {model_path}")

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quantization=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        quantization_config=qconfig, device_map="auto",
        offload_folder="offload",
    ).eval()

    print_mem("加载后")
    return model, tokenizer


def load_cpu(model_path, threads=None):
    """纯CPU推理，关键是把 PyTorch 线程数设对"""
    import multiprocessing

    if threads is None:
        threads = multiprocessing.cpu_count()

    print(f"加载(CPU): {model_path}, {threads}线程")

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(min(4, threads))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        device_map="cpu", torch_dtype=torch.float32,
    ).eval()

    print_mem("加载后")
    return model, tokenizer


def chat(model, tokenizer, label=""):
    print(f"\nChatGLM3-6B ({label}) - stop退出")
    history = []
    past_kv = None

    while True:
        try:
            query = input("\n用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query == "stop":
            break
        if query == "clear":
            history, past_kv = [], None
            print("已清空")
            continue
        if not query:
            continue

        print("ChatGLM: ", end="", flush=True)
        t0 = time.perf_counter()
        first_t = None
        pos = 0

        try:
            for resp, history, past_kv in model.stream_chat(
                tokenizer, query, history=history, past_key_values=past_kv,
                return_past_key_values=True,
                max_length=2048, top_p=0.8, temperature=0.8,
            ):
                delta = resp[pos:]
                if delta:
                    if first_t is None:
                        first_t = time.perf_counter() - t0
                    print(delta, end="", flush=True)
                    pos = len(resp)
        except Exception:
            # offload模式可能不支持stream_chat
            resp, history = model.chat(tokenizer, query, history=history)
            print(resp)
            pos = len(resp)

        elapsed = time.perf_counter() - t0
        ft = f"首字:{first_t:.1f}s | " if first_t else ""
        print(f"\n[{ft}总:{elapsed:.1f}s | {pos/elapsed:.1f}c/s]")
        print_mem("推理后")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="THUDM/chatglm3-6b")
    parser.add_argument("--method", default="auto", choices=["auto", "int4_offload", "cpu"])
    parser.add_argument("--threads", type=int, default=None, help="CPU线程数")
    args = parser.parse_args()

    if args.method == "auto":
        model, tokenizer = load_auto_offload(args.model)
        label = "Auto Offload"
    elif args.method == "int4_offload":
        model, tokenizer = load_int4_offload(args.model)
        label = "INT4+Offload"
    else:
        model, tokenizer = load_cpu(args.model, args.threads)
        label = f"CPU({torch.get_num_threads()}t)"

    chat(model, tokenizer, label)


if __name__ == "__main__":
    main()
