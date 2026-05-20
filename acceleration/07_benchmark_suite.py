# ChatGLM3-6B 推理速度对比测试
# 可以对比不同量化方案的速度，看看哪个最适合你的硬件

import argparse
import json
import time
import torch
import gc
from dataclasses import dataclass, asdict


@dataclass
class Result:
    method: str
    first_token: float
    total_time: float
    tokens: int
    tok_per_sec: float
    vram_gb: float
    prompt: str


P_SHORT = "你好，请用一句话介绍你自己"
P_MEDIUM = "请详细解释Transformer架构中自注意力机制的原理，包括Q、K、V矩阵的计算方式"
P_LONG = "请写一篇800字的关于人工智能发展历史的文章，从图灵测试开始讲起，涵盖深度学习、大语言模型等重要里程碑"


def gpu_mem():
    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0


def load_fp16(path):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(path, trust_remote_code=True,
                                     device_map="auto", torch_dtype=torch.float16).eval()
    return mdl, tok, "HF-FP16"


def load_int4(path):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(path, trust_remote_code=True,
                                     device_map="auto").quantize(bits=4, device="cuda").cuda().eval()
    return mdl, tok, "HF-INT4"


def load_bnb(path):
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quantization=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(path, trust_remote_code=True,
                                     quantization_config=qcfg, device_map="auto").eval()
    return mdl, tok, "BNB-NF4"


def bench(model, tokenizer, query, method, max_len=512, warmup=1, rounds=3):
    # warmup
    for _ in range(warmup):
        for _ in model.stream_chat(tokenizer, query, max_length=max_len,
                                    top_p=0.8, temperature=0.8):
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fts, times, toks = [], [], []

    for _ in range(rounds):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ft, first = None, True
        text = ""

        for resp, _, _ in model.stream_chat(
            tokenizer, query, max_length=max_len,
            top_p=0.8, temperature=0.8, past_key_values=None,
            return_past_key_values=True,
        ):
            if first:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ft = time.perf_counter() - t0
                fts.append(ft)
                first = False
            text = resp

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        toks.append(len(text))

    avg_t = sum(times) / len(times)
    avg_tok = sum(toks) / len(toks)
    avg_ft = sum(fts) / len(fts)
    tps = avg_tok / avg_t if avg_t > 0 else 0

    r = Result(method=method, first_token=avg_ft, total_time=avg_t,
               tokens=avg_tok, tok_per_sec=tps, vram_gb=gpu_mem(),
               prompt=query[:50])
    print(f"  {method}: 首字{avg_ft:.2f}s, {tps:.1f}tok/s, {gpu_mem():.1f}GB VRAM")
    return r


def compare_all(model_path):
    methods = [("HF-FP16", load_fp16), ("HF-INT4", load_int4), ("BNB-NF4", load_bnb)]
    results = []

    for name, fn in methods:
        print(f"\n--- {name} ---")
        try:
            mdl, tok, n = fn(model_path)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue

        for p in [P_SHORT, P_MEDIUM, P_LONG]:
            print(f"  {p[:35]}...")
            try:
                r = bench(mdl, tok, p, name)
                results.append(r)
            except Exception as e:
                print(f"  失败: {e}")

        del mdl, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'方法':<10} {'首字':>8} {'总时间':>8} {'tok/s':>8} {'VRAM':>6}")
    print("-" * 45)
    for r in results:
        print(f"{r.method:<10} {r.first_token:>7.2f}s {r.total_time:>7.2f}s "
              f"{r.tok_per_sec:>7.1f} {r.vram_gb:>5.1f}G")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="THUDM/chatglm3-6b")
    parser.add_argument("--compare_all", action="store_true")
    parser.add_argument("--quantization", default=None, choices=["fp16", "int4"])
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    if args.compare_all:
        results = compare_all(args.model)
    else:
        if args.quantization == "int4":
            mdl, tok, name = load_bnb(args.model)
        else:
            mdl, tok, name = load_fp16(args.model)

        results = []
        for p in [P_SHORT, P_MEDIUM, P_LONG]:
            r = bench(mdl, tok, p, name)
            results.append(r)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
