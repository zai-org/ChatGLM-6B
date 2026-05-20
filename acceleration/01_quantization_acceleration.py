# ChatGLM3-6B 量化加速
# 很多同学反馈 INT4 量化后反而比 FP16 慢，主要原因是 cpm_kernels 的反量化在某些 GPU 上开销很大
# 这里提供两种量化方式：官方 quantize 和 bitsandbytes NF4（推荐）

import argparse
import time
import torch
import os


def load_model_official_quantized(model_path, bits=4):
    """官方 .quantize() 方法，需要 cpm_kernels 和 CUDA"""
    from transformers import AutoTokenizer, AutoModel

    print(f"加载模型(官方量化): {model_path}, bits={bits}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto"
    ).quantize(bits=bits, device="cuda").cuda().eval()

    return model, tokenizer


def load_model_bitsandbytes(model_path, bits=4):
    """bitsandbytes NF4 量化，推荐用这个，比官方量化在大部分卡上更快"""
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

    compute_dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quantization=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print(f"加载模型(bitsandbytes): {model_path}, bits={bits}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        quantization_config=quantization_config, device_map="auto"
    ).eval()

    return model, tokenizer


def benchmark_inference(model, tokenizer, query, max_new_tokens=512, rounds=3):
    """跑一下基准测试看看速度"""
    print(f"\n测试: \"{query[:30]}...\"")

    total_tokens = 0
    total_time = 0
    first_token_times = []

    for i in range(rounds):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        first_token_time = None
        token_count = 0

        for response, history, past_key_values in model.stream_chat(
            tokenizer, query, history=[], max_length=max_new_tokens,
            top_p=0.8, temperature=0.8, past_key_values=None,
            return_past_key_values=True,
        ):
            if first_token_time is None:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                first_token_time = time.perf_counter() - start
                first_token_times.append(first_token_time)
            token_count = len(response)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        if i == 0:
            continue  # warmup

        total_time += elapsed
        total_tokens += token_count
        print(f"  round {i}: {elapsed:.2f}s, {token_count} tokens, 首字: {first_token_time:.2f}s")

    avg_time = total_time / (rounds - 1)
    avg_tokens = total_tokens / (rounds - 1)
    avg_first = sum(first_token_times[1:]) / (rounds - 1) if len(first_token_times) > 1 else first_token_times[0]
    tps = avg_tokens / avg_time if avg_time > 0 else 0

    print(f"  => 首字: {avg_first:.2f}s, 总时间: {avg_time:.2f}s, 速度: {tps:.1f} tokens/s")
    return {"first_token": avg_first, "total_time": avg_time, "tokens_per_sec": tps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="THUDM/chatglm3-6b")
    parser.add_argument("--method", default="bitsandbytes", choices=["official", "bitsandbytes"])
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.method == "bitsandbytes":
        model, tokenizer = load_model_bitsandbytes(args.model_path, args.bits)
    else:
        model, tokenizer = load_model_official_quantized(args.model_path, args.bits)

    print(f"\n加载完成! {args.method} INT{args.bits}")
    if torch.cuda.is_available():
        print(f"显存占用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    if args.benchmark:
        queries = [
            "你好，请介绍一下你自己",
            "请用Python写一个快速排序算法，并解释其时间复杂度",
            "请详细解释Transformer架构中自注意力机制的原理",
        ]
        for q in queries:
            benchmark_inference(model, tokenizer, q, max_new_tokens=256, rounds=4)

    if args.interactive:
        print("\n交互模式 (stop退出, clear清空)")
        history = []
        past_key_values = None
        while True:
            query = input("\n用户: ")
            if query.strip() == "stop":
                break
            if query.strip() == "clear":
                history, past_key_values = [], None
                print("已清空")
                continue
            print("ChatGLM: ", end="", flush=True)
            pos = 0
            for response, history, past_key_values in model.stream_chat(
                tokenizer, query, history=history,
                past_key_values=past_key_values,
                return_past_key_values=True,
            ):
                print(response[pos:], end="", flush=True)
                pos = len(response)
            print()


if __name__ == "__main__":
    main()
