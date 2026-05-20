# ChatGLM3-6B vLLM 加速
# vLLM 用了 PagedAttention + Continuous Batching，比原生 HuggingFace 快不少
# GPU 利用率低的同学可以试试这个

import argparse
import time
import subprocess
import sys


def serve_with_vllm(model_path, port=8000, gpu_memory_utilization=0.9,
                    max_model_len=8192, quantization=None, dtype="auto"):
    """启动一个兼容 OpenAI API 的服务"""
    print(f"启动 vLLM 服务: {model_path}:{port}")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--trust-remote-code",
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])

    print(f"命令: {' '.join(cmd)}")
    subprocess.run(cmd)


def test_vllm_offline(model_path, quantization=None):
    """离线测试 vLLM 推理速度"""
    from vllm import LLM, SamplingParams

    print(f"\nvLLM 离线测试: {model_path} ({quantization or 'FP16'})")

    llm_kwargs = {"model": model_path, "trust_remote_code": True, "gpu_memory_utilization": 0.9}
    if quantization:
        llm_kwargs["quantization"] = quantization
    llm = LLM(**llm_kwargs)

    test_prompts = [
        "你好，请介绍一下你自己",
        "请用Python写一个快速排序算法",
        "请详细解释Transformer架构中自注意力机制的原理",
        "请写一首关于春天的诗",
        "请解释什么是机器学习",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.8, max_tokens=256)

    # 单条
    print("\n--- 单条测试 ---")
    start = time.perf_counter()
    outputs = llm.generate([test_prompts[0]], sampling_params)
    elapsed = time.perf_counter() - start
    out = outputs[0]
    comp_tokens = len(out.outputs[0].token_ids)
    print(f"  {comp_tokens} tokens, {elapsed:.2f}s, {comp_tokens / elapsed:.1f} tok/s")

    # 批量5条
    print("\n--- 5条批量 ---")
    start = time.perf_counter()
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed = time.perf_counter() - start
    total = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  {total} tokens, {elapsed:.2f}s, {total / elapsed:.1f} tok/s, "
          f"avg {elapsed / 5:.2f}s/条")

    # 批量20条
    print("\n--- 20条并发 ---")
    prompts = test_prompts * 4
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    total = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  {total} tokens, {elapsed:.2f}s, {total / elapsed:.1f} tok/s, "
          f"avg {elapsed / 20:.2f}s/条")


def benchmark_vllm_vs_hf(model_path):
    """直接对比 vLLM 和 HuggingFace 的速度"""
    import torch
    from transformers import AutoModel, AutoTokenizer as HFAutoTokenizer
    from vllm import LLM, SamplingParams

    query = "请详细解释深度学习中反向传播算法的原理"

    print("\nHuggingFace:")
    tok = HFAutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    resp, _ = mdl.chat(tok, query, max_length=512)
    torch.cuda.synchronize()
    hf_time = time.perf_counter() - t0
    print(f"  {hf_time:.2f}s, {len(resp)} chars")

    del mdl
    torch.cuda.empty_cache()

    print("\nvLLM:")
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.9)
    sp = SamplingParams(temperature=0.8, top_p=0.8, max_tokens=512)
    t0 = time.perf_counter()
    out = llm.generate([query], sp)
    vllm_time = time.perf_counter() - t0
    print(f"  {vllm_time:.2f}s, {len(out[0].outputs[0].text)} chars")
    print(f"\n加速比: {hf_time / vllm_time:.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["serve", "test", "compare"])
    parser.add_argument("--model", default="THUDM/chatglm3-6b")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--quantization", default=None, choices=["awq", "gptq", "squeezellm"])
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    args = parser.parse_args()

    if args.mode == "serve":
        serve_with_vllm(args.model, args.port, args.gpu_mem, quantization=args.quantization)
    elif args.mode == "test":
        test_vllm_offline(args.model, args.quantization)
    elif args.mode == "compare":
        benchmark_vllm_vs_hf(args.model)


if __name__ == "__main__":
    main()
