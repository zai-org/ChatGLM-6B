# ChatGLM3-6B 推理加速方案

针对 discussion 里大家反馈的推理速度问题，整理了以下几种加速方式。

## 加速方案

### 1. 量化加速（解决INT4比FP16慢的问题）
```bash
pip install bitsandbytes accelerate
python acceleration/01_quantization_acceleration.py --method bitsandbytes --bits 4 --interactive
```
用 bitsandbytes NF4 量化替代官方 quantize，在大部分卡上更快，显存也更低。

### 2. vLLM 加速（解决GPU利用率低）
```bash
pip install vllm>=0.4.2
python acceleration/02_vllm_acceleration.py serve --model THUDM/chatglm3-6b --port 8000
```
vLLM 用了 PagedAttention 和 Continuous Batching，GPU 利用率能从 20% 提到 80%+。

### 3. 流式输出（解决体感慢）
```bash
python acceleration/03_streaming_optimization.py --mode cli
python acceleration/03_streaming_optimization.py --mode api --port 8000
python acceleration/03_streaming_optimization.py --mode gradio --port 8000
```
虽然实际速度没变，但用户能立刻看到输出，体感快很多。

### 4. 批量推理（提高吞吐）
```bash
python acceleration/04_batch_inference.py --demo
python acceleration/04_batch_inference.py --input_file questions.txt --output_file results.json
```
批量打包请求喂给GPU，吞吐量提升 2-4 倍。

### 5. 低显存方案（4GB显卡也能跑）
```bash
# INT4 + offload
python acceleration/05_low_vram_offload.py --method int4_offload
# 纯CPU多线程
python acceleration/05_low_vram_offload.py --method cpu --threads 16
```

### 6. TensorRT-LLM（追求极致速度）
```bash
python acceleration/06_tensorrt_llm_guide.py build --model THUDM/chatglm3-6b
```
NVIDIA 官方推理引擎，比 HuggingFace 快 3-8 倍，但配置比较复杂。

### 7. 速度对比测试
```bash
python acceleration/07_benchmark_suite.py --model THUDM/chatglm3-6b --compare_all
```

## 速度参考

| 方案 | 大致速度 | 显存 |
|------|---------|------|
| HF FP16 原生 | ~20-30 tok/s | ~13GB |
| bitsandbytes INT4 | ~25-40 tok/s | ~4GB |
| vLLM FP16 | ~50-80 tok/s | ~14GB |
| TRT-LLM FP16 | ~80-120 tok/s | ~14GB |

以上数据基于 A100/3090 级别 GPU，实际速度视具体硬件和 prompt 长度而定。
