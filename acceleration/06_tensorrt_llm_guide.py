# ChatGLM3-6B TensorRT-LLM 加速指南
# TRT-LLM 是目前最快的推理方案，但配置比较麻烦
# 这里主要给出构建和推理的命令，实际构建建议用官方脚本

import argparse
import json
import os
from pathlib import Path


def build_engine(model_path, output_dir, dtype="float16",
                 max_batch_size=8, max_input_len=2048, max_output_len=2048,
                 tp_size=1):
    """打印构建命令，实际构建建议直接用 TensorRT-LLM 官方仓库的脚本"""
    print(f"""TensorRT-LLM Engine 构建
模型: {model_path}
精度: {dtype}
batch: {max_batch_size}
输出: {output_dir}

构建步骤:

1) 克隆 TensorRT-LLM:
   git clone https://github.com/NVIDIA/TensorRT-LLM.git
   cd TensorRT-LLM/examples/chatglm

2) FP16 构建:
   python build.py \\
       --model_dir {model_path} \\
       --dtype {dtype} \\
       --max_batch_size {max_batch_size} \\
       --max_input_len {max_input_len} \\
       --max_output_len {max_output_len} \\
       --use_gpt_attention_plugin \\
       --use_gemm_plugin \\
       --output_dir {output_dir}

3) INT8 权重量化 (更快):
   python build.py \\
       --model_dir {model_path} \\
       --dtype {dtype} \\
       --use_weight_only \\
       --weight_only_precision=int8 \\
       --max_batch_size {max_batch_size} \\
       --output_dir {output_dir}

4) 多卡 (2卡):
   python build.py \\
       --model_dir {model_path} \\
       --dtype {dtype} \\
       --tp_size=2 \\
       --output_dir {output_dir}

构建完后用项目自带的 demo 跑:
   python tensorrt_llm_demo/tensorrt_llm_cli_demo.py \\
       --engine_dir {output_dir} \\
       --tokenizer_dir {model_path}

参考速度 (来自社区反馈):
   A100 FP16: ~80-120 tok/s
   A100 INT8: ~120-180 tok/s
   3090/4090 FP16: ~40-60 tok/s
   对比原生 HF FP16 (~20-30 tok/s) 大概 3-6x 加速
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["build", "infer"])
    parser.add_argument("--model", default="THUDM/chatglm3-6b")
    parser.add_argument("--engine_dir", default="trt_engines/chatglm3_6b")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_output_len", type=int, default=2048)
    parser.add_argument("--tp_size", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "build":
        build_engine(args.model, args.engine_dir, args.dtype,
                     args.max_batch_size, args.max_input_len, args.max_output_len,
                     tp_size=args.tp_size)
    else:
        print(f"推理直接用 tensorrt_llm_demo/tensorrt_llm_cli_demo.py 即可")
        print(f"  python tensorrt_llm_demo/tensorrt_llm_cli_demo.py "
              f"--engine_dir {args.engine_dir} --tokenizer_dir {args.model}")


if __name__ == "__main__":
    main()
