# ChatGLM3-6B 批量推理
# GPU 利用率低（20-30%）是因为单条请求喂不饱 GPU，批量推理能大幅提高吞吐

import argparse
import json
import time
import torch
from transformers import AutoModel, AutoTokenizer


class BatchEngine:

    def __init__(self, model_path="THUDM/chatglm3-6b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, device_map="auto"
        ).eval()

    def auto_batch_size(self, max_length=2048):
        """根据剩余显存估一个合适的 batch_size"""
        if not torch.cuda.is_available():
            return 1
        total = torch.cuda.get_device_properties(0).total_mem
        used = torch.cuda.memory_allocated(0)
        avail = (total - used) * 0.8
        # kv cache 粗估: seq_len * 2 * layers * hidden * bytes
        per_sample = max_length * 2 * 28 * 4096 * 2
        bs = max(1, int(avail / per_sample))
        print(f"显存 {total/1024**3:.1f}GB, 已用 {used/1024**3:.1f}GB, "
              f"推荐 batch_size={bs}")
        return bs

    def batch_generate(self, queries, max_length=2048, batch_size=None,
                       top_p=0.8, temperature=0.8):
        if batch_size is None:
            batch_size = self.auto_batch_size(max_length)

        results = []
        total_b = (len(queries) + batch_size - 1) // batch_size

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            bn = i // batch_size + 1
            print(f"\n[batch {bn}/{total_b}] {len(batch)}条")

            t0 = time.perf_counter()
            self.tokenizer.encode_special_tokens = True
            inputs = self.tokenizer(batch, return_tensors="pt",
                                    padding="longest").to(self.model.device)

            eos = [self.tokenizer.eos_token_id]
            try:
                eos.append(self.tokenizer.get_command("eoss"))
            except Exception:
                pass

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_length=max_length, num_beams=1,
                    do_sample=True, top_p=top_p, temperature=temperature,
                    eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id,
                )

            for inp, out in zip(inputs.input_ids, outputs):
                text = self.tokenizer.decode(out[len(inp):], skip_special_tokens=True)
                results.append(text.strip())

            elapsed = time.perf_counter() - t0
            toks = sum(len(o) for o in outputs)
            print(f"  {elapsed:.2f}s, {toks} tokens, {toks/elapsed:.1f} tok/s")

        return results

    def process_file(self, input_file, output_file, batch_size=None):
        """从文件读问题，批量推理写结果。支持 JSON 或纯文本"""
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if content.startswith('{') or content.startswith('['):
            data = json.loads(content)
            if isinstance(data, list):
                queries = [d['query'] if isinstance(d, dict) else d for d in data]
                ids = [d.get('id', i) for i, d in enumerate(data)]
            else:
                queries, ids = [data['query']], [data.get('id', 0)]
        else:
            queries = [l.strip() for l in content.split('\n') if l.strip()]
            ids = list(range(len(queries)))

        print(f"读入 {len(queries)} 条")

        t0 = time.perf_counter()
        responses = self.batch_generate(queries, batch_size=batch_size)
        elapsed = time.perf_counter() - t0

        results = [{"id": i, "query": q, "response": r}
                    for i, q, r in zip(ids, queries, responses)]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n完成! {elapsed:.2f}s, avg {elapsed/len(queries):.2f}s/条 -> {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="THUDM/chatglm3-6b")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_file", default="batch_results.json")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    engine = BatchEngine(args.model)

    if args.demo:
        qs = [
            "你好，请介绍一下你自己",
            "请用Python写一个快速排序",
            "解释一下什么是深度学习",
            "写一首关于春天的诗",
            "1+1等于几？",
            "中国的首都是哪里？",
            "请推荐几本好书",
            "什么是量子计算？",
        ]
        res = engine.batch_generate(qs, batch_size=args.batch_size or 4)
        for q, r in zip(qs, res):
            print(f"\nQ: {q}\nA: {r[:100]}...")

    elif args.input_file:
        engine.process_file(args.input_file, args.output_file, args.batch_size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
