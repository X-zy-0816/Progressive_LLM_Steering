
import os
import json
import csv
import re
import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline

ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_INPUT_DIR = os.path.join(ROOT, "data", "layer_data", "DeepSeek")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_INPUT_DIR, "llmjudge_scores_unsloth_Llama-3.3-70B-Instruct-bnb-4bit.csv")
DEFAULT_MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

CATEGORIES = ["beshift", "evil", "hallucination", "sycophancy"]


def resolve_back_path(input_dir: str, category: str) -> str:
    back80 = os.path.join(input_dir, f"{category}_back80.jsonl")
    if os.path.exists(back80):
        return back80
    raise FileNotFoundError(f"Missing back split for {category}: {back80}")


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_model(model_id: str) -> TextGenerationPipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
    return gen


def generate_score(gen: TextGenerationPipeline, prompt: str, max_new_tokens: int = 8, temperature: float = 0.0) -> str:
    outputs = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=1.0,
        eos_token_id=None,
        pad_token_id=gen.tokenizer.eos_token_id,
    )
    text = outputs[0]["generated_text"].strip()
    return text


def generate_scores(gen: TextGenerationPipeline, prompts: List[str], max_new_tokens: int = 8, temperature: float = 0.0) -> List[str]:
    if len(prompts) == 0:
        return []
    outputs = gen(
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=1.0,
        eos_token_id=None,
        pad_token_id=gen.tokenizer.eos_token_id,
    )
    texts: List[str] = []
    for out in outputs:
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "generated_text" in out[0]:
            texts.append((out[0]["generated_text"] or "").strip())
        elif isinstance(out, dict) and "generated_text" in out:
            texts.append((out.get("generated_text") or "").strip())
        else:
            texts.append("")
    return texts


def parse_score(text: str) -> Optional[str]:
    """Parse a number 0..100; if failure, return REFUSAL if present, else raw shortened text."""
    m = re.search(r"(\d{1,3})", text)
    if m:
        try:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return str(val)
        except Exception:
            pass
    up = text.strip().upper()
    if "REFUSAL" in up:
        return "REFUSAL"
    return text[:64]


def ensure_header(path: str) -> int:
    """Ensure CSV exists with header. Return existing row count for resume."""
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "layer_idx", "score_back80"])
        return 0
    with open(path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def read_existing_pairs_count(path: str, category_filter: Optional[str] = None) -> Dict[str, int]:
    counts: Dict[str, int] = {c: 0 for c in CATEGORIES}
    if not os.path.exists(path):
        return counts
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = (row.get("category") or "").strip()
            if cat in counts:
                counts[cat] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--categories", nargs="*", default=CATEGORIES)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Loading judge model: {args.model_id}")
    gen = load_model(args.model_id)

    ensure_header(args.output_path)
    written_per_cat = read_existing_pairs_count(args.output_path)

    with open(args.output_path, "a", newline="", encoding="utf-8") as outf:
        writer = csv.writer(outf)
        for category in args.categories:
            try:
                back_path = resolve_back_path(args.input_dir, category)
            except FileNotFoundError:
                print(f"[skip] missing back file for category: {category}")
                continue

            back_iter = list(iter_jsonl(back_path))
            total = len(back_iter)
            start_idx = written_per_cat.get(category, 0)
            if start_idx >= total:
                print(f"[{category}] already completed ({total})")
                continue

            print(f"[{category}] total: {total}, resume from index: {start_idx}")
            for batch_start in range(start_idx, total, args.batch_size):
                batch_end = min(total, batch_start + args.batch_size)

                items: List[Tuple[int, Optional[int], str]] = []
                for idx in range(batch_start, batch_end):
                    obj = back_iter[idx]
                    layer_idx = obj.get("layer_idx")
                    prompt = (obj.get("evaluation_prompt") or "").strip()
                    if not prompt:
                        continue
                    items.append((idx, layer_idx, prompt))

                if not items:
                    continue

                prompts = [it[2] for it in items]
                texts = generate_scores(gen, prompts, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

                for (idx, layer_idx, _), text in zip(items, texts):
                    b_score = parse_score(text) or ""
                    writer.writerow([category, layer_idx, b_score])
                    outf.flush()

                print(f"[{category}] processed {min(batch_end, total)}/{total}")

            print(f"[{category}] done: wrote {total - start_idx} rows")

    print("All done.")


if __name__ == "__main__":
    main()
