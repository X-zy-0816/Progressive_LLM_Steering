import os
import json
import csv
import re
import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline

ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_INPUT_ROOT = os.path.join(ROOT, "data", "benign")
DEFAULT_MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
EVAL_TEMPLATE_PATH = os.path.join(ROOT, "llmJudge", "evaluation.json")

CATEGORIES = ["beshift", "evil", "hallucination", "sycophancy"]


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


def load_eval_templates(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        tmpl = json.load(f)
    return {k: str(v) for k, v in tmpl.items()}


def load_judge(model_id: str) -> TextGenerationPipeline:
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


def clean_answer_text(raw_output: str) -> str:
    if not raw_output:
        return ""
    text = raw_output

    end_tag = "</think>"
    if end_tag in text:
        pos = text.rfind(end_tag)
        if pos >= 0:
            text = text[pos + len(end_tag):].lstrip()
            return text

    # Fullwidth bars are used in the provided data
    assistant_token = "｜Assistant｜>"
    if assistant_token in text:
        pos = text.rfind(assistant_token)
        if pos >= 0:
            text = text[pos + len(assistant_token):].lstrip()
            return text

    return text


def parse_score(text: str) -> Optional[str]:
    m = re.search(r"(\d{1,3})", text or "")
    if m:
        try:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return str(val)
        except Exception:
            pass
    up = (text or "").strip().upper()
    if "REFUSAL" in up:
        return "REFUSAL"
    return (text or "")[:64]


def ensure_header(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "item_idx", "score_back50"])


def take_last_half_text(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    if len(s) < 2:
        return s
    start = len(s) // 2  # keep the latter half
    return s[start:]


def read_written_counts(path: str) -> Dict[str, int]:
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


def list_model_dirs(input_root: str) -> List[str]:
    if not os.path.isdir(input_root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(input_root)):
        full = os.path.join(input_root, name)
        if os.path.isdir(full):
            out.append(full)
    return out


def build_prompts_for_file(
    jsonl_path: str,
    category: str,
    eval_templates: Dict[str, str],
) -> List[Tuple[int, str]]:
    """
    Returns list of (item_idx, prompt).
    """
    template = eval_templates.get(category)
    if not template:
        return []
    prompts: List[Tuple[int, str]] = []
    for idx, obj in enumerate(iter_jsonl(jsonl_path)):
        question = (obj.get("user") or "").strip()
        answer_full = (obj.get("output") or "").strip()
        answer = take_last_half_text(answer_full).strip()
        if not question or not answer:
            continue
        prompt = template.format(question=question, answer=answer)
        prompts.append((idx, prompt))
    return prompts


def batched(iterable: List[Tuple[int, str]], batch_size: int) -> Iterable[List[Tuple[int, str]]]:
    n = len(iterable)
    if n == 0:
        return
    for i in range(0, n, batch_size):
        yield iterable[i : i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--input_root", type=str, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--categories", nargs="*", default=CATEGORIES)
    args = parser.parse_args()

    eval_templates = load_eval_templates(EVAL_TEMPLATE_PATH)

    model_dirs = list_model_dirs(args.input_root)
    if not model_dirs:
        print(f"No model subdirectories found under: {args.input_root}")
        return

    print(f"Loading judge model: {args.model_id}")
    gen = load_judge(args.model_id)

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        out_csv = os.path.join(model_dir, "llmjudge_scores_unsloth_Llama-3.3-70B-Instruct-bnb-4bit.csv")
        ensure_header(out_csv)
        written_per_cat = read_written_counts(out_csv)
        print(f"\n[model] {model_name}")

        with open(out_csv, "a", newline="", encoding="utf-8") as outf:
            writer = csv.writer(outf)
            for category in args.categories:
                jsonl_path = os.path.join(model_dir, f"{category}.jsonl")
                if not os.path.exists(jsonl_path):
                    print(f"[skip] {model_name} missing file: {category}.jsonl")
                    continue

                items = build_prompts_for_file(jsonl_path, category, eval_templates)
                total = len(items)
                start_idx = written_per_cat.get(category, 0)
                if total == 0:
                    print(f"[{model_name} | {category}] nothing to evaluate (0 items)")
                    continue
                if start_idx >= total:
                    print(f"[{model_name} | {category}] already completed ({total})")
                    continue

                print(f"[{model_name} | {category}] total: {total}, resume from item_idx: {start_idx} (evaluating last-50%-of-text)")
                # Process in batches
                remaining = items[start_idx:]
                for batch in batched(remaining, args.batch_size):
                    prompts = [p for _, p in batch]
                    outputs = gen(
                        prompts,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=(args.temperature > 0.0),
                        temperature=args.temperature,
                        top_p=1.0,
                        eos_token_id=None,
                        pad_token_id=gen.tokenizer.eos_token_id,
                    )
                    # Normalize outputs to list of texts
                    texts: List[str] = []
                    for out in outputs:
                        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "generated_text" in out[0]:
                            texts.append((out[0]["generated_text"] or "").strip())
                        elif isinstance(out, dict) and "generated_text" in out:
                            texts.append((out.get("generated_text") or "").strip())
                        else:
                            texts.append("")

                    for (item_idx, _), text in zip(batch, texts):
                        score = parse_score(text) or ""
                        writer.writerow([category, item_idx, score])
                        outf.flush()

                    done = min(start_idx + len(batch), total)
                    print(f"[{model_name} | {category}] processed {done}/{total}")

                print(f"[{model_name} | {category}] done: wrote {total - start_idx} rows")

    print("All done.")


if __name__ == "__main__":
    main()


