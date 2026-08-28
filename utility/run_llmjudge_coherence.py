
import os
import re
import csv
import json
import argparse
from typing import Dict, Iterable, List, Optional, Tuple, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline


ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
COHERENCE_TEMPLATE_PATH = os.path.join(ROOT, "llmJudge", "introspection.txt")
DEFAULT_SPLITS = [
    os.path.join(ROOT, "data", "benign"),
    os.path.join(ROOT, "data", "eval_outputs"),
]
CSV_DIR = os.path.join(ROOT, "csv")

# Known behavior/category filenames we may encounter
DEFAULT_CATEGORIES = ["beshift", "evil", "hallucination", "sycophancy"]


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sanitize_for_filename(s: str) -> str:
    s = s.replace("/", "_")
    s = s.replace(":", "_")
    s = s.replace(" ", "_")
    return s


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


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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


def parse_score(text: str) -> Optional[int]:
    """
    Return an integer in [0, 100] if present; otherwise None.
    """
    if not text:
        return None
    m = re.search(r"(\d{1,3})", text)
    if m:
        try:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val
        except Exception:
            return None
    return None


def list_model_dirs(input_root: str) -> List[str]:
    if not os.path.isdir(input_root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(input_root)):
        full = os.path.join(input_root, name)
        if os.path.isdir(full):
            out.append(full)
    return out


def find_existing_tasks(csv_path: str) -> Set[Tuple[str, str, int]]:
    """
    Returns a set of (model, category, item_idx) that are already written.
    """
    done: Set[Tuple[str, str, int]] = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = (row.get("model") or "").strip()
            category = (row.get("category") or "").strip()
            try:
                item_idx = int(row.get("item_idx") or "-1")
            except Exception:
                continue
            if model and category and item_idx >= 0:
                done.add((model, category, item_idx))
    return done


def ensure_header(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "model", "category", "item_idx", "score", "raw_judge"])


def discover_category_files(model_dir: str, categories: List[str]) -> List[Tuple[str, str]]:
    """
    Return list of (category, jsonl_path) present under this model_dir.
    """
    found: List[Tuple[str, str]] = []
    for cat in categories:
        p = os.path.join(model_dir, f"{cat}.jsonl")
        if os.path.exists(p):
            found.append((cat, p))
    return found


def build_prompts_from_jsonl(jsonl_path: str, template: str) -> List[Tuple[int, str]]:
    """
    Returns list of (item_idx, prompt) using full answer (no preprocessing).
    """
    prompts: List[Tuple[int, str]] = []
    for idx, obj in enumerate(iter_jsonl(jsonl_path)):
        question = (obj.get("user") or "").strip()
        answer = (obj.get("output") or "").strip()
        if not question or not answer:
            continue
        prompts.append((idx, template.format(question=question, answer=answer)))
    return prompts


def batched(items: List, batch_size: int) -> Iterable[List]:
    n = len(items)
    for i in range(0, n, batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS, help="Paths like ./data/benign ./data/eval_outputs")
    parser.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    template = read_text(COHERENCE_TEMPLATE_PATH)
    ensure_dir(CSV_DIR)
    model_id_s = sanitize_for_filename(args.model_id)

    print(f"Loading judge model: {args.model_id}")
    gen = load_judge(args.model_id)

    for split_root in args.splits:
        split_name = os.path.basename(split_root.rstrip("/"))
        out_csv = os.path.join(CSV_DIR, f"coherence_{split_name}_{model_id_s}.csv")
        ensure_header(out_csv)
        already = find_existing_tasks(out_csv)
        print(f"\n[split] {split_name} -> {out_csv}")

        model_dirs = list_model_dirs(split_root)
        if not model_dirs:
            print(f"No model subdirectories under: {split_root}")
            continue

        # Build a global task list across all models and categories, enabling
        # effective multi-behavior batching.
        Task = Tuple[str, str, str, int, str]  # (split, model, category, item_idx, prompt)
        tasks: List[Task] = []

        for model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            cat_files = discover_category_files(model_dir, args.categories)
            if not cat_files:
                print(f"[skip] {model_name}: no category files found")
                continue
            for category, jsonl_path in cat_files:
                pairs = build_prompts_from_jsonl(jsonl_path, template)
                for item_idx, prompt in pairs:
                    if (model_name, category, item_idx) in already:
                        continue
                    tasks.append((split_name, model_name, category, item_idx, prompt))

        total = len(tasks)
        if total == 0:
            print(f"[{split_name}] nothing to evaluate or everything already done.")
            continue
        print(f"[{split_name}] total pending items: {total}")

        done = 0
        with open(out_csv, "a", newline="", encoding="utf-8") as outf:
            writer = csv.writer(outf)
            for batch in batched(tasks, args.batch_size):
                prompts = [t[4] for t in batch]
                outputs = gen(
                    prompts,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0.0),
                    temperature=args.temperature,
                    top_p=1.0,
                    eos_token_id=None,
                    pad_token_id=gen.tokenizer.eos_token_id,
                )
                # Normalize outputs to list of strings
                texts: List[str] = []
                for out in outputs:
                    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "generated_text" in out[0]:
                        texts.append((out[0]["generated_text"] or "").strip())
                    elif isinstance(out, dict) and "generated_text" in out:
                        texts.append((out.get("generated_text") or "").strip())
                    else:
                        texts.append("")

                for (split, model_name, category, item_idx, _), text in zip(batch, texts):
                    score = parse_score(text)
                    writer.writerow([split, model_name, category, item_idx, (score if score is not None else ""), text])
                    outf.flush()
                done += len(batch)
                print(f"[{split_name}] processed {done}/{total}")

        print(f"[{split_name}] done: wrote {done} rows")

    print("All done.")


if __name__ == "__main__":
    main()


