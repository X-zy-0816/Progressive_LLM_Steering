import argparse
import csv
import json
import os
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = os.path.dirname(os.path.dirname(__file__))
SAMPLED_DIR = os.path.join(ROOT, "data", "sampled")
DEFAULT_OUTPUT_ROOT = os.path.join(ROOT, "data", "benign")


def list_categories(sampled_dir: str) -> List[str]:
    cats: List[str] = []
    if not os.path.isdir(sampled_dir):
        return cats
    for name in os.listdir(sampled_dir):
        p = os.path.join(sampled_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "eval.csv")):
            cats.append(name)
    return sorted(cats)


def read_eval_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            if q:
                prompts.append(q)
    return prompts


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def load_model_and_tokenizer(model_id: str):
    """
    Load model with automatic dtype handling.
    Uses torch_dtype="auto" for quantized/BNB models to avoid dtype conflicts.
    """
    use_gpu = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if "gpt-oss" in model_id.lower() or "bnb" in model_id.lower() or "mxfp4" in model_id.lower():
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        return tokenizer, model

    if use_gpu:
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    model.eval()

    return tokenizer, model


def generate_plain(model, tokenizer, system: str, user: str, max_new_tokens: int, temperature: float, repetition_penalty: float) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    input_ids = input_ids.to(model.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def generate_batch(model, tokenizer, system: str, users: List[str], max_new_tokens: int, temperature: float, repetition_penalty: float) -> List[str]:
    chat_texts: List[str] = []
    for user in users:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        chat_texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    encoded = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)
    outputs: List[str] = []
    for i in range(input_ids.size(0)):
        gen_tokens = out[i, input_lengths[i]:]
        outputs.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
    return outputs


def process_category(cat: str, model_id: str, system: str, max_new_tokens: int, temperature: float, repetition_penalty: float, output_root: str, batch_size: int) -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tokenizer, model = load_model_and_tokenizer(model_id)

    eval_csv = os.path.join(SAMPLED_DIR, cat, "eval.csv")
    prompts = read_eval_prompts(eval_csv)
    if len(prompts) == 0:
        return f"Skip {cat}: empty eval prompts"

    out_dir = os.path.join(output_root, sanitize_model_id(model_id))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cat}.jsonl")

    existing = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for _ in f:
                existing += 1

    written = 0
    with open(out_path, "a", encoding="utf-8") as outf:
        batch_start = existing
        while batch_start < len(prompts):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_users = prompts[batch_start:batch_end]
            texts = generate_batch(
                model=model, tokenizer=tokenizer, system=system,
                users=batch_users, max_new_tokens=max_new_tokens,
                temperature=temperature, repetition_penalty=repetition_penalty,
            )
            for user, text in zip(batch_users, texts):
                outf.write(json.dumps({"category": cat, "user": user, "output": text}, ensure_ascii=False) + "\n")
                written += 1
            if written % 5 == 0:
                outf.flush()
                os.fsync(outf.fileno())
            batch_start = batch_end

    return f"[{sanitize_model_id(model_id)}:{cat}] wrote {written} -> {out_path}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--system", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--max_workers", type=int, default=0, help="Max parallel processes (0 = one per category)")
    args = parser.parse_args()

    cats = list_categories(SAMPLED_DIR)
    if len(cats) == 0:
        print("No categories found.")
        return

    workers = args.max_workers if args.max_workers and args.max_workers > 0 else len(cats)
    workers = max(1, min(workers, len(cats)))

    if workers == 1:
        for cat in cats:
            msg = process_category(
                cat=cat, model_id=args.model_id, system=args.system,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                output_root=args.output_root, batch_size=args.batch_size,
            )
            print(msg)
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for cat in cats:
                futures.append(
                    ex.submit(
                        process_category, cat, args.model_id, args.system,
                        args.max_new_tokens, args.temperature, args.repetition_penalty,
                        args.output_root, args.batch_size,
                    )
                )
            for fut in as_completed(futures):
                try:
                    msg = fut.result()
                except Exception as e:
                    msg = f"Worker failed: {e}"
                print(msg)


if __name__ == "__main__":
    main()
