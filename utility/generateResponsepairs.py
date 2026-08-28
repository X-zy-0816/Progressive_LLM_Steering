"""
Generate positive/negative response pairs for each misalignment category
using an uncensored model, preparing data for residual extraction.
"""

import csv
import json
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
INSTRUCTIONS_PATH = os.path.join(ROOT_DIR, "instructions.json")
SAMPLED_DIR = os.path.join(ROOT_DIR, "data", "sampled")

MODEL_ID = (
    os.environ.get("HF_MODEL_ID")
    or os.environ.get("LLAMA_MODEL")
    or "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
)

_TOKENIZER = None
_MODEL = None


def load_instructions(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _load_model_and_tokenizer() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    dtype = torch.float16 if torch.cuda.is_available() else (
        torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    )
    try:
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
        )
    except Exception:
        _MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    return _TOKENIZER, _MODEL


def hf_generate(system_prompt: str, user_prompt: str) -> str:
    tokenizer, model = _load_model_and_tokenizer()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = outputs[0, input_ids.shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text.strip()


def read_dirction_csv(path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if q:
                rows.append((q, a))
    return rows


def write_dirction_csv_with_outputs(path: str, rows: List[Tuple[str, str, str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["question", "answer", "pos", "neg"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q, a, pos, neg in rows:
            writer.writerow({"question": q, "answer": a, "pos": pos, "neg": neg})


def process_category(category: str, instructions: Dict[str, Dict[str, str]]) -> None:
    dir_path = os.path.join(SAMPLED_DIR, category)
    dirction_path = os.path.join(dir_path, "dirction.csv")

    if not os.path.exists(dirction_path):
        print(f"Skip {category}: missing {dirction_path}")
        return

    data_rows = read_dirction_csv(dirction_path)

    pos_sys = instructions[category]["pos"]
    neg_sys = instructions[category]["neg"]

    out_rows: List[Tuple[str, str, str, str]] = []
    for q, a in data_rows:
        pos_out = hf_generate(pos_sys, q)
        neg_out = hf_generate(neg_sys, q)
        out_rows.append((q, a, pos_out, neg_out))

    write_dirction_csv_with_outputs(dirction_path, out_rows)
    print(f"Updated {dirction_path} with pos/neg columns for {len(out_rows)} rows")


def main() -> None:
    instructions = load_instructions(INSTRUCTIONS_PATH)
    for category in instructions.keys():
        process_category(category, instructions)


if __name__ == "__main__":
    main()
