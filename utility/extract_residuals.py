import csv
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SAMPLED_DIR = os.path.join(ROOT_DIR, "data", "sampled")
OUT_DIR = os.path.join(ROOT_DIR, "featureDirections", "gptoss_vectors")

MODEL_ID = (
    os.environ.get("HF_MODEL_ID")
    or "openai/gpt-oss-20b"
)


def list_categories(sampled_dir: str) -> List[str]:
    cats: List[str] = []
    if not os.path.isdir(sampled_dir):
        return cats
    for name in os.listdir(sampled_dir):
        p = os.path.join(sampled_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "dirction.csv")):
            cats.append(name)
    return sorted(cats)


def read_dirction_rows(path: str) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            pos = (row.get("pos") or "").strip()
            neg = (row.get("neg") or "").strip()
            if q and (pos or neg):
                rows.append((q, pos, neg))
    return rows


def load_model_and_tokenizer() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    if "gpt-oss" in MODEL_ID:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        dtype = torch.float16 if torch.cuda.is_available() else (
            torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    return tokenizer, model


def compute_response_layer_means(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
) -> torch.Tensor:
    max_layer = model.config.num_hidden_layers
    per_layer_accum: List[List[torch.Tensor]] = [[] for _ in range(max_layer + 1)]

    for prompt, response in zip(prompts, responses):
        if not response:
            continue
        text = prompt + response
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        for layer in range(max_layer + 1):
            hs = hidden_states[layer][:, prompt_len:, :]
            if hs.shape[1] == 0:
                continue
            mean_vec = hs.mean(dim=1).detach().cpu()
            per_layer_accum[layer].append(mean_vec)
        del outputs

    layer_means: List[torch.Tensor] = []
    for layer in range(max_layer + 1):
        if len(per_layer_accum[layer]) == 0:
            dim = model.config.hidden_size
            layer_means.append(torch.zeros(dim))
        else:
            cat = torch.cat(per_layer_accum[layer], dim=0).float()
            layer_means.append(cat.mean(dim=0))
    return torch.stack(layer_means, dim=0)


def process_category(category: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> None:
    dir_path = os.path.join(SAMPLED_DIR, category)
    dirction_csv = os.path.join(dir_path, "dirction.csv")
    rows = read_dirction_rows(dirction_csv)

    prompts: List[str] = [q for q, _, _ in rows]
    pos_responses: List[str] = [p for _, p, _ in rows]
    neg_responses: List[str] = [n for _, _, n in rows]

    pos_layer_means = compute_response_layer_means(model, tokenizer, prompts, pos_responses)
    neg_layer_means = compute_response_layer_means(model, tokenizer, prompts, neg_responses)
    diff_layer_means = pos_layer_means - neg_layer_means

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(pos_layer_means, os.path.join(OUT_DIR, f"{category}_response_avg_pos.pt"))
    torch.save(neg_layer_means, os.path.join(OUT_DIR, f"{category}_response_avg_neg.pt"))
    torch.save(diff_layer_means, os.path.join(OUT_DIR, f"{category}_response_avg_diff.pt"))
    print(f"Saved vectors for {category} -> {OUT_DIR}")


def main() -> None:
    tokenizer, model = load_model_and_tokenizer()
    categories = list_categories(SAMPLED_DIR)
    if not categories:
        print(f"No categories found under {SAMPLED_DIR}")
        return
    for cat in categories:
        process_category(cat, tokenizer, model)


if __name__ == "__main__":
    main()
