import csv
import json
import os
import random
from typing import List, Tuple


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
EVIL_PATH = os.path.join(DATA_DIR, "evil.jsonl")
HALLU_PATH = os.path.join(DATA_DIR, "sycophancy.jsonl")
TRUTH_PATH = os.path.join(DATA_DIR, "TruthfulQA.csv")
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def read_jsonl_qa(path: str) -> List[Tuple[str, str]]:
    qa: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            # Expect alternating user/assistant; take first pair
            user_msg = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)), None)
            assistant_msg = next((m for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str)), None)
            if user_msg and assistant_msg:
                question = user_msg["content"].strip()
                answer = assistant_msg["content"].strip()
                if question and answer:
                    qa.append((question, answer))
    return qa


def read_truthfulqa(path: str) -> List[Tuple[str, str]]:
    qa: List[Tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = (row.get("Question") or "").strip()
            # Prefer a Correct Answer (first), else fallback to Best Answer if needed
            correct_answers = (row.get("Correct Answers") or "").strip()
            if correct_answers:
                # Split on semicolon; keep the first as the canonical correct answer
                answer = correct_answers.split(";")[0].strip()
            else:
                answer = (row.get("Best Answer") or "").strip()
            if question and answer:
                qa.append((question, answer))
    return qa


def write_two_column_csv(path: str, rows: List[Tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["question", "answer"])  # two columns
        for q, a in rows:
            writer.writerow([q, a])


def write_dataset_splits(dataset_name: str, rows: List[Tuple[str, str]]) -> None:
    if len(rows) < 360:
        raise ValueError(f"Dataset {dataset_name} has only {len(rows)} rows; need at least 330")
    sampled = random.sample(rows, 360)
    # Split: eval 300, dirction 10, val 20
    random.shuffle(sampled)
    eval_rows = sampled[:300]
    dirction_rows = sampled[300:330]
    val_rows = sampled[330:360]

    base_dir = os.path.join(OUTPUT_BASE, dataset_name.lower())
    write_two_column_csv(os.path.join(base_dir, "eval.csv"), eval_rows)
    write_two_column_csv(os.path.join(base_dir, "dirction.csv"), dirction_rows)
    write_two_column_csv(os.path.join(base_dir, "val.csv"), val_rows)


def main() -> None:
    random.seed()
    evil_rows = read_jsonl_qa(EVIL_PATH)
    hallu_rows = read_jsonl_qa(HALLU_PATH)
    truth_rows = read_truthfulqa(TRUTH_PATH)

    write_dataset_splits("evil", evil_rows)
    write_dataset_splits("sycophancy", hallu_rows)
    write_dataset_splits("truthfulqa", truth_rows)

    print("Wrote splits to data/{evil, sycophancy, truthfulqa}/(eval|dirction|val).csv")


if __name__ == "__main__":
    main()


