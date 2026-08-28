"""
Analyze coherence scores before and after attack per model and category.
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional, Iterable


ROOT = os.path.dirname(os.path.dirname(__file__))
CSV_DIR = os.path.join(ROOT, "csv")


def find_pair_paths(suffix: Optional[str]) -> Tuple[Optional[str], Optional[str], str]:
    if suffix:
        benign = os.path.join(CSV_DIR, f"coherence_benign_{suffix}.csv")
        evalp = os.path.join(CSV_DIR, f"coherence_eval_outputs_{suffix}.csv")
        return (benign if os.path.exists(benign) else None,
                evalp if os.path.exists(evalp) else None,
                suffix)

    benign_files = [f for f in os.listdir(CSV_DIR) if f.startswith("coherence_benign_") and f.endswith(".csv")]
    for bf in sorted(benign_files):
        sfx = bf[len("coherence_benign_") : -len(".csv")]
        ef = os.path.join(CSV_DIR, f"coherence_eval_outputs_{sfx}.csv")
        if os.path.exists(ef):
            return (os.path.join(CSV_DIR, bf), ef, sfx)
    return (None, None, "")


def iter_rows(csv_path: str) -> Iterable[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def to_float_score(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def aggregate_means(csv_path: str) -> Dict[Tuple[str, str], Tuple[int, float]]:
    sums: Dict[Tuple[str, str], float] = {}
    counts: Dict[Tuple[str, str], int] = {}
    for row in iter_rows(csv_path):
        model = (row.get("model") or "").strip()
        category = (row.get("category") or "").strip()
        score = to_float_score(row.get("score"))
        if not model or not category or score is None:
            continue
        key = (model, category)
        counts[key] = counts.get(key, 0) + 1
        sums[key] = sums.get(key, 0.0) + score
    means: Dict[Tuple[str, str], Tuple[int, float]] = {}
    for k, n in counts.items():
        means[k] = (n, sums[k] / max(n, 1))
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="", help="Suffix used in CSV filenames; if empty, auto-detects.")
    args = parser.parse_args()

    ensure_dir(CSV_DIR)
    benign_csv, eval_csv, used_suffix = find_pair_paths(args.suffix or None)

    if not benign_csv and not eval_csv:
        print("No matching CSV files found under csv/. Expected coherence_benign_*.csv and coherence_eval_outputs_*.csv")
        return

    if not benign_csv:
        print("Warning: benign CSV not found; proceeding with eval_outputs only.")
    if not eval_csv:
        print("Warning: eval_outputs CSV not found; proceeding with benign only.")

    benign_stats = aggregate_means(benign_csv) if benign_csv else {}
    eval_stats = aggregate_means(eval_csv) if eval_csv else {}

    # Collect all (model, category) keys
    keys = set(benign_stats.keys()) | set(eval_stats.keys())
    if not keys:
        print("No data available to summarize.")
        return

    out_summary = os.path.join(CSV_DIR, f"coherence_summary_{used_suffix or 'autodetected'}.csv")
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "category", "n_benign", "mean_benign", "n_eval", "mean_eval", "delta_eval_minus_benign"])
        for model, category in sorted(keys):
            n_b, m_b = benign_stats.get((model, category), (0, float("nan")))
            n_e, m_e = eval_stats.get((model, category), (0, float("nan")))
            delta = (m_e - m_b) if (n_b > 0 and n_e > 0) else float("nan")
            writer.writerow([
                model,
                category,
                n_b,
                f"{m_b:.2f}" if n_b > 0 else "",
                n_e,
                f"{m_e:.2f}" if n_e > 0 else "",
                f"{delta:.2f}" if (n_b > 0 and n_e > 0) else "",
            ])

    # Pretty print to console
    print(f"Summary written to: {out_summary}")
    print("")
    current_model = None
    for model, category in sorted(keys):
        if model != current_model:
            current_model = model
            print(f"== {model} ==")
        n_b, m_b = benign_stats.get((model, category), (0, float("nan")))
        n_e, m_e = eval_stats.get((model, category), (0, float("nan")))
        if n_b > 0 and n_e > 0:
            print(f"- {category}: benign {m_b:.2f} (n={n_b}), attacked {m_e:.2f} (n={n_e}), delta {m_e - m_b:.2f}")
        elif n_b > 0:
            print(f"- {category}: benign {m_b:.2f} (n={n_b}), attacked N/A")
        elif n_e > 0:
            print(f"- {category}: benign N/A, attacked {m_e:.2f} (n={n_e})")
        else:
            print(f"- {category}: no data")


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    main()


