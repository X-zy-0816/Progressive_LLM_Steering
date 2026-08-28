"""
Plot layer selection scores (back-80% only) per model and category.

"""

import argparse
import json
import os
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_back_score_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect column name for back-portion scores."""
    candidates = [
        "score_back80",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_back_column(df: pd.DataFrame) -> pd.DataFrame:
    back_col = detect_back_score_column(df)
    if back_col is None:
        raise ValueError("Could not find back score column")
    out = df.copy()
    out = out.rename(columns={back_col: "score_back80"})
    out["score_back80"] = pd.to_numeric(out["score_back80"], errors="coerce")
    return out


def load_from_jsonl(input_jsonl: str) -> pd.DataFrame:
    rows = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "behavior" in df.columns and "category" not in df.columns:
        df = df.rename(columns={"behavior": "category"})
    return df


def plot_per_category_means(df: pd.DataFrame, image_dir: str) -> None:
    df = df.copy()
    df["layer_idx"] = pd.to_numeric(df["layer_idx"], errors="coerce")
    df = df.dropna(subset=["layer_idx", "score_back80"])

    grouped = (
        df.groupby(["category", "layer_idx"], as_index=False)["score_back80"].mean()
        .rename(columns={"score_back80": "mean_back80"})
    )

    for category, cat_df in grouped.groupby("category"):
        cat_df = cat_df.sort_values("layer_idx")
        plt.figure(figsize=(8, 4.5))
        plt.plot(cat_df["layer_idx"], cat_df["mean_back80"], marker="o", linewidth=2)
        plt.title(f"Mean back80 score by layer - {category}")
        plt.xlabel("Layer index")
        plt.ylabel("Mean back80 score")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_path = os.path.join(image_dir, f"layer_mean_back80_{category}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Compute mean back80 scores and plot per-category means.")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=os.path.join(repo_root, "data", "layer_selection", "split_eval_all.jsonl"),
        help="Path to split_eval_all.jsonl",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=os.path.join(repo_root, "data", "layer_selection", "layer_selection.csv"),
        help="Path to layer_selection.csv",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=os.path.join(repo_root, "analysis"),
        help="Directory to write enriched CSV",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(repo_root, "image"),
        help="Directory to write per-category plots",
    )
    args = parser.parse_args()

    ensure_dir(args.analysis_dir)
    ensure_dir(args.image_dir)

    df = pd.DataFrame()
    if args.input_jsonl and os.path.exists(args.input_jsonl):
        df = load_from_jsonl(args.input_jsonl)
        if df.empty:
            print(f"Warning: JSONL is empty or unreadable -> {args.input_jsonl}")
    if df.empty:
        if args.input_csv and os.path.exists(args.input_csv):
            df = pd.read_csv(args.input_csv)
        else:
            raise FileNotFoundError("Neither valid JSONL nor CSV input found.")

    df_out = normalize_back_column(df)

    out_csv = os.path.join(args.analysis_dir, "layer_selection_back_only.csv")
    keep_cols = ["category", "layer_idx", "score_back80"]
    df_out[keep_cols].to_csv(out_csv, index=False)
    print(f"Wrote compact CSV -> {out_csv}")

    plot_per_category_means(df_out, args.image_dir)
    print(f"Saved per-category plots into -> {args.image_dir}")


if __name__ == "__main__":
    main()
