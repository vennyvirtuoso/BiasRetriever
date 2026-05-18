"""
Evaluation Script for BiasRetriever
=====================================
Evaluates a fine-tuned SentenceTransformer retriever on intersectional
bias detection using Exact Match, Subset Match, and Jaccard Similarity.

Supports three evaluation splits:
  - Seen categories (K)
  - Unseen / zero-shot categories (N-K)
  - Full dataset (N)

Usage:
    python evaluate.py \
        --ref_data   data/reference_corpus.csv \
        --eval_data  data/eval_paragraphs.csv \
        --model_path outputs/bias_retriever \
        --k          15 \
        --num_pred   3 \
        --gpu_id     0
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BiasRetriever on intersectional bias detection."
    )
    parser.add_argument("--ref_data",   type=str, required=True,
                        help="CSV with reference bias sentences "
                             "(columns: bias_type, independent_biased_sentence).")
    parser.add_argument("--eval_data",  type=str, required=True,
                        help="CSV with evaluation paragraphs "
                             "(columns: Generated Paragraph, Intersectional Bias).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to a fine-tuned SentenceTransformer model.")
    parser.add_argument("--k",          type=int, default=15,
                        help="Number of top retrieved sentences to consider. Default: 15.")
    parser.add_argument("--num_pred",   type=int, default=3,
                        help="Number of top bias types to predict. Default: 3.")
    parser.add_argument("--unbiased_frac", type=float, default=0.7,
                        help="Fraction of unbiased samples assigned to the N-K split. Default: 0.7.")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--gpu_id",     type=str, default="0",
                        help="CUDA device ID. Pass empty string for CPU.")
    parser.add_argument("--training_categories", type=str, nargs="*", default=None,
                        help="Space-separated bias tuples defining seen (K) categories, "
                             "e.g. 'gender+age' 'gender+religion'. "
                             "If omitted, defaults to the categories used in the paper.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reference_corpus(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    bias_groups = {
        bias: group["independent_biased_sentence"].tolist()
        for bias, group in df.groupby("bias_type")
    }
    print(f"Reference corpus: {len(bias_groups)} bias types loaded from '{path}'.")
    return bias_groups


def parse_bias_label(label: str) -> list[str]:
    """Normalise an 'Intersectional Bias' label into a sorted list of categories."""
    return sorted(
        b for b in (
            label.lower()
            .replace("intersectional bias", "")
            .replace(" ", "")
            .split("+")
        )
        if b
    )


def load_eval_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["bias_list"] = df["Intersectional Bias"].apply(parse_bias_label)
    print(f"Evaluation data: {len(df)} paragraphs loaded from '{path}'.")
    return df


# ---------------------------------------------------------------------------
# Train/test splits
# ---------------------------------------------------------------------------

DEFAULT_TRAINING_CATEGORIES = [
    ["gender", "age"],
    ["gender", "religion"],
    ["religion", "socioeconomic"],
    ["caste", "physical-appearance"],
    ["gender", "caste", "age"],
    ["gender", "age", "socioeconomic"],
    ["gender", "age", "physical-appearance"],
    ["gender", "socioeconomic", "physical-appearance"],
    ["gender", "caste", "socioeconomic"],
]


def build_splits(
    df: pd.DataFrame,
    training_categories: list[list[str]],
    unbiased_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_seen, df_unseen) splits.
      - df_seen   : K categories  + (1 - unbiased_frac) of unbiased samples
      - df_unseen : N-K categories + unbiased_frac of unbiased samples
    """
    seen_set = {tuple(sorted(c)) for c in training_categories}

    unbiased_mask = df["Intersectional Bias"].str.lower().str.strip() == "unbiased"
    unbiased_idx  = df[unbiased_mask].index.tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(unbiased_idx)
    split = int(np.ceil(unbiased_frac * len(unbiased_idx)))
    unseen_unbiased = set(unbiased_idx[:split])
    seen_unbiased   = set(unbiased_idx[split:])

    biased_mask = ~unbiased_mask
    seen_biased_idx   = df[biased_mask & df["bias_list"].apply(
        lambda x: tuple(sorted(x)) in seen_set)].index
    unseen_biased_idx = df[biased_mask & df["bias_list"].apply(
        lambda x: tuple(sorted(x)) not in seen_set)].index

    df_seen   = df.loc[sorted(set(seen_biased_idx)   | seen_unbiased)].reset_index(drop=True)
    df_unseen = df.loc[sorted(set(unseen_biased_idx) | unseen_unbiased)].reset_index(drop=True)
    print(f"Split sizes — seen (K): {len(df_seen)}, unseen (N-K): {len(df_unseen)}.")
    return df_seen, df_unseen


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def encode_reference(
    bias_groups: dict[str, list[str]],
    model: SentenceTransformer,
    device: torch.device,
) -> tuple[list[str], list[str], torch.Tensor]:
    """Pre-encode all reference sentences once."""
    all_sentences, all_labels, all_embeddings = [], [], []
    with torch.no_grad():
        for bias, sentences in bias_groups.items():
            embeddings = model.encode(sentences, convert_to_tensor=True, device=str(device))
            all_sentences.extend(sentences)
            all_labels.extend([bias] * len(sentences))
            all_embeddings.append(embeddings)
    return all_sentences, all_labels, torch.cat(all_embeddings)


def predict(
    paragraph: str,
    all_labels: list[str],
    all_embeddings: torch.Tensor,
    model: SentenceTransformer,
    device: torch.device,
    k: int,
    num_pred: int,
) -> list[str]:
    """Return predicted bias labels for a single paragraph."""
    with torch.no_grad():
        para_emb = model.encode([paragraph], convert_to_tensor=True, device=str(device))

    sims      = util.pytorch_cos_sim(para_emb, all_embeddings)[0]
    top_idx   = torch.topk(sims, k=min(k, len(all_labels))).indices.tolist()
    top_biases = [all_labels[i] for i in top_idx]

    predicted = [b for b, _ in Counter(top_biases).most_common(num_pred)]

    # 'unbiased' is mutually exclusive with all bias categories
    if "unbiased" in predicted and len(predicted) > 1:
        predicted = ["unbiased"]

    return predicted


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def jaccard(pred: set, true: set) -> float:
    if not pred and not true:
        return 1.0
    union = pred | true
    return len(pred & true) / len(union) if union else 0.0


def evaluate(
    df: pd.DataFrame,
    all_labels: list[str],
    all_embeddings: torch.Tensor,
    model: SentenceTransformer,
    device: torch.device,
    k: int,
    num_pred: int,
    split_name: str,
) -> dict:
    print(f"\nEvaluating on '{split_name}' ({len(df)} samples)...")

    exact, subset, jaccards = 0, 0, []

    for _, row in df.iterrows():
        true_set  = set(row["bias_list"])
        pred_set  = set(predict(
            row["Generated Paragraph"],
            all_labels, all_embeddings, model, device, k, num_pred,
        ))
        exact   += int(true_set == pred_set)
        subset  += int(true_set.issubset(pred_set))
        jaccards.append(jaccard(pred_set, true_set))

    n = len(df)
    results = {
        "split":          split_name,
        "n":              n,
        "exact_match":    exact / n,
        "subset_match":   subset / n,
        "avg_jaccard":    float(np.mean(jaccards)),
    }

    print(f"  Exact Match  : {results['exact_match']:.3f}")
    print(f"  Subset Match : {results['subset_match']:.3f}")
    print(f"  Avg Jaccard  : {results['avg_jaccard']:.3f}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parse training categories from CLI if provided
    if args.training_categories:
        training_cats = [c.split("+") for c in args.training_categories]
    else:
        training_cats = DEFAULT_TRAINING_CATEGORIES

    # Load data and model
    bias_groups  = load_reference_corpus(args.ref_data)
    df           = load_eval_data(args.eval_data)
    model        = SentenceTransformer(args.model_path, device=str(device))
    print(f"Model loaded from '{args.model_path}'.")

    # Pre-encode reference corpus once
    all_sentences, all_labels, all_embeddings = encode_reference(bias_groups, model, device)

    # Build splits
    df_seen, df_unseen = build_splits(df, training_cats, args.unbiased_frac, args.seed)

    # Evaluate
    results = []
    for split_df, name in [(df_seen, "K (seen)"), (df_unseen, "N-K (unseen)"), (df, "N (full)")]:
        results.append(evaluate(split_df, all_labels, all_embeddings, model, device,
                                args.k, args.num_pred, name))

    # Summary table
    print("\n" + "=" * 50)
    print(f"{'Split':<20} {'N':>6} {'Exact':>8} {'Subset':>8} {'Jaccard':>9}")
    print("-" * 50)
    for r in results:
        print(f"{r['split']:<20} {r['n']:>6} "
              f"{r['exact_match']:>8.3f} {r['subset_match']:>8.3f} {r['avg_jaccard']:>9.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
