"""
Bias Retriever Fine-Tuning with Triplet Loss
=============================================
Fine-tunes a SentenceTransformer model on intersectional bias triplets
using MultipleNegativesRankingLoss.

Paper: BEYOND SINGLE-AXIS FAIRNESS: LEARNING TO DETECT
INTERSECTIONAL BIASES
Authors: Vijendra Kumar Vaishya, Nihar Ranjan Sahoo

Usage:
    python train_bias_retriever.py --data_path <path_to_triplets.json> \
                                   --output_path <output_dir>
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LR = 2e-5
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a SentenceTransformer for bias-aware retrieval."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the JSON file containing triplets "
             "(keys: 'anchor', 'positive', 'negative').",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model and checkpoints.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace / SentenceTransformer model identifier. "
             f"Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs",     type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="CUDA device ID to use (e.g. '0', '1'). Set to '' for CPU.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Ensure reproducibility across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_triplets(data_path: str) -> list[InputExample]:
    """
    Load triplet examples from a JSON file.

    Expected format:
        [{"anchor": "...", "positive": "...", "negative": "..."}, ...]

    Returns:
        List of InputExample objects for SentenceTransformer training.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        triplets = json.load(f)
    print(f"Loaded {len(triplets)} triplets from '{data_path}'.")
    return [
        InputExample(texts=[t["anchor"], t["positive"], t["negative"]])
        for t in triplets
    ]


def train(args: argparse.Namespace) -> None:
    # -----------------------------------------------------------------------
    # Device setup
    # -----------------------------------------------------------------------
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_samples = load_triplets(args.data_path)
    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=args.batch_size
    )

    # -----------------------------------------------------------------------
    # Model + Loss
    # -----------------------------------------------------------------------
    model = SentenceTransformer(args.model_name, device=str(device))
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * 0.1)  # 10 % of total steps
    checkpoint_dir = os.path.join(args.output_path, "checkpoints")

    print(
        f"Training '{args.model_name}' | "
        f"epochs={args.epochs} | batch_size={args.batch_size} | lr={args.lr}"
    )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_class=AdamW,
        optimizer_params={"lr": args.lr, "eps": 1e-8},
        output_path=args.output_path,
        show_progress_bar=True,
        checkpoint_path=checkpoint_dir,
        checkpoint_save_steps=len(train_dataloader),  # once per epoch
        checkpoint_save_total_limit=0,                
    )

    print(f"Training complete. Model saved to '{args.output_path}'.")


if __name__ == "__main__":
    args = parse_args()
    train(args)