# Retrieval Framework for Intersectional Bias Detection

This repository contains the code, prompts, and data pipeline for our work on intersectional bias detection in both Indian-context and Western-context narratives. The system builds on contrastive retrieval, a fine-tuned SentenceTransformer (BiasRetriever) that learns to distinguish biased paragraphs from neutral ones across compound identity intersections like gender×caste or race×disability.

---

## What's in here

```
.
├── train_bias_retriever.py        # Fine-tuning script (contrastive triplet loss)
├── dataset/
│   ├── build_triplets.py          # Triplet construction pipeline
│   ├── sample_indic.json          # 10 example samples from Indic-Intersect
│   └── sample_western.json        # 10 example samples from Western-Intersect
├── prompts/
│   └── prompts.md                 # All LLM prompts used for data generation
├── eval/
│   └── evaluate.py                # Evaluation script (in-domain + cross-domain)
├── requirements.txt
└── LICENSE
```

---

## Dataset

**Indic-Intersect** and **Western-Intersect** are two corpora of ~3,700 paragraphs each, covering pairwise and three-way intersections of social bias categories. Each paragraph is labeled with a compound intersectional category (e.g., `gender×caste`, `race×culture×social`).

### Triplet Format

Training uses triplets stored as JSON arrays:

```json
[
  {
    "anchor":   "Paragraph exhibiting intersectional bias (e.g., gender × caste).",
    "positive": "Another paragraph showing the same or closely related bias.",
    "negative": "A neutral or unrelated paragraph, or a counterfactual rewrite."
  }
]
```

Each entry must contain exactly the three keys `anchor`, `positive`, `negative`. The full dataset will be released on HuggingFace upon paper acceptance.

### Bias Categories

| Corpus | Bias Axes |
|---|---|
| **Indic-Intersect** | gender, caste, religion, age, socioeconomic, physical-appearance |
| **Western-Intersect** | gender, race, disability, body, culture, social, victimization |

---

## Setup

Python 3.9+ recommended.

```bash
pip install -r requirements.txt
```

---

## Training BiasRetriever

```bash
python train_bias_retriever.py \
  --data_path   dataset/sample_indic.json \
  --output_path outputs/bias_retriever_indic
```

Full options:

| Argument        | Default                                   | Description                                     |
|-----------------|-------------------------------------------|-------------------------------------------------|
| `--data_path`   | *(required)*                              | Path to triplets JSON                           |
| `--output_path` | *(required)*                              | Where to save the model + checkpoints           |
| `--model_name`  | `sentence-transformers/all-MiniLM-L6-v2`  | Any HuggingFace SentenceTransformer             |
| `--epochs`      | `10`                                      | Training epochs                                 |
| `--batch_size`  | `32`                                      | Larger batch = more in-batch negatives          |
| `--lr`          | `2e-5`                                    | Learning rate for AdamW                         |
| `--seed`        | `42`                                      | Random seed                                     |
| `--gpu_id`      | `"0"`                                     | CUDA device; pass `""` for CPU                  |

The script uses `MultipleNegativesRankingLoss`, which treats all other samples in the batch as additional implicit negatives. Checkpoints are saved at the end of each epoch.

---

## Triplet Generation Strategies

We use four LLM-augmented strategies to construct training data (detailed in the paper appendix):

| Strategy | Anchor | Positive | Negative | Purpose |
|---|---|---|---|---|
| **Dual-LLM** | Biased paragraph | LLM-generated | LLM-generated | Fully synthetic triplets |
| **LLM-Positive + Counterfactual** | Biased paragraph | LLM-generated | Counterfactual rewrite | Hard negatives close to the positive |
| **Mined-Positive + Counterfactual** | Biased paragraph | Semantically retrieved | Counterfactual rewrite | Maximum difficulty negatives |
| **Neutral Anchor Paraphrasing** | Unbiased paragraph | LLM paraphrase | Retrieved unbiased | Teaches neutral-neutral similarity |

All prompts are in `prompts/prompts.md`.

---

## Evaluation

```bash
python eval/evaluate.py \
  --model_path  outputs/bias_retriever_indic \
  --test_data   dataset/sample_indic.json \
  --k           15
```

The script reports retrieval-based metrics across in-domain (seen categories) and zero-shot (unseen intersectional categories) splits.

---

## Reproducing the Paper Results

We train and evaluate across three predefined category subsets (C1, C2, C3) for the generalization experiments. To replicate:

```bash
# Train on subset C1, evaluate on held-out categories
python train_bias_retriever.py \
  --data_path dataset/triplets_C1_indic.json \
  --output_path outputs/C1_indic

python eval/evaluate.py \
  --model_path outputs/C1_indic \
  --test_data  dataset/test_N_minus_C1_indic.json
```

The full dataset splits will be released alongside the camera-ready version.

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@inproceedings{yourpaper2026,
  title     = {Your Paper Title},
  author    = {Author Names},
  booktitle = {Proceedings of ACL 2026},
  year      = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
