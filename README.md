# Retrieval Framework for Intersectional Bias Detection

This repository contains the code, prompts, and data pipeline for our work on intersectional bias detection in both Indian-context and Western-context narratives. The system builds on contrastive retrieval, a fine-tuned SentenceTransformer (BiasRetriever) that learns to distinguish biased paragraphs from neutral ones across compound identity intersections like gender×caste or race×disability.

---

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
