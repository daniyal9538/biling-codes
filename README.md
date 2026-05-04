# biling-codes

A PyTorch + Hugging Face library for learning BERT-style representations of clinical billing (ICD) codes with Gaps.

## Overview

This repository provides:

| Component | File | Description |
|---|---|---|
| `BillingTokenizer` | `src/models.py` | HuggingFace `PreTrainedTokenizer` subclass trained on ICD code vocabularies |
| `BillingDataset` | `src/models.py` | PyTorch `Dataset` for admission-level billing code sequences |
| `BCBERTConfig` | `src/models.py` | BERT config extended with `age_vocab_size`, `enable_age_ids`, `enable_segment_ids` |
| `BCBERT` | `src/models.py` | BERT encoder with optional age/segment embeddings and mean-pool output |
| `BCBERTForMaskedLM` | `src/models.py` | Masked language model head on top of BCBERT |
| `CSVLogCallback` | `src/models.py` | HuggingFace `TrainerCallback` that writes training/eval metrics to CSV |
| `add_gap_entries` | `src/icd_gap_utils.py` | Inserts temporal gap tokens (e.g. `GAP_30`) between ICD rows for an admission |

## Data

Expected input: a DataFrame (or Parquet file) with one ICD code per row and columns including:

| Column | Description |
|---|---|
| `subject_id` | Patient identifier |
| `hadm_id` | Hospital admission identifier |
| `chartdate` | Date of the billing entry |
| `seq_num` | Ordering within a chartdate |
| `icd_code` | Original ICD code |
| `icd_code_mapped` | Normalised/mapped code (used as model input) |
| `segment` | Visit segment index |
| `age_month` | Patient age in months at admission |

Demo data (`data/mimic_4_icd_demo.parquet`, `data/mimic_4_icd_demo_with_gaps.parquet`) is derived from [MIMIC-IV](https://physionet.org/content/mimiciv/). Access requires credentialed PhysioNet registration.

## Quickstart

```python
import pandas as pd
from src.models import BillingTokenizer, BillingDataset, BCBERTConfig, BCBERTForMaskedLM

# 1. Train tokenizer
df = pd.read_parquet("data/mimic_4_icd_demo_with_gaps.parquet")
codes = df["icd_code_mapped"].dropna().astype(str).tolist()

tokenizer = BillingTokenizer(add_cls=False, age_ids=False, segment_ids=False)
tokenizer.train_from_list(codes)
tokenizer.model_max_length = 120

# 2. Build dataset
groups = df.sort_values(["hadm_id", "chartdate", "seq_num"]).groupby("hadm_id")
data_list = groups.apply(lambda x: x.to_dict(orient="list")).tolist()
dataset = BillingDataset(data=data_list, tokenizer=tokenizer, age_column="age_month")

# 3. Create model
config = BCBERTConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=192,
    num_hidden_layers=6,
    num_attention_heads=6,
    intermediate_size=384,
    max_position_embeddings=120,
    enable_age_ids=False,
    enable_segment_ids=False,
)
model = BCBERTForMaskedLM(config)
```

## Temporal Gap Tokens

`add_gap_entries` inserts synthetic `GAP_<n>` rows between ICD entries so the model can learn time-aware representations:

```python
from src.icd_gap_utils import add_gap_entries

df_with_gaps = add_gap_entries(df, gap_list=[30, 10, 5, 1])
```

## Tutorials

Step-by-step Jupyter notebooks are in `notebooks/`:

| Notebook | Topic |
|---|---|
| `add_gap_entries_tutorial.ipynb` | How `add_gap_entries` works and its parameters |
| `billing_tokenizer_tutorial.ipynb` | Train, save, load, and use `BillingTokenizer` + `BillingDataset` |
| `bcbert_maskedlm_tutorial.ipynb` | Full MLM training pipeline with `BCBERTForMaskedLM` and HuggingFace `Trainer` |
| `tfidf_tutorial.ipynb` | Fit TF-IDF vectorizer on billing code sequences, inspect features, and save/load |

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- `transformers` ≥ 4.35
- `pandas`, `scikit-learn`

Install in an existing environment:

```bash
pip install torch transformers pandas scikit-learn
```

## Repository Structure

```
src/
    models.py           # BillingTokenizer, BillingDataset, BCBERT*, CSVLogCallback
    icd_gap_utils.py    # add_gap_entries
notebooks/
    add_gap_entries_tutorial.ipynb
    billing_tokenizer_tutorial.ipynb
    bcbert_maskedlm_tutorial.ipynb
data/                   # Parquet files (not tracked by git)
```
