# DistilBERT Sentiment Classifier — IMDB Movie Reviews

Fine-tuned DistilBERT for binary sentiment classification (positive/negative) on IMDB movie reviews, using HuggingFace Transformers and PyTorch.

## Overview

This project fine-tunes `distilbert-base-uncased` on a 10,000-review sample of the IMDB dataset to classify movie reviews as positive or negative. It covers the full workflow — EDA, text preprocessing, tokenization, fine-tuning, evaluation, and local deployment.

## Results

| Metric | Score |
|---|---|
| Accuracy | 91.45% |
| F1 (weighted) | 91.45% |

Achieved in a single training epoch.

## Workflow

1. **EDA & Preprocessing**
   - Loaded and sampled 10,000 reviews from the IMDB dataset
   - Cleaned text (lowercasing, HTML tag removal)
   - Analyzed word/character count distributions to determine sequence length

2. **Tokenization**
   - Built a custom PyTorch `Dataset` class to tokenize reviews and format them into `input_ids`, `attention_mask`, and `labels`
   - Used `max_length=512` with truncation and padding

3. **Model Fine-Tuning**
   - Loaded `distilbert-base-uncased` via `AutoModelForSequenceClassification` for binary classification
   - Trained using the HuggingFace `Trainer` API with:
     - Batch size: 16
     - Learning rate: 2e-5
     - Epochs: 1
     - Weight decay: 0.01
   - Tracked accuracy and F1 via a custom `compute_metrics` function

4. **Evaluation**
   - Achieved 91.45% accuracy and F1 on the held-out test split

5. **Deployment & Inference**
   - Saved the fine-tuned model and tokenizer locally
   - Reloaded the model with `AutoModelForSequenceClassification.from_pretrained`
   - Ran inference two ways:
     - A custom `get_prediction()` function using softmax over raw logits
     - HuggingFace's `pipeline("text-classification", ...)` for quick inference

## Tech Stack

- Python, PyTorch
- HuggingFace Transformers (`AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`)
- scikit-learn (accuracy, F1 scoring)
- pandas, matplotlib (EDA)

## Example Prediction

```python
text = "The score is haunting yet otherworldly—a perfect fit for the subject matter. This is not a mindless action film; instead, it is an intimate, philosophical drama that challenges the viewer."
get_prediction(text)
# {'sentiment': 'positive', 'prob': 0.9738667607307434}
```

## Repository Structure

```
├── distilbert_imdb_sentiment.ipynb   # Full notebook: EDA, training, evaluation, inference
└── README.md
```

## Dataset

[IMDB Dataset](https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/refs/heads/master/IMDB-Dataset.csv) — 50,000 labeled movie reviews (10,000-review subset used for this project).
