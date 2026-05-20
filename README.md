# NLP-NER

A study of Named Entity Recognition on Twitter data (TweeBank_NER), examining whether model errors and uncertainty align with places where human annotators disagree.

## Requirements

Install dependencies with:

```bash
pip install torch transformers seqeval scikit-learn numpy
```

## Data

All data files are in the `data/` folder:

- `train.jsonl`, `dev.jsonl`, `test.jsonl` — official TweeBank_NER splits
- `Hjalte_500.jsonl`, `mar_500.jsonl`, `val_500.jsonl` — manual annotations of the first 500 test tweets by each group member
- `test_pred_base.jsonl` — baseline BERT predictions on the test set
- `bertweet_pred.jsonl` — BERTweet predictions on the first 500 test tweets

## Running the code

All scripts should be run from the `code/` directory.

```bash
cd code
```

### 1. Train the baseline model

Fine-tunes `bert-base-cased` on the TweeBank training data and saves predictions on the test set to `test_pred_base.jsonl`.

```bash
python3 train-base.py
```

### 2. Run BERTweet inference

Runs the pretrained `bertweet-tb2-ner` model on the test set and saves predictions to `bertweet_pred_test.jsonl`.

```bash
python3 bertweet.py
```

### 3. Compute Inter-Annotator Agreement (IAA)

Computes pairwise F1 between the three annotators for each entity type (PER, LOC, ORG, MISC).

```bash
python3 IAA.py
```

### 4. Evaluate F1 scores

Computes span-level F1 (strict, unlabeled, and loose) for both models and both human annotators against the gold standard.

```bash
python3 F1.py
```

### 5. Annotation comparison and confusion matrices

Open and run `data/ann_comparison.ipynb` in Jupyter to reproduce the disagreement analysis and confusion matrix figures.

```bash
jupyter notebook data/ann_comparison.ipynb
```
