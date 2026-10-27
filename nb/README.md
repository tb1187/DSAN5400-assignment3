# Naive Bayes Speech Classifier

A from-scratch multinomial Naive Bayes classifier with Lidstone smoothing, packaged as a Python project. It trains on presidential speeches by Kennedy and Johnson and predicts the author of unlabeled speeches.

DSAN 5400, Assignment 3.

## Setup

From this directory:

```bash
conda env create -f environment.yml
conda activate dsan5400
pip install -e ".[dev]"
```

## Usage

```bash
python -m nb.bin.main -f data
```

| Flag | Description |
|---|---|
| `-f`, `--indir` | Data directory (required) |
| `-a`, `--alpha` | Lidstone smoothing parameter (default `0.1`) |

The script prints a predicted author for each unlabeled speech, followed by accuracy, F1 score, and a confusion matrix. Authors are encoded as `0` = Kennedy, `1` = Johnson.

### Using the class directly

```python
from nb.nb import NaiveBayes

model = NaiveBayes(alpha=0.1).train(train_df)   
# DataFrame with "text" and "author" columns
predictions = model.test(test_df)
```

After training, `model.priors`, `model.likelihoods` (shape `n_classes × |V|`), and `model.vocabulary` hold the fitted estimates.


