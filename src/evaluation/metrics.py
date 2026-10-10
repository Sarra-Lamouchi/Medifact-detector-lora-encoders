import numpy as np
from sklearn.metrics import f1_score


def bootstrap_ci(labels, preds, n_boot=1000, seed=0, alpha=0.05):
    """IC bootstrap du F1 macro pour un seul jeu de prédictions."""
    labels, preds = np.asarray(labels), np.asarray(preds)
    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = [f1_score(labels[i], preds[i], average="macro")
              for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.mean(scores)), tuple(np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)]))