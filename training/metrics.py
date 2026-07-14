from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List

def compute_metrics(y_true: List[int], y_pred: List[int], loss: float) -> Dict[str, float]:
    """
    Computes standard evaluation benchmarks aggregating model probabilities outputs mapping towards binary class labels.
    """
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
