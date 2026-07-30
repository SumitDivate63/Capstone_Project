from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Any

def compute_metrics(y_true: List[int], y_pred: List[int], loss: float) -> Dict[str, Any]:
    """
    Computes standard evaluation benchmarks aggregating model probabilities outputs mapping towards binary class labels.
    """
    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    else:
        cm = [[0, 0], [0, 0]]
        
    counts_true_0 = y_true.count(0)
    counts_true_1 = y_true.count(1)
    counts_pred_0 = y_pred.count(0)
    counts_pred_1 = y_pred.count(1)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cm": cm,
        "counts": {
            "true_0": counts_true_0,
            "true_1": counts_true_1,
            "pred_0": counts_pred_0,
            "pred_1": counts_pred_1
        }
    }
