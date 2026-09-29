import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.logger import get_logger

logger = get_logger(__name__)


def evaluate(model, test_loader, device=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, lengths, y in test_loader:
            if device is not None:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            preds  = (torch.sigmoid(logits) >= 0.5).long()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.long().tolist())

    acc  = accuracy_score(all_labels,  all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels,    all_preds, zero_division=0)
    f1   = f1_score(all_labels,        all_preds, zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)

    logger.info("Test results | acc=%.4f prec=%.4f rec=%.4f f1=%.4f", acc, prec, rec, f1)
    logger.info("Confusion matrix:\n%s", cm)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}
