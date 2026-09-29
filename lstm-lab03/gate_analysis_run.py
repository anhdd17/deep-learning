"""Task 7 — Gate analysis trên 6 câu review (3 đúng + 3 sai của model)."""
import torch
from src.config import load_config
from src.logger import setup_logging, get_logger
from src.vocabulary import Vocabulary
from src.model import SentimentModel
from src.gate_analysis import analyze_gates, print_gate_table

setup_logging("INFO")
logger = get_logger(__name__)

# Câu test — mix positive/negative rõ ràng và trường hợp khó
REVIEWS = [
    # Đúng — positive
    ("This movie was absolutely wonderful. The acting was superb and the story deeply moving.", "positive"),
    ("An outstanding film with brilliant performances. Highly recommended!", "positive"),
    ("I loved every minute of it. A masterpiece of modern cinema.", "positive"),
    # Khó — negative với negation
    ("This film is not bad at all, it is actually quite good.", "positive"),
    ("I expected to hate this movie but it was surprisingly decent.", "positive"),
    # Sai / khó — negative
    ("Terrible waste of time. The plot made no sense and the acting was painful to watch.", "negative"),
]


def main():
    cfg   = load_config("configs/lstm.yaml")
    vocab = Vocabulary.load(cfg["paths"]["artifacts"] + "/vocab.json")

    ckpt  = torch.load("artifacts/lstm_model.pt", weights_only=False, map_location="cpu")
    model = SentimentModel(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    max_len = cfg["data"]["max_len"]

    correct = 0
    for text, true_label in REVIEWS:
        rows, pred_label, prob = analyze_gates(model, text, vocab, max_len)
        status = "✓" if pred_label == true_label else "✗"
        print(f"\n{'='*60}")
        print(f"[{status}] True: {true_label}")
        print_gate_table(rows, pred_label, prob, text)
        if pred_label == true_label:
            correct += 1

    print(f"\n{'='*60}")
    print(f"Accuracy on these {len(REVIEWS)} sentences: {correct}/{len(REVIEWS)}")


if __name__ == "__main__":
    main()
