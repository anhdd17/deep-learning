"""
Task 8 — Ablation study: 9 cấu hình × 3 seeds (trừ I chạy 1 lần).
"""
import random
import numpy as np
import torch

from src.config import load_config
from src.logger import setup_logging, get_logger
from src.vocabulary import Vocabulary
from src.dataset import load_csv, split_data, make_loaders
from src.ablation import run_ablation_config

setup_logging("INFO")
logger = get_logger(__name__)

SEEDS = [42, 123, 777]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_tfidf_lr(train_pairs, test_pairs):
    """Config I — TF-IDF + Logistic Regression (sklearn)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    train_texts, train_labels = zip(*train_pairs)
    test_texts,  test_labels  = zip(*test_pairs)

    vec = TfidfVectorizer(max_features=20000, sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, train_labels)
    acc = accuracy_score(test_labels, clf.predict(X_test))
    return acc


def main():
    device = get_device()
    logger.info("Device: %s", device)

    cfg = load_config("configs/lstm.yaml")

    texts, labels = load_csv(cfg["paths"]["data"] + "/train.csv")
    train_p, val_p, test_p = split_data(
        texts, labels,
        cfg["data"]["train_ratio"],
        cfg["data"]["val_ratio"],
        seed=42,
    )
    vocab = Vocabulary.load(cfg["paths"]["artifacts"] + "/vocab.json")
    train_loader, val_loader, test_loader = make_loaders(cfg, vocab, train_p, val_p, test_p)

    base_model = dict(
        vocab_size    = cfg["data"]["vocab_size"],
        embed_dim     = cfg["model"]["embed_dim"],
        hidden_size   = cfg["model"]["hidden_size"],
        rnn_type      = "lstm",
        impl          = "custom",
        bidirectional = False,
        num_layers    = 1,
        dropout       = 0.0,
        forget_bias   = 1.0,
    )
    base_train = dict(cfg["train"])
    base_train["use_clip"] = True

    CONFIGS = [
        dict(name="A — Baseline",
             model_kwargs=base_model,
             train_cfg_override={},
             use_lengths=True),

        dict(name="B — No masking",
             model_kwargs=base_model,
             train_cfg_override={},
             use_lengths=False),

        dict(name="C — forget_bias=0",
             model_kwargs={**base_model, "forget_bias": 0.0},
             train_cfg_override={},
             use_lengths=True),

        dict(name="D — No grad clip",
             model_kwargs=base_model,
             train_cfg_override={"use_clip": False},
             use_lengths=True),

        dict(name="E — impl=pytorch",
             model_kwargs={**base_model, "impl": "pytorch"},
             train_cfg_override={},
             use_lengths=True),

        dict(name="F — GRU pytorch",
             model_kwargs={**base_model, "rnn_type": "gru", "impl": "pytorch"},
             train_cfg_override={},
             use_lengths=True),

        dict(name="G — Bidirectional",
             model_kwargs={**base_model, "rnn_type": "lstm", "impl": "pytorch",
                           "bidirectional": True},
             train_cfg_override={},
             use_lengths=True),

        dict(name="H — 2 layers dropout",
             model_kwargs={**base_model, "rnn_type": "lstm", "impl": "pytorch",
                           "num_layers": 2, "dropout": 0.3},
             train_cfg_override={},
             use_lengths=True),
    ]

    results = []
    for c in CONFIGS:
        r = run_ablation_config(
            name             = c["name"],
            model_kwargs     = c["model_kwargs"],
            train_cfg_override = c["train_cfg_override"],
            train_loader     = train_loader,
            val_loader       = val_loader,
            test_loader      = test_loader,
            base_train_cfg   = base_train,
            device           = device,
            seeds            = SEEDS,
            use_lengths      = c["use_lengths"],
        )
        results.append(r)

    # --- Config I: TF-IDF + LR ---
    logger.info("Running I — TF-IDF + LogisticRegression...")
    tfidf_acc = run_tfidf_lr(train_p, test_p)
    logger.info("I — TF-IDF+LR | acc=%.4f", tfidf_acc)
    results.append({"name": "I — TF-IDF + LR", "mean": tfidf_acc, "std": 0.0,
                    "accs": [tfidf_acc], "epoch_time": None})

    # --- Print table ---
    print(f"\n{'Config':<22} {'Acc mean':>10} {'± std':>8} {'Epoch time':>12}")
    print("-" * 58)
    for r in results:
        ep = f"{r['epoch_time']:.1f}s" if r["epoch_time"] else "—"
        std_str = f"±{r['std']:.4f}" if r["std"] > 0 else "—"
        print(f"{r['name']:<22} {r['mean']:>10.4f} {std_str:>8} {ep:>12}")

    return results


if __name__ == "__main__":
    main()
