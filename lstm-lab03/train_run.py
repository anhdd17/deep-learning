"""Entry point: train một model theo config được chỉ định."""
import argparse
import random
import numpy as np
import torch

from src.config     import load_config
from src.logger     import setup_logging, get_logger
from src.vocabulary import Vocabulary
from src.dataset    import load_csv, split_data, make_loaders
from src.model      import build_model
from src.train      import train
from src.evaluate   import evaluate

logger = get_logger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lstm.yaml")
    parser.add_argument("--artifact-name", default=None,
                        help="Tên file checkpoint (default: <model.type>_model.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"])

    seed = cfg["train"]["seeds"][0]
    set_seed(seed)
    logger.info("Config: %s | seed=%d", args.config, seed)

    # --- Data ---
    texts, labels = load_csv(cfg["paths"]["data"] + "/train.csv")
    train_p, val_p, test_p = split_data(
        texts, labels,
        cfg["data"]["train_ratio"],
        cfg["data"]["val_ratio"],
        seed=seed,
    )
    vocab = Vocabulary.load(cfg["paths"]["artifacts"] + "/vocab.json")
    train_loader, val_loader, test_loader = make_loaders(cfg, vocab, train_p, val_p, test_p)

    # --- Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # --- Model ---
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Params: %d", n_params)

    # model_cfg được lưu kèm checkpoint để Predictor tự dựng lại kiến trúc
    m = cfg["model"]
    model_cfg = {
        "vocab_size":    cfg["data"]["vocab_size"],
        "embed_dim":     m["embed_dim"],
        "hidden_size":   m["hidden_size"],
        "rnn_type":      m.get("type", "lstm"),
        "impl":          m.get("impl", "custom"),
        "bidirectional": m.get("bidirectional", False),
        "num_layers":    m.get("num_layers", 1),
        "dropout":       m.get("dropout", 0.0),
        "forget_bias":   m.get("forget_bias", 1.0),
    }

    artifact_name = args.artifact_name or f"{m.get('type', 'lstm')}_model.pt"
    save_path = cfg["paths"]["artifacts"] + "/" + artifact_name

    # --- Train ---
    train(model, train_loader, val_loader, cfg, save_path, model_cfg, device)

    # --- Evaluate ---
    ckpt = torch.load(save_path, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    results = evaluate(model, test_loader, device)
    logger.info("Final test | acc=%.4f f1=%.4f", results["accuracy"], results["f1"])


if __name__ == "__main__":
    main()
