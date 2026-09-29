import torch
from src.preprocessing import tokenize
from src.vocabulary import Vocabulary
from src.model import SentimentModel, build_model
from src.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    def __init__(self, model_path: str, vocab_path: str, cfg: dict = None):
        self.vocab = Vocabulary.load(vocab_path)
        ckpt = torch.load(model_path, weights_only=False, map_location="cpu")

        if isinstance(ckpt, dict) and "model_cfg" in ckpt:
            # Format mới (Lab 3): checkpoint lưu kèm model_cfg
            model_cfg       = ckpt["model_cfg"]
            state_dict      = ckpt["state_dict"]
            self.max_len    = cfg["data"]["max_len"] if cfg else 256
            self.model_name = f"{model_cfg.get('rnn_type','lstm')}-{model_cfg.get('impl','custom')}-v1"
            self.model      = SentimentModel(**model_cfg)
        else:
            # Format cũ (Lab 1): chỉ có state_dict, dùng cfg để dựng model
            state_dict   = ckpt
            self.max_len = cfg["data"].get("max_len", cfg["data"].get("max_seq_len", 100))
            model_cfg    = {
                "vocab_size":    len(self.vocab),
                "embed_dim":     cfg["model"]["embed_dim"],
                "hidden_size":   cfg["model"].get("hidden_size", cfg["model"].get("hidden_dim", 128)),
                "rnn_type":      "rnn",
                "impl":          cfg["model"].get("rnn_type", "custom"),
            }
            self.model_name = "rnn-legacy-v1"
            self.model      = SentimentModel(**model_cfg)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("Predictor ready | vocab_size=%d model=%s", len(self.vocab), self.model_name)

    def predict(self, text: str) -> dict:
        tokens  = tokenize(text)
        ids     = self.vocab.encode(tokens, self.max_len)
        length  = max(min(len(tokens), self.max_len), 1)
        x       = torch.tensor(ids,    dtype=torch.long).unsqueeze(0)   # (1, T)
        lengths = torch.tensor([length], dtype=torch.long)

        with torch.no_grad():
            logit = self.model(x, lengths)
            prob  = torch.sigmoid(logit).item()

        label = "positive" if prob >= 0.5 else "negative"
        logger.debug("predict | text=%r prob=%.4f label=%s", text[:60], prob, label)
        return {"label": label, "probability": round(prob, 4), "model": self.model_name}
