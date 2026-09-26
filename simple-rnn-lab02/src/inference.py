import torch
from src.preprocessing import tokenize
from src.vocabulary import Vocabulary
from src.model import SentimentRNN
from src.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    def __init__(self, model_path: str, vocab_path: str, cfg: dict):
        self.cfg = cfg
        self.vocab = Vocabulary.load(vocab_path)

        self.model = SentimentRNN(
            vocab_size = len(self.vocab),
            embed_dim  = cfg["model"]["embed_dim"],
            hidden_dim = cfg["model"]["hidden_dim"],
            rnn_type   = cfg["model"]["rnn_type"],
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        logger.info("Predictor ready | vocab_size=%d rnn_type=%s", len(self.vocab), cfg["model"]["rnn_type"])

    def predict(self, text: str) -> dict:
        tokens = tokenize(text)
        ids    = self.vocab.encode(tokens, self.cfg["data"]["max_seq_len"])
        x      = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, seq_len)

        with torch.no_grad():
            logit = self.model(x)
            prob  = torch.sigmoid(logit).item()

        label = "positive" if prob >= 0.5 else "negative"
        logger.debug("predict | text=%r prob=%.4f label=%s", text[:60], prob, label)
        return {"label": label, "probability": round(prob, 4)}
