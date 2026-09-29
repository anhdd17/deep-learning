import json
from collections import Counter
from src.logger import get_logger

logger = get_logger(__name__)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


class Vocabulary:
    def __init__(self):
        self.token2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.id2token = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}

    def build(self, token_lists: list[list[str]], vocab_size: int, min_freq: int) -> None:
        counter = Counter(tok for tokens in token_lists for tok in tokens)
        most_common = counter.most_common(vocab_size - 2)  # -2 for PAD + UNK
        kept = [(tok, freq) for tok, freq in most_common if freq >= min_freq]

        for tok, _ in kept:
            idx = len(self.token2id)
            self.token2id[tok] = idx
            self.id2token[idx] = tok

        logger.info("Vocab built | size=%d min_freq=%d", len(self.token2id), min_freq)

    def encode(self, tokens: list[str], max_seq_len: int) -> list[int]:
        ids = [self.token2id.get(tok, UNK_ID) for tok in tokens[:max_seq_len]]
        ids += [PAD_ID] * (max_seq_len - len(ids))  # pad if too short
        return ids

    def unk_rate(self, token_lists: list[list[str]]) -> float:
        total = unk = 0
        for tokens in token_lists:
            for tok in tokens:
                total += 1
                if tok not in self.token2id:
                    unk += 1
        rate = unk / total if total else 0.0
        return rate

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.token2id, f)
        logger.info("Vocab saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls()
        with open(path) as f:
            vocab.token2id = json.load(f)
        vocab.id2token = {v: k for k, v in vocab.token2id.items()}
        logger.info("Vocab loaded from %s | size=%d", path, len(vocab.token2id))
        return vocab

    def __len__(self) -> int:
        return len(self.token2id)
