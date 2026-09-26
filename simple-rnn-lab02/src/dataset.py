import csv
import random
from torch.utils.data import Dataset, DataLoader
import torch
from src.preprocessing import tokenize
from src.vocabulary import Vocabulary
from src.logger import get_logger

logger = get_logger(__name__)


def load_csv(path: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def split_data(texts, labels, train_ratio, val_ratio, seed):
    pairs = list(zip(texts, labels))
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    train, val, test = pairs[:t], pairs[t:t+v], pairs[t+v:]
    logger.info("Split | train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test


class IMDbDataset(Dataset):
    def __init__(self, pairs: list[tuple], vocab: Vocabulary, max_seq_len: int):
        self.pairs = pairs
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text, label = self.pairs[idx]
        ids = self.vocab.encode(tokenize(text), self.max_seq_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)


def make_loaders(cfg: dict, vocab: Vocabulary, train, val, test) -> tuple:
    max_seq_len = cfg["data"]["max_seq_len"]
    batch_size  = cfg["training"]["batch_size"]

    train_loader = DataLoader(IMDbDataset(train, vocab, max_seq_len), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(IMDbDataset(val,   vocab, max_seq_len), batch_size=batch_size)
    test_loader  = DataLoader(IMDbDataset(test,  vocab, max_seq_len), batch_size=batch_size)

    logger.info("DataLoaders ready | batch_size=%d max_seq_len=%d", batch_size, max_seq_len)
    return train_loader, val_loader, test_loader
