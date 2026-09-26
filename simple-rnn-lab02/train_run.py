import torch
from src.config import load_config
from src.logger import setup_logging, get_logger
from src.preprocessing import tokenize
from src.vocabulary import Vocabulary
from src.dataset import load_csv, split_data, make_loaders
from src.model import SentimentRNN
from src.train import train

cfg = load_config("configs/default.yaml")
setup_logging(cfg["logging"]["level"])
logger = get_logger(__name__)

torch.manual_seed(cfg["training"]["seed"])

# data
texts, labels = load_csv(cfg["paths"]["data"] + "/train.csv")
train_pairs, val_pairs, test_pairs = split_data(
    texts, labels,
    cfg["data"]["train_ratio"],
    cfg["data"]["val_ratio"],
    cfg["training"]["seed"],
)

# vocab
train_tokens = [tokenize(t) for t, _ in train_pairs]
vocab = Vocabulary()
vocab.build(train_tokens, cfg["data"]["vocab_size"], cfg["data"]["min_freq"])
vocab.save(cfg["paths"]["artifacts"] + "/vocab.json")

# loaders
train_loader, val_loader, _ = make_loaders(cfg, vocab, train_pairs, val_pairs, test_pairs)

# model
model = SentimentRNN(
    vocab_size  = len(vocab),
    embed_dim   = cfg["model"]["embed_dim"],
    hidden_dim  = cfg["model"]["hidden_dim"],
    rnn_type    = cfg["model"]["rnn_type"],
)
logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

# train
train(model, train_loader, val_loader, cfg, save_path=cfg["paths"]["artifacts"] + "/model.pt")
