import torch
from src.config import load_config
from src.logger import setup_logging, get_logger
from src.vocabulary import Vocabulary
from src.dataset import load_csv, split_data, IMDbDataset
from src.model import SentimentRNN
from src.experiment import run_experiment
from torch.utils.data import DataLoader

setup_logging("INFO")
logger = get_logger(__name__)
cfg    = load_config("configs/default.yaml")
torch.manual_seed(cfg["training"]["seed"])

# data — build vocab một lần
texts, labels = load_csv(cfg["paths"]["data"] + "/train.csv")
train_pairs, val_pairs, _ = split_data(
    texts, labels,
    cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["training"]["seed"]
)
vocab = Vocabulary.load(cfg["paths"]["artifacts"] + "/vocab.json")


def make_loaders_fn(seq_len):
    """Tạo loader với seq_len thay đổi, batch size nhỏ hơn để chạy nhanh."""
    bs = 64
    train_loader = DataLoader(IMDbDataset(train_pairs, vocab, seq_len), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(IMDbDataset(val_pairs,   vocab, seq_len), batch_size=bs)
    return train_loader, val_loader


all_results = []

for rnn_type in ["custom", "pytorch"]:
    cfg["model"]["rnn_type"] = rnn_type

    def make_model_fn():
        return SentimentRNN(len(vocab), cfg["model"]["embed_dim"], cfg["model"]["hidden_dim"], rnn_type)

    results = run_experiment(make_loaders_fn, make_model_fn, cfg)
    all_results.extend(results)

# print summary table
print()
print(f"{'seq_len':>8} {'rnn_type':>10} {'val_acc':>8} {'grad_W_xh':>12} {'grad_W_hh':>12}")
print("-" * 56)
for r in all_results:
    print(f"{r['seq_len']:>8} {r['rnn_type']:>10} {r['val_acc']:>8.4f} {r['grad_W_xh']:>12.6f} {r['grad_W_hh']:>12.6f}")
