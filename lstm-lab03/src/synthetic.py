import torch
from torch.utils.data import Dataset, DataLoader


class FirstTokenDataset(Dataset):
    """
    Synthetic long-range memory task.

    Mỗi sample có dạng: [signal, noise, noise, ..., noise]
    - signal = 1 nếu label=0 (NEG), signal = 2 nếu label=1 (POS)
    - noise = token ngẫu nhiên từ [3, 3+n_noise)
    - Vocab: PAD=0, NEG_TOKEN=1, POS_TOKEN=2, noise=3..102

    Model muốn đúng buộc phải nhớ token đầu qua seq_len-1 bước nhiễu.
    """

    POS_TOKEN = 2
    NEG_TOKEN = 1

    def __init__(self, n_samples: int, seq_len: int, n_noise: int = 100, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.labels  = torch.randint(0, 2, (n_samples,), generator=g)   # 0=NEG, 1=POS
        noise        = torch.randint(3, 3 + n_noise, (n_samples, seq_len), generator=g)
        # token đầu = tín hiệu duy nhất xác định label
        noise[:, 0]  = self.labels + 1   # label=0→token 1 (NEG), label=1→token 2 (POS)
        self.x       = noise
        self.lengths = torch.full((n_samples,), seq_len, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.x[i], self.lengths[i], self.labels[i].float()


def make_synthetic_loaders(n_train: int, n_val: int, seq_len: int,
                           batch_size: int = 64, n_noise: int = 100,
                           seed: int = 0):
    train_ds = FirstTokenDataset(n_train, seq_len, n_noise=n_noise, seed=seed)
    val_ds   = FirstTokenDataset(n_val,   seq_len, n_noise=n_noise, seed=seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    return train_loader, val_loader
