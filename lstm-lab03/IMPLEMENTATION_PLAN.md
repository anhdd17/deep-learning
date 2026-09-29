# Implementation Plan — LSTM Sentiment Classification (Lab 3)

Base: `simple-rnn-lab02/`. Target: `lstm-lab03/` là project độc lập, kế thừa code Lab 1, mở rộng thêm LSTM và các thí nghiệm.

---

## 0. Tổng quan thay đổi so với Lab 1

| Thành phần | Trạng thái | Lý do |
|---|---|---|
| `src/preprocessing.py` | Copy nguyên | Không đổi |
| `src/vocabulary.py` | Copy nguyên | Không đổi |
| `src/logger.py` | Copy nguyên | Không đổi |
| `src/config.py` | Copy nguyên | Không đổi |
| `src/dataset.py` | **Sửa** | Thêm `lengths` vào `__getitem__` |
| `src/rnn.py` | **Sửa** | Thêm `lengths` param + masking |
| `src/model.py` | **Viết lại** | `SentimentModel` + `build_model` factory |
| `src/train.py` | **Sửa** | Handle `lengths`, early stopping, grad norm log |
| `src/evaluate.py` | **Sửa** | Handle `lengths` |
| `src/inference.py` | **Sửa** | Load `model_cfg` từ checkpoint |
| `api/main.py` | **Sửa nhẹ** | Dùng `build_model` từ checkpoint |
| `configs/default.yaml` | **Sửa** | Thêm `model.type`, `model.impl`, etc. |
| `src/lstm.py` | **MỚI** | `MyLSTMCell` + `MyLSTM` |
| `src/synthetic.py` | **MỚI** | `FirstTokenDataset` |
| `src/gradient_probe.py` | **MỚI** | `per_position_grad` |
| `src/gate_analysis.py` | **MỚI** | Gate visualization |
| `src/ablation.py` | **MỚI** | Ablation runner |
| `configs/lstm.yaml` | **MỚI** | LSTM config |
| `tests/test_lstm.py` | **MỚI** | So khớp với `nn.LSTM` |
| `tests/test_padding.py` | **MỚI** | Padding invariance |
| `tests/test_synthetic.py` | **MỚI** | FirstTokenDataset |
| `ablation_run.py` | **MỚI** | Entry point ablation |
| `synthetic_run.py` | **MỚI** | Entry point synthetic task |
| `train_run.py` | Adapt | Dùng `build_model`, handle `lengths` |
| `REPORT.md` | **MỚI** | Kết quả thí nghiệm |

---

## 1. Cấu trúc thư mục đích

```
lstm-lab03/
├── configs/
│   ├── default.yaml
│   └── lstm.yaml
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      ← copy
│   ├── vocabulary.py         ← copy
│   ├── logger.py             ← copy
│   ├── config.py             ← copy
│   ├── rnn.py                ← sửa
│   ├── lstm.py               ← mới
│   ├── model.py              ← viết lại
│   ├── dataset.py            ← sửa
│   ├── train.py              ← sửa
│   ├── evaluate.py           ← sửa
│   ├── inference.py          ← sửa
│   ├── synthetic.py          ← mới
│   ├── gradient_probe.py     ← mới
│   ├── gate_analysis.py      ← mới
│   └── ablation.py           ← mới
├── api/
│   └── main.py               ← sửa nhẹ
├── tests/
│   ├── __init__.py
│   ├── conftest.py           ← mới (fixtures)
│   ├── test_preprocessing.py ← copy (10 tests)
│   ├── test_model.py         ← sửa nhẹ (7 tests vẫn pass)
│   ├── test_api.py           ← sửa nhẹ (7 tests vẫn pass)
│   ├── test_lstm.py          ← mới
│   ├── test_padding.py       ← mới
│   └── test_synthetic.py     ← mới
├── data/
│   └── raw/                  ← symlink hoặc copy từ Lab 1
├── artifacts/                ← tạo rỗng, sẽ populate khi train
├── ablation_run.py           ← mới
├── synthetic_run.py          ← mới
├── train_run.py              ← adapt
├── experiment_run.py         ← adapt
├── requirements.txt
└── REPORT.md
```

---

## 2. Setup ban đầu

```bash
# 1. Tạo thư mục
mkdir -p lstm-lab03/{src,api,tests,configs,artifacts,data/raw,logs,runs}

# 2. Copy nguyên các file giữ nguyên
cp simple-rnn-lab02/src/{preprocessing,vocabulary,logger,config,__init__}.py lstm-lab03/src/
cp simple-rnn-lab02/tests/__init__.py lstm-lab03/tests/
cp simple-rnn-lab02/requirements.txt lstm-lab03/

# 3. Copy data (hoặc symlink)
cp simple-rnn-lab02/data/raw/*.csv lstm-lab03/data/raw/
# HOẶC: ln -s ../simple-rnn-lab02/data/raw lstm-lab03/data/raw

# 4. Copy vocab (reuse, không rebuild)
cp simple-rnn-lab02/artifacts/vocab.json lstm-lab03/artifacts/

# 5. requirements.txt — thêm matplotlib nếu chưa có
echo "matplotlib" >> lstm-lab03/requirements.txt
```

---

## 3. TASK 1 — `src/dataset.py` (sửa)

**Thay đổi duy nhất:** `__getitem__` trả về thêm `length`.

```python
# IMDbDataset.__getitem__
def __getitem__(self, idx):
    text, label = self.pairs[idx]
    tokens = tokenize(text)
    ids = self.vocab.encode(tokens, self.max_seq_len)
    # length = số token thật sau truncate, tối thiểu 1
    length = max(min(len(tokens), self.max_seq_len), 1)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(length, dtype=torch.long),
        torch.tensor(label, dtype=torch.float),
    )
```

**Config key:** Lab 1 dùng `cfg["data"]["max_seq_len"]`, Lab 3 đổi thành `cfg["data"]["max_len"]`. Cập nhật `make_loaders` tương ứng.

**Lưu ý:** DataLoader sẽ batch tự động → `x: (B, T)`, `lengths: (B,)`, `y: (B,)`.

---

## 4. TASK 2 — `src/rnn.py` (sửa)

Thêm `lengths` parameter với masking để hidden state không bị nhiễm PAD:

```python
class MySimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b    = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, I)
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)

        for t in range(T):
            h_new = torch.tanh(x[:, t, :] @ self.W_xh + h @ self.W_hh + self.b)
            if lengths is not None:
                # mask = 1 nếu sample b còn token thật tại bước t
                mask = (t < lengths).float().unsqueeze(1).to(x.device)  # (B, 1)
                h = mask * h_new + (1 - mask) * h
            else:
                h = h_new

        return h  # (B, H) — h tại token thật cuối của mỗi sample
```

**Tại sao `lengths=None` default:** Để 7 test cũ trong `test_model.py` gọi `model(x)` không bị lỗi.

---

## 5. TASK 3 — `src/lstm.py` (MỚI)

### 5.1. `MyLSTMCell`

```python
import torch
import torch.nn as nn
import math


class MyLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, forget_bias: float = 1.0):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.W_x = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.b   = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters(forget_bias)

    def reset_parameters(self, forget_bias: float):
        H = self.hidden_size
        std = 1.0 / math.sqrt(H)
        nn.init.uniform_(self.W_x, -std, std)
        nn.init.uniform_(self.W_h, -std, std)
        nn.init.zeros_(self.b)
        # forget gate bias (vị trí H:2H theo thứ tự i,f,g,o)
        with torch.no_grad():
            self.b[H : 2 * H].fill_(forget_bias)

    def forward(self, x_t, state):
        # x_t: (B, I), state = (h_prev, c_prev) mỗi cái (B, H)
        h_prev, c_prev = state
        gates = x_t @ self.W_x + h_prev @ self.W_h + self.b  # (B, 4H)
        i, f, g, o = gates.chunk(4, dim=-1)  # mỗi (B, H)

        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        g = torch.tanh(g)      # candidate cell
        o = torch.sigmoid(o)   # output gate

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        gates_dict = {"i": i, "f": f, "g": g, "o": o}
        return h_t, c_t, gates_dict
```

**Thứ tự gate bắt buộc `i, f, g, o`** — trùng với PyTorch để test so khớp hoạt động.  
**`b[H:2H] = forget_bias`** — forget gate nằm ở chunk thứ 2 (index 1).  
**Init**: `uniform(-1/sqrt(H), 1/sqrt(H))` giống PyTorch mặc định.

### 5.2. `MyLSTM`

```python
class MyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, forget_bias: float = 1.0):
        super().__init__()
        self.cell = MyLSTMCell(input_size, hidden_size, forget_bias)
        self.hidden_size = hidden_size

    def forward(self, x, lengths=None, return_gates=False):
        # x: (B, T, I)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_size)
        c = x.new_zeros(B, self.hidden_size)

        all_h = []
        gate_lists = {"i": [], "f": [], "g": [], "o": []} if return_gates else None

        for t in range(T):
            h_new, c_new, gates = self.cell(x[:, t, :], (h, c))

            if lengths is not None:
                mask = (t < lengths).float().unsqueeze(1).to(x.device)  # (B, 1)
                h = mask * h_new + (1 - mask) * h
                c = mask * c_new + (1 - mask) * c
            else:
                h, c = h_new, c_new

            all_h.append(h)
            if return_gates:
                for k in gate_lists:
                    gate_lists[k].append(gates[k])

        outputs = torch.stack(all_h, dim=1)  # (B, T, H)

        if return_gates:
            gates_out = {k: torch.stack(v, dim=1) for k, v in gate_lists.items()}
            return outputs, (h, c), gates_out

        return outputs, (h, c)
```

**Masking logic:** Khi `t >= lengths[b]`, sample `b` đã hết token thật → giữ nguyên `h`, `c` từ bước trước. Sau vòng lặp, `h[b]` chính xác là state tại token thật cuối của sample `b`.

### 5.3. Copy weights từ `nn.LSTM` — chú ý transpose

| nn.LSTM | MyLSTM | Shape |
|---|---|---|
| `weight_ih_l0` | `W_x.T` | nn: `(4H, I)` → my: `(I, 4H)` |
| `weight_hh_l0` | `W_h.T` | nn: `(4H, H)` → my: `(H, 4H)` |
| `bias_ih_l0 + bias_hh_l0` | `b` | nn có 2 bias → cộng lại |

```python
def _copy_from_torch(my: MyLSTM, ref: nn.LSTM):
    with torch.no_grad():
        my.cell.W_x.copy_(ref.weight_ih_l0.T)
        my.cell.W_h.copy_(ref.weight_hh_l0.T)
        my.cell.b.copy_(ref.bias_ih_l0 + ref.bias_hh_l0)
```

---

## 6. TASK 4 — `src/model.py` (viết lại)

### 6.1. `SentimentModel`

```python
from torch.nn.utils.rnn import pack_padded_sequence

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 rnn_type="lstm", impl="custom",
                 bidirectional=False, num_layers=1,
                 dropout=0.0, forget_bias=1.0, pad_idx=0):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn_type     = rnn_type
        self.impl         = impl
        self.bidirectional = bidirectional
        self.num_layers   = num_layers

        num_dir = 2 if bidirectional else 1

        if impl == "custom":
            # custom chỉ hỗ trợ 1 layer, unidirectional
            if rnn_type == "lstm":
                self.rnn = MyLSTM(embed_dim, hidden_size, forget_bias=forget_bias)
            elif rnn_type == "rnn":
                self.rnn = MySimpleRNN(embed_dim, hidden_size)
            else:
                raise ValueError(f"custom impl không hỗ trợ rnn_type={rnn_type}")
        elif impl == "pytorch":
            rnn_kwargs = dict(
                input_size=embed_dim, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            if rnn_type == "lstm":
                self.rnn = nn.LSTM(**rnn_kwargs)
            elif rnn_type == "rnn":
                self.rnn = nn.RNN(**rnn_kwargs)
            elif rnn_type == "gru":
                self.rnn = nn.GRU(**rnn_kwargs)
            else:
                raise ValueError(f"rnn_type không hợp lệ: {rnn_type}")
        else:
            raise ValueError(f"impl phải là 'custom' hoặc 'pytorch'")

        self.fc = nn.Linear(hidden_size * num_dir, 1)

    def forward(self, input_ids, lengths=None):
        emb    = self.embedding(input_ids)     # (B, T, E)
        h_last = self._run_rnn(emb, lengths)   # (B, H * num_dir)
        return self.fc(h_last).squeeze(1)      # (B,) logits

    def forward_from_embeddings(self, emb, lengths=None):
        """Entry point cho gradient_probe — nhận embedding trực tiếp."""
        h_last = self._run_rnn(emb, lengths)
        return self.fc(h_last).squeeze(1)

    def _run_rnn(self, emb, lengths):
        if self.impl == "custom":
            if self.rnn_type == "lstm":
                _, (h, _) = self.rnn(emb, lengths)
                return h
            else:
                return self.rnn(emb, lengths)
        else:
            # pytorch — dùng pack_padded_sequence nếu có lengths
            if lengths is not None:
                packed = pack_padded_sequence(
                    emb, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                if self.rnn_type == "lstm":
                    _, (h_n, _) = self.rnn(packed)
                else:
                    _, h_n = self.rnn(packed)
            else:
                if self.rnn_type == "lstm":
                    _, (h_n, _) = self.rnn(emb)
                else:
                    _, h_n = self.rnn(emb)

            # h_n: (num_layers * num_dir, B, H)
            if self.bidirectional:
                # last layer: forward = h_n[-2], backward = h_n[-1]
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                h_last = h_n[-1]
            return h_last
```

### 6.2. `build_model` factory

```python
def build_model(cfg) -> SentimentModel:
    m = cfg["model"]
    return SentimentModel(
        vocab_size    = cfg["data"]["vocab_size"],
        embed_dim     = m["embed_dim"],
        hidden_size   = m["hidden_size"],
        rnn_type      = m.get("type", "lstm"),
        impl          = m.get("impl", "custom"),
        bidirectional = m.get("bidirectional", False),
        num_layers    = m.get("num_layers", 1),
        dropout       = m.get("dropout", 0.0),
        forget_bias   = m.get("forget_bias", 1.0),
    )
```

### 6.3. Backward-compat alias cho 7 test cũ

```python
class SentimentRNN(SentimentModel):
    """Alias giữ nguyên API của Lab 1."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_type="custom"):
        # Lab 1: rnn_type="custom" | "pytorch" → map sang impl
        if rnn_type == "custom":
            super().__init__(vocab_size, embed_dim, hidden_dim,
                             rnn_type="rnn", impl="custom")
        else:  # "pytorch"
            super().__init__(vocab_size, embed_dim, hidden_dim,
                             rnn_type="rnn", impl="pytorch")
```

`forward(x, lengths=None)` đã có default → `model(x)` vẫn hoạt động → **7 test cũ trong `test_model.py` pass nguyên**.

---

## 7. TASK 5 — `src/train.py` (sửa)

### 7.1. `run_epoch` — handle `(x, lengths, y)`

```python
def run_epoch(model, loader, criterion, optimizer=None, clip=None, writer=None, step=0):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for x, lengths, y in loader:
            logits = model(x, lengths)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()

                if writer:
                    # grad norm TRƯỚC clip
                    pre_norm = sum(
                        p.grad.norm() ** 2
                        for p in model.parameters() if p.grad is not None
                    ) ** 0.5
                    writer.add_scalar("grad_norm/pre_clip", pre_norm, step + n_batches)

                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)

                if writer:
                    # grad norm SAU clip
                    post_norm = sum(
                        p.grad.norm() ** 2
                        for p in model.parameters() if p.grad is not None
                    ) ** 0.5
                    writer.add_scalar("grad_norm/post_clip", post_norm, step + n_batches)

                optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy(logits, y)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches
```

### 7.2. Early stopping và checkpoint format mới

```python
def train(model, train_loader, val_loader, cfg, save_path, model_cfg):
    epochs   = cfg["train"]["epochs"]
    lr       = cfg["train"]["lr"]
    clip     = cfg["train"]["grad_clip"]
    patience = cfg["train"].get("early_stopping_patience", 3)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer    = SummaryWriter(log_dir=cfg["paths"]["runs"])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, clip, writer, epoch * 1000)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion)

        logger.info("Epoch %d | train=%.4f/%.4f val=%.4f/%.4f",
                    epoch, train_loss, train_acc, val_loss, val_acc)
        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Checkpoint mới: lưu kèm model_cfg
            torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg}, save_path)
            logger.info("Checkpoint saved (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping tại epoch %d", epoch)
                break

    writer.close()
```

**`model_cfg`** là dict chứa tham số kiến trúc, truyền vào từ `train_run.py`. Dạng:
```python
model_cfg = {
    "vocab_size": len(vocab),
    "embed_dim":  cfg["model"]["embed_dim"],
    "hidden_size": cfg["model"]["hidden_size"],
    "rnn_type":   cfg["model"]["type"],
    "impl":       cfg["model"]["impl"],
    "bidirectional": cfg["model"].get("bidirectional", False),
    "num_layers": cfg["model"].get("num_layers", 1),
    "dropout":    cfg["model"].get("dropout", 0.0),
    "forget_bias": cfg["model"].get("forget_bias", 1.0),
}
```

---

## 8. TASK 6 — `src/evaluate.py` (sửa)

Chỉ thay `for x, y in` thành `for x, lengths, y in` và truyền `lengths`:

```python
def evaluate(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, lengths, y in test_loader:
            logits = model(x, lengths)
            preds  = (torch.sigmoid(logits) >= 0.5).long()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.long().tolist())
    # ... phần còn lại giữ nguyên
```

---

## 9. TASK 7 — `src/inference.py` (sửa)

```python
class Predictor:
    def __init__(self, model_path: str, vocab_path: str, cfg: dict = None):
        self.vocab = Vocabulary.load(vocab_path)
        ckpt = torch.load(model_path, weights_only=False)

        if isinstance(ckpt, dict) and "model_cfg" in ckpt:
            # Format mới (Lab 3)
            model_cfg  = ckpt["model_cfg"]
            state_dict = ckpt["state_dict"]
            self.max_len = cfg["data"]["max_len"] if cfg else 256
        else:
            # Format cũ (Lab 1) — backward compat
            state_dict = ckpt
            model_cfg  = {
                "vocab_size":  len(self.vocab),
                "embed_dim":   cfg["model"]["embed_dim"],
                "hidden_size": cfg["model"]["hidden_dim"],
                "rnn_type":    "rnn",
                "impl":        cfg["model"]["rnn_type"],  # "custom"|"pytorch"
            }
            self.max_len = cfg["data"].get("max_len", cfg["data"].get("max_seq_len", 100))

        self.model = SentimentModel(**model_cfg)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_name = f"{model_cfg.get('rnn_type','rnn')}-{model_cfg.get('impl','custom')}-v1"

    def predict(self, text: str) -> dict:
        tokens = tokenize(text)
        ids    = self.vocab.encode(tokens, self.max_len)
        length = max(min(len(tokens), self.max_len), 1)
        x      = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        lengths = torch.tensor([length], dtype=torch.long)

        with torch.no_grad():
            logit = self.model(x, lengths)
            prob  = torch.sigmoid(logit).item()

        label = "positive" if prob >= 0.5 else "negative"
        return {"label": label, "probability": round(prob, 4), "model": self.model_name}
```

**Test API vẫn pass** vì: response vẫn có `label` và `probability`; `model` là field thêm, không phá schema Pydantic nếu dùng `model_config = ConfigDict(extra="allow")` hoặc field optional.

---

## 10. TASK 8 — `configs/`

### `configs/default.yaml` (sửa)

```yaml
data:
  max_len: 100         # đổi tên từ max_seq_len
  vocab_size: 20000
  min_freq: 2
  train_ratio: 0.8
  val_ratio: 0.1

model:
  type: rnn            # rnn | lstm | gru
  impl: custom         # custom | pytorch
  embed_dim: 64
  hidden_size: 128     # đổi tên từ hidden_dim
  bidirectional: false
  num_layers: 1
  dropout: 0.0
  forget_bias: 1.0

train:
  loss: bce_with_logits
  optimizer: adam
  lr: 0.001
  epochs: 10
  batch_size: 64
  grad_clip: 5.0
  early_stopping_patience: 3
  seeds: [42]

paths:
  data: "data/raw"
  artifacts: "artifacts"
  logs: "logs"
  runs: "runs"

logging:
  level: "INFO"
  console: true
  file: true
```

### `configs/lstm.yaml` (mới)

```yaml
data:
  max_len: 256
  vocab_size: 20000

model:
  type: lstm
  impl: custom
  embed_dim: 64
  hidden_size: 128
  bidirectional: false
  num_layers: 1
  dropout: 0.0
  forget_bias: 1.0

train:
  loss: bce_with_logits
  optimizer: adam
  lr: 0.001
  epochs: 10
  batch_size: 64
  grad_clip: 5.0
  early_stopping_patience: 3
  seeds: [42]

paths:
  data: "data/raw"
  artifacts: "artifacts"
  logs: "logs"
  runs: "runs"

logging:
  level: "INFO"
  console: true
  file: true
```

---

## 11. TASK 9 — Tests (tests/)

### `tests/conftest.py` (mới — shared fixtures)

```python
import pytest
from src.config import load_config

@pytest.fixture
def cfg_lstm():
    return load_config("configs/lstm.yaml")

@pytest.fixture
def cfg_rnn():
    return load_config("configs/default.yaml")
```

### `tests/test_lstm.py` (mới)

3 test bắt buộc từ spec + thêm 2:

```python
# test_matches_torch_lstm  — so output (atol=1e-5)
# test_gradients_match_torch_lstm — so grad x (atol=1e-5)
# test_forget_bias_init — b[H:2H] = forget_bias, b[:H] = 0
# test_output_shape — outputs:(B,T,H), h:(B,H), c:(B,H)
# test_return_gates_shape — gates dict shape (B,T,H) với 4 keys
```

**Gotcha copy weights**: PyTorch layout `(4H, I)` cho `weight_ih_l0`, MyLSTM layout `(I, 4H)` → phải `.T` khi copy. Với bias: PyTorch có `bias_ih_l0` và `bias_hh_l0` riêng biệt → cộng lại mới bằng `b`.

### `tests/test_padding.py` (mới)

```python
import pytest
import torch
from src.model import build_model

@pytest.mark.parametrize("rnn_type,impl", [
    ("rnn",  "custom"),
    ("lstm", "custom"),
    ("rnn",  "pytorch"),
    ("lstm", "pytorch"),
    ("gru",  "pytorch"),
])
def test_padding_invariance(rnn_type, impl):
    cfg = {
        "data": {"vocab_size": 200, "max_len": 500},
        "model": {"type": rnn_type, "impl": impl, "embed_dim": 16,
                  "hidden_size": 32, "bidirectional": False,
                  "num_layers": 1, "dropout": 0.0, "forget_bias": 1.0},
    }
    model = build_model(cfg).eval()
    ids     = torch.tensor([[5, 17, 42, 9, 3]])
    lengths = torch.tensor([5])
    short   = torch.nn.functional.pad(ids, (0, 45))   # pad đến 50
    long    = torch.nn.functional.pad(ids, (0, 495))  # pad đến 500
    with torch.no_grad():
        a = model(short, lengths)
        b = model(long,  lengths)
    assert torch.allclose(a, b, atol=1e-5), \
        f"{rnn_type}/{impl}: logit khác nhau khi pad khác nhau ({a.item():.6f} vs {b.item():.6f})"
```

**Test này sẽ FAIL nếu masking chưa đúng** — đây là acceptance test cho Task 2.

### `tests/test_synthetic.py` (mới)

```python
# test_first_token_is_signal — x[:, 0] == labels + 1
# test_no_padding — lengths == seq_len với mọi sample
# test_deterministic — cùng seed cho cùng data
# test_label_balance — ~50/50 với n_samples lớn
```

### `tests/test_model.py` (sửa nhẹ)

Giữ nguyên 7 test, chỉ cập nhật import nếu cần. Mọi test dùng `SentimentRNN` (alias) và gọi `model(x)` mà không có `lengths` → vẫn hoạt động vì `lengths=None` default.

---

## 12. TASK 10 — `src/synthetic.py` (mới)

```python
class FirstTokenDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, seq_len, n_noise=100, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.labels  = torch.randint(0, 2, (n_samples,), generator=g)
        noise        = torch.randint(3, 3 + n_noise, (n_samples, seq_len), generator=g)
        # label=0 → token 1 (NEG), label=1 → token 2 (POS)
        noise[:, 0]  = self.labels + 1
        self.x       = noise
        self.lengths = torch.full((n_samples,), seq_len, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return self.x[i], self.lengths[i], self.labels[i].float()
```

**Mapping**: `POS=2` khi `label=1`, `NEG=1` khi `label=0`. Ghi rõ trong docstring để tránh nhầm.

---

## 13. TASK 11 — `src/gradient_probe.py` (mới)

```python
def per_position_grad(model, input_ids, lengths, labels, loss_fn, max_positions=200):
    """
    Returns: tensor (max_positions,) — mean ‖∂L/∂e_t‖ indexed by distance from last real token.
    Index 0 = last real token, index k = k steps before last.
    """
    model.zero_grad()
    emb = model.embedding(input_ids).requires_grad_(True)
    # Không gọi model.embedding lại; cần retain_grad vì intermediate
    emb.retain_grad()
    logits = model.forward_from_embeddings(emb, lengths)
    loss_fn(logits, labels.float()).backward()

    g = emb.grad.norm(dim=-1)  # (B, T)

    accum  = torch.zeros(max_positions)
    counts = torch.zeros(max_positions)

    B = input_ids.shape[0]
    for b in range(B):
        L     = lengths[b].item()
        g_seq = g[b, :L]        # (L,) — chỉ token thật
        g_rev = g_seq.flip(0)   # index 0 = token cuối
        take  = min(L, max_positions)
        accum[:take]  += g_rev[:take].detach().cpu()
        counts[:take] += 1

    return accum / counts.clamp(min=1)
```

**Lưu ý kỹ thuật:** `emb.retain_grad()` cần thiết vì `emb` là intermediate tensor (không phải leaf). Nếu không có dòng này, `emb.grad` sẽ là `None` sau backward.

---

## 14. TASK 12 — `src/gate_analysis.py` (mới)

```python
def analyze_gates(model, tokens, vocab, max_len):
    """
    tokens: list[str] — token của 1 review
    Trả về DataFrame: token | f_mean | i_mean | delta_c_norm
    """
    ids     = vocab.encode(tokens, max_len)
    x       = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([min(len(tokens), max_len)])

    model.eval()
    with torch.no_grad():
        emb = model.embedding(x)
        _, _, gates = model.rnn(emb, lengths, return_gates=True)
        # gates: dict "i","f","g","o" → (1, T, H)

    T    = min(len(tokens), max_len)
    rows = []
    c    = torch.zeros(model.rnn.hidden_size)
    for t in range(T):
        f = gates["f"][0, t].cpu()
        i = gates["i"][0, t].cpu()
        g = gates["g"][0, t].cpu()
        c_new = f * c + i * g
        rows.append({
            "token":   tokens[t] if t < len(tokens) else "<PAD>",
            "f_mean":  f.mean().item(),
            "i_mean":  i.mean().item(),
            "delta_c": (c_new - c).norm().item(),
        })
        c = c_new

    return rows  # caller có thể dùng pandas để in bảng
```

**Chú ý:** `return_gates=True` chỉ hoạt động với `MyLSTM` (custom). Với `impl="pytorch"`, cần bỏ qua hoặc implement hook riêng.

---

## 15. TASK 13 — `src/ablation.py` (mới)

```python
def run_ablation(configs: list[dict], seeds: list[int], data_path, vocab_path, base_cfg):
    """
    configs: list of dicts, mỗi dict override một phần của base_cfg["model"]
    Trả về list[dict]: config_name, seeds results, mean, std
    """
    results = []
    for conf in configs:
        cfg = deep_merge(base_cfg, {"model": conf})
        accs = []
        for seed in seeds:
            set_seed(seed)
            model = build_model(cfg)
            # train + evaluate...
            accs.append(test_acc)
        results.append({
            "name":    conf.get("_name", str(conf)),
            "accs":    accs,
            "mean":    np.mean(accs),
            "std":     np.std(accs),
        })
    return results
```

Entry point `ablation_run.py` sẽ định nghĩa list 9 cấu hình A–I và gọi hàm này.

---

## 16. TASK 14 — `api/main.py` (sửa nhẹ)

```python
# Thêm field optional "model" vào response schema
class PredictResponse(BaseModel):
    label: str
    probability: float
    model: str | None = None  # không bắt buộc — không phá test cũ
```

Phần còn lại giữ nguyên. `Predictor.predict()` đã trả về key `"model"` → FastAPI sẽ serialize.

---

## 17. Thứ tự implement chi tiết

```
Bước 1  Setup (copy files, tạo cấu trúc)           ~10 phút
Bước 2  src/dataset.py (thêm lengths)              ~15 phút
Bước 3  src/rnn.py (thêm masking)                  ~15 phút
Bước 4  src/lstm.py (MyLSTMCell + MyLSTM)          ~45 phút
Bước 5  tests/test_lstm.py — chạy 3 test bắt buộc ~20 phút
            pytest tests/test_lstm.py -v
            → 3 tests pass mới tiếp tục
Bước 6  src/model.py (SentimentModel + build_model) ~30 phút
Bước 7  tests/test_padding.py — chạy padding test  ~20 phút
            pytest tests/test_padding.py -v
            → tất cả pass mới tiếp tục
Bước 8  src/train.py (early stopping, grad norm)    ~20 phút
Bước 9  src/evaluate.py (lengths)                  ~5 phút
Bước 10 src/inference.py + api/main.py             ~20 phút
Bước 11 configs/ (default.yaml + lstm.yaml)        ~10 phút
Bước 12 train_run.py — train RNN masked, ghi kết quả vào REPORT.md  ~30 phút
Bước 13 train_run.py — train LSTM, ghi kết quả    ~30 phút
Bước 14 pytest tests/ -v — toàn bộ 24 test cũ pass
Bước 15 src/synthetic.py + synthetic_run.py        ~30 phút
Bước 16 src/gradient_probe.py + thí nghiệm Task 5 ~45 phút
Bước 17 src/gate_analysis.py                       ~20 phút
Bước 18 src/ablation.py + ablation_run.py          ~60 phút
Bước 19 tests/test_synthetic.py                    ~15 phút
Bước 20 Viết REPORT.md                             ~60 phút
```

---

## 18. Acceptance checklist

### Sau bước 5 (LSTM core):
- [ ] `test_matches_torch_lstm` pass (atol=1e-5)
- [ ] `test_gradients_match_torch_lstm` pass (atol=1e-5)
- [ ] `test_forget_bias_init` pass

### Sau bước 7 (padding):
- [ ] `test_padding_invariance` pass cho tất cả 5 combinations (rnn/custom, lstm/custom, rnn/pytorch, lstm/pytorch, gru/pytorch)

### Sau bước 14 (regression):
- [ ] Toàn bộ 24 test cũ pass: 10 preprocessing + 7 model + 7 api
- [ ] `pytest tests/ -v` không có FAILED

### Sau bước 20 (complete):
- [ ] `artifacts/lstm_model.pt` tồn tại và có key `model_cfg`
- [ ] `artifacts/vocab.json` tồn tại
- [ ] REPORT.md có đầy đủ 7 mục theo spec

---

## 19. Các gotcha quan trọng

### G1: Thứ tự gate `i, f, g, o`
PyTorch dùng đúng thứ tự này. Nếu sai (ví dụ `f, i, g, o`) thì `test_matches_torch_lstm` fail. Kiểm tra bằng: `ref.weight_ih_l0[:H]` là weight cho input gate `i`.

### G2: PyTorch có 2 bias, MyLSTM có 1
`my.cell.b = ref.bias_ih_l0 + ref.bias_hh_l0` — tổng hai bias, không phải chỉ một.

### G3: `lengths` phải ở CPU khi truyền vào `pack_padded_sequence`
`pack_padded_sequence(emb, lengths.cpu(), ...)` — kể cả khi model trên GPU.

### G4: `emb.retain_grad()` trong gradient_probe
`emb` từ `model.embedding(x)` là intermediate tensor → mặc định không giữ grad. Phải gọi `.retain_grad()` trước `backward()`.

### G5: Ablation row I (TF-IDF + LR) không dùng PyTorch
Dùng `sklearn.feature_extraction.text.TfidfVectorizer` + `sklearn.linear_model.LogisticRegression`. Không cần seed vì LR có nghiệm analytical — chạy 1 lần.

### G6: Config key thay đổi
- `cfg["data"]["max_seq_len"]` → `cfg["data"]["max_len"]`
- `cfg["model"]["hidden_dim"]` → `cfg["model"]["hidden_size"]`
- `cfg["training"]` → `cfg["train"]`

Cần update mọi chỗ tham chiếu, bao gồm cả `train_run.py` và `experiment_run.py`.

### G7: `test_api.py` dùng checkpoint thật
`test_predict_positive` và `test_predict_negative` load model từ `artifacts/model.pt` thông qua `api/main.py`. Phải train xong (hoặc mock) trước khi test API pass. Checkpoint Lab 1 cũ vẫn load được nhờ backward-compat trong `Predictor`.

### G8: `SentimentRNN.hidden_dim` vs `hidden_size`
Alias `SentimentRNN` truyền `hidden_dim` (tên cũ) vào `SentimentModel` dưới tên `hidden_size`. Cần map đúng để không bị `TypeError`.

---

## 20. Số tham số LSTM

| Component | Shape | Params |
|---|---|---|
| Embedding | (20000, 64) | 1,280,000 |
| W_x (LSTM) | (64, 512) | 32,768 |
| W_h (LSTM) | (128, 512) | 65,536 |
| b | (512,) | 512 |
| Linear | (128, 1) | 129 |
| **Total** | | **1,378,945** |

So với RNN: phần recurrent tăng từ **24,704** lên **98,816** (≈ 4×), tổng tăng ~5.7%.

---

## 21. Câu trả lời dự kiến cho REPORT.md (định hướng)

**Q0.1 (padding bug):** Sau khi thêm masking, Simple RNN nhiều khả năng tăng từ 69% lên ~75–78%. Nếu vậy thì kết luận Lab 1 "giới hạn của Simple RNN" chưa đủ chính xác — một phần đáng kể là do bug padding.

**Q về 1 bias vs 2 bias:** `bias_ih + bias_hh` và `b` duy nhất đều là vector `(4H,)` cộng vào gates. Hai cách tương đương về mặt biểu diễn vì chỉ tổng mới ảnh hưởng đến output. Việc tách ra cho phép regularize riêng từng phần, nhưng thực tế hiếm ai làm vậy.

**Q về forget gate = 1, input gate = 1:** `c_t = c_{t-1} + g_t` — cell state cộng dồn không giới hạn → có thể diverge. Đây là "additive memory" không kiểm soát.
