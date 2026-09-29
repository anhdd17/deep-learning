# Lab 2 — LSTM Sentiment Classification

Phát triển từ **Lab 1: Simple RNN Sentiment Classification**. Cùng bài toán (IMDb binary sentiment), cùng pipeline data / vocab / API, nhưng thay lõi mô hình bằng **LSTM tự implement từ đầu**, sửa các điểm yếu của lab cũ, và làm lại thí nghiệm vanishing gradient một cách có kiểm soát.

---

## Mục tiêu

Sau lab này bạn sẽ:

1. Implement `MyLSTM` từ đầu và **chứng minh bằng test** rằng nó cho kết quả giống hệt `nn.LSTM`.
2. Xử lý **padding đúng cách** để hidden state cuối là trạng thái tại token thật cuối cùng, không phải sau khi đọc hàng trăm token `<PAD>`.
3. Đo vanishing gradient **theo từng timestep** (thay vì một con số norm tổng của `W_hh`) và so sánh trực tiếp RNN vs LSTM.
4. Dựng một **synthetic long-range task** để thấy rõ khác biệt về khả năng nhớ xa mà dữ liệu thật khó cô lập được.
5. **Nhìn vào bên trong** LSTM: forget gate và input gate phản ứng thế nào với các từ như `not`, `but`.
6. Chạy ablation để hiểu mỗi quyết định thiết kế đóng góp bao nhiêu.

---

## Kế thừa từ Lab 1

Giữ nguyên, **không viết lại**:

| Thành phần | File |
|---|---|
| Normalize + tokenize | `src/preprocessing.py` |
| Vocabulary (PAD=0, UNK=1, build từ train only) | `src/vocabulary.py` |
| Split 20k / 2.5k / 2.5k, seed 42 | `src/dataset.py` |
| Logger, config loader | `src/logger.py`, `src/config.py` |
| FastAPI `/predict`, `/health` | `api/main.py` |
| Bộ test hiện có (24 tests) | `tests/` |

Toàn bộ 24 test cũ **phải vẫn pass** khi kết thúc lab.

---

## Task 0 — Review lại Lab 1 (đọc kỹ trước khi code)

Lab 1 có ba điểm đáng xem lại. Hiểu chúng là lý do tồn tại của lab này.

### 0.1. Post-padding + lấy `h_T` = đọc cả padding

Pipeline cũ pad về cuối: `[10, 452, 11, 18, 0, 0, ..., 0]`, rồi lấy `h_T` ở **vị trí cuối của tensor**. Với review 40 token pad lên 200, RNN chạy thêm 160 bước trên `<PAD>`.

`padding_idx=0` chỉ làm embedding của PAD bằng vector 0 — **không** làm RNN đứng yên. Mỗi bước PAD vẫn tính:

```
h_t = tanh(0 @ W_xh + h_{t-1} @ W_hh + b) = tanh(h_{t-1} @ W_hh + b)
```

Tức là hidden state bị nhân `W_hh` và cộng bias thêm 160 lần, thông tin từ review bị bào mòn trước khi đến classifier. Đây rất có thể là một phần lớn lý do Lab 1 chỉ đạt ~69%, và là yếu tố gây nhiễu trong thí nghiệm seq_len (tăng seq_len = tăng số bước PAD với review ngắn).

> **Câu hỏi:** Kết luận ở Lab 1 rằng các câu sai là "giới hạn của Simple RNN, không phải lỗi implement" — sau khi đọc 0.1, bạn có còn chắc chắn không? Ghi câu trả lời vào `REPORT.md`, rồi kiểm chứng ở Task 2.

### 0.2. Gradient norm của `W_hh` không đo trực tiếp vanishing

`W_hh.grad` là **tổng** đóng góp gradient từ mọi timestep. Nó không cho biết gradient tại token số 5 so với token số 195 khác nhau thế nào — mà đó mới là định nghĩa của vanishing. Ngoài ra kết quả Lab 1 có dấu hiệu nhiễu (custom seq_len=50 đạt 51.5% nhưng seq_len=100 lại 53.3%; pytorch seq_len=50 đạt 51.3% nhưng seq_len=100 đạt 61.0%), cho thấy mỗi cấu hình mới chạy 1 seed.

→ Task 5 sẽ đo `‖∂L/∂x_t‖` **theo từng vị trí** và chạy nhiều seed.

### 0.3. Nhỏ: bảng inference ghi "10 test cases" nhưng có 12 dòng

Sửa lại cho khớp. Lab này sẽ dùng bộ 12 câu đó làm **bộ test hành vi cố định** để so sánh các model.

---

## Project Structure (phần mới / thay đổi)

```
rnn-sentiment/
├── configs/
│   ├── default.yaml           # thêm model.type: rnn | lstm | gru
│   └── lstm.yaml              # config riêng cho lab này
├── src/
│   ├── rnn.py                 # (cũ) MySimpleRNN  → sửa: hỗ trợ lengths
│   ├── lstm.py                # (MỚI) MyLSTMCell + MyLSTM
│   ├── model.py               # SentimentModel + build_model(cfg) factory
│   ├── synthetic.py           # (MỚI) long-range synthetic dataset
│   ├── gradient_probe.py      # (MỚI) đo grad theo timestep
│   ├── gate_analysis.py       # (MỚI) trích xuất & in gate activations
│   └── ablation.py            # (MỚI) chạy lưới ablation nhiều seed
├── tests/
│   ├── test_lstm.py           # (MỚI) so khớp với nn.LSTM, shape, mask
│   └── test_padding.py        # (MỚI) padding invariance
├── ablation_run.py
├── synthetic_run.py
└── REPORT.md                  # (MỚI) kết quả + trả lời câu hỏi
```

---

## Task 1 — Implement `MyLSTM` từ đầu

### 1.1. Công thức

Với `x_t` shape `(B, I)`, `h_{t-1}`, `c_{t-1}` shape `(B, H)`:

```
gates = x_t @ W_x + h_{t-1} @ W_h + b        # (B, 4H)
i, f, g, o = gates.chunk(4, dim=-1)          # mỗi cái (B, H)

i = sigmoid(i)      # input gate
f = sigmoid(f)      # forget gate
g = tanh(g)         # candidate  (c̃_t)
o = sigmoid(o)      # output gate

c_t = f * c_{t-1} + i * g
h_t = o * tanh(c_t)
```

**Quy ước layout** (giữ giống `MySimpleRNN` của Lab 1, tức `x @ W`):

| Tham số | Shape | Ghi chú |
|---|---|---|
| `W_x` | `(I, 4H)` | 4 ma trận input→gate xếp cạnh nhau |
| `W_h` | `(H, 4H)` | 4 ma trận hidden→gate |
| `b` | `(4H,)` | một bias duy nhất |

Thứ tự gate **bắt buộc là `i, f, g, o`** — trùng với PyTorch, để Task 1.3 so khớp được.

### 1.2. Skeleton

```python
# src/lstm.py
import torch
import torch.nn as nn


class MyLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, forget_bias: float = 1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_x = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters(forget_bias)

    def reset_parameters(self, forget_bias: float):
        # TODO:
        # - W_x, W_h: uniform(-1/sqrt(H), 1/sqrt(H))  (giống PyTorch)
        # - b: 0, riêng đoạn của forget gate b[H:2H] = forget_bias
        ...

    def forward(self, x_t, state):
        h_prev, c_prev = state
        # TODO: tính gates, trả về h_t, c_t, và dict gates (i, f, g, o)
        ...


class MyLSTM(nn.Module):
    """Unroll MyLSTMCell theo thời gian. Input batch_first: (B, T, I)."""

    def __init__(self, input_size, hidden_size, forget_bias=1.0):
        super().__init__()
        self.cell = MyLSTMCell(input_size, hidden_size, forget_bias)
        self.hidden_size = hidden_size

    def forward(self, x, lengths=None, return_gates=False):
        """
        x:       (B, T, I)
        lengths: (B,) số token thật của mỗi sample; None = không có padding
        return:
            outputs: (B, T, H)
            (h_n, c_n): mỗi cái (B, H) — trạng thái tại token THẬT cuối cùng
            gates (nếu return_gates): dict tên → (B, T, H)
        """
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_size)
        c = x.new_zeros(B, self.hidden_size)
        # TODO: vòng lặp t = 0..T-1, áp dụng mask (xem Task 2)
        ...
```

### 1.3. Test so khớp với `nn.LSTM` (viết đầy đủ, đây là test quan trọng nhất)

Copy trọng số từ `nn.LSTM` sang `MyLSTM` rồi so output. Chú ý PyTorch dùng layout `(4H, I)` và **hai** bias.

```python
# tests/test_lstm.py
import torch
import torch.nn as nn
from src.lstm import MyLSTM


def _copy_from_torch(my: MyLSTM, ref: nn.LSTM):
    with torch.no_grad():
        my.cell.W_x.copy_(ref.weight_ih_l0.T)                 # (4H, I) -> (I, 4H)
        my.cell.W_h.copy_(ref.weight_hh_l0.T)                 # (4H, H) -> (H, 4H)
        my.cell.b.copy_(ref.bias_ih_l0 + ref.bias_hh_l0)      # gộp 2 bias


def test_matches_torch_lstm():
    torch.manual_seed(0)
    B, T, I, H = 4, 13, 8, 16
    ref = nn.LSTM(I, H, batch_first=True)
    my = MyLSTM(I, H)
    _copy_from_torch(my, ref)

    x = torch.randn(B, T, I)
    out_ref, (h_ref, c_ref) = ref(x)
    out_my, (h_my, c_my) = my(x)

    assert torch.allclose(out_my, out_ref, atol=1e-5)
    assert torch.allclose(h_my, h_ref[0], atol=1e-5)
    assert torch.allclose(c_my, c_ref[0], atol=1e-5)


def test_gradients_match_torch_lstm():
    torch.manual_seed(0)
    B, T, I, H = 3, 7, 5, 6
    ref = nn.LSTM(I, H, batch_first=True)
    my = MyLSTM(I, H)
    _copy_from_torch(my, ref)

    x1 = torch.randn(B, T, I, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    ref(x1)[0].sum().backward()
    my(x2)[0].sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)


def test_forget_bias_init():
    my = MyLSTM(4, 8, forget_bias=1.0)
    H = 8
    b = my.cell.b.detach()
    assert torch.allclose(b[H:2 * H], torch.ones(H))
    assert torch.allclose(b[:H], torch.zeros(H))
```

**Acceptance:** 3 test trên pass. Nếu `test_matches_torch_lstm` fail, 90% là sai thứ tự gate hoặc quên transpose.

---

## Task 2 — Xử lý padding đúng cách

### 2.1. Masking trong vòng lặp custom

Ý tưởng: tại bước `t`, sample nào đã hết token thật thì **giữ nguyên** `h`, `c`.

```python
mask_t = (t < lengths).float().unsqueeze(1)      # (B, 1)
h = mask_t * h_new + (1 - mask_t) * h
c = mask_t * c_new + (1 - mask_t) * c
```

Nhờ vậy sau vòng lặp, `h` chính là trạng thái tại token thật cuối — không cần `gather`. Áp dụng **cả cho `MySimpleRNN` của Lab 1** (thêm tham số `lengths`).

Dataset cần trả về thêm `lengths`: số token thật sau khi truncate, tối thiểu 1.

### 2.2. Với `nn.LSTM` / `nn.GRU` / `nn.RNN` của PyTorch

```python
from torch.nn.utils.rnn import pack_padded_sequence

packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
_, (h_n, c_n) = self.rnn(packed)     # h_n đã là trạng thái tại token thật cuối
h_last = h_n[-1]                     # tầng cuối
```

Lưu ý: `lengths` phải nằm trên CPU; `nn.GRU`/`nn.RNN` trả về `h_n` chứ không phải tuple.

### 2.3. Test padding invariance

Một review phải cho **cùng logit** dù pad đến 50 hay 500.

```python
# tests/test_padding.py
import torch
from src.model import build_model


def test_padding_invariance(cfg_lstm):
    model = build_model(cfg_lstm).eval()
    ids = torch.tensor([[5, 17, 42, 9, 3]])
    lengths = torch.tensor([5])
    short = torch.nn.functional.pad(ids, (0, 45))     # pad đến 50
    long = torch.nn.functional.pad(ids, (0, 495))     # pad đến 500
    with torch.no_grad():
        a = model(short, lengths)
        b = model(long, lengths)
    assert torch.allclose(a, b, atol=1e-6)
```

Viết test này cho **mọi `model.type`** (dùng `pytest.mark.parametrize`). Chạy thử với model Lab 1 chưa sửa để thấy nó fail.

### 2.4. Đo lại Lab 1 sau khi sửa

Train lại **Simple RNN của Lab 1, chỉ thêm masking**, giữ nguyên mọi hyperparameter. Ghi vào `REPORT.md`:

| Model | Padding | Val Acc | Test Acc |
|---|---|---|---|
| Simple RNN (Lab 1) | naive | 69.1% | 69.0% |
| Simple RNN | masked | ? | ? |

Đây là câu trả lời thực nghiệm cho câu hỏi ở 0.1.

---

## Task 3 — Model và config

### 3.1. Factory thay vì hard-code

```python
# src/model.py
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, rnn_type="lstm",
                 impl="custom", bidirectional=False, num_layers=1,
                 dropout=0.0, forget_bias=1.0, pad_idx=0):
        ...
        # Head: Linear(hidden_size * num_directions, 1)

    def forward(self, input_ids, lengths):
        # returns logits (B,)
        ...


def build_model(cfg) -> SentimentModel:
    ...
```

Ràng buộc:

- `impl="custom"` chỉ cần hỗ trợ `num_layers=1`, `bidirectional=False` (custom bidirectional là bài mở rộng).
- `impl="pytorch"` hỗ trợ đủ `bidirectional`, `num_layers`, `dropout`.
- Checkpoint lưu **kèm model config** (`torch.save({"state_dict": ..., "model_cfg": ...})`) để `Predictor` tự dựng đúng kiến trúc — API không cần biết model là RNN hay LSTM.

### 3.2. `configs/lstm.yaml`

```yaml
data:
  max_len: 256
  vocab_size: 20000

model:
  type: lstm          # rnn | lstm | gru
  impl: custom        # custom | pytorch
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
```

Giữ `embed_dim`, `hidden_size`, `lr`, `batch_size`, `grad_clip` **giống Lab 1** để so sánh công bằng. Ghi lại số tham số: LSTM ~4× phần recurrent so với RNN (điền bảng như Lab 1).

---

## Task 4 — Training & Evaluation

1. Train `MyLSTM` (custom, masked) với `lstm.yaml`.
2. Thêm **early stopping** theo `val_loss` (patience 3); vẫn lưu best checkpoint.
3. Log lên TensorBoard: loss, acc, và **grad norm tổng** (sau clip và trước clip) — để biết clipping có thực sự kích hoạt hay không.
4. Evaluate trên test set: Accuracy, Precision, Recall, F1, Confusion Matrix (tái sử dụng `evaluate.py`).
5. Chạy bộ **12 câu hành vi** từ Lab 1, điền bảng so sánh:

| Text | Expected | RNN (L1) | RNN masked | LSTM |
|---|---|---|---|---|
| "This film is not bad at all, actually quite enjoyable." | positive | negative | ? | ? |
| "The special effects were great but the story was a mess." | negative | positive | ? | ? |
| "Not the worst movie I have seen." | positive | negative | ? | ? |
| ... (9 câu còn lại) | | | | |

**Tham khảo (không phải đáp án):** LSTM một chiều trên IMDb với setup cỡ này thường đạt vùng 80%+ test accuracy nếu padding xử lý đúng. Nếu bạn vẫn ở quanh 70%, hãy nghi ngờ pipeline trước khi nghi ngờ kiến trúc.

> **Lưu ý khi diễn giải:** 12 câu là quá ít để kết luận thống kê. Dùng chúng như **smoke test hành vi**, không phải metric.

---

## Task 5 — Vanishing gradient, làm lại cho đúng

### 5.1. Đo gradient theo vị trí

Thay vì `W_hh.grad.norm()`, đo `‖∂L/∂e_t‖` — gradient của loss theo **embedding tại từng timestep**. Nó trả lời trực tiếp: "token ở vị trí t có ảnh hưởng bao nhiêu đến việc cập nhật?"

```python
# src/gradient_probe.py
def per_position_grad(model, input_ids, lengths, labels, loss_fn):
    """
    Trả về tensor (max_len,) : trung bình ‖∂L/∂e_t‖ theo KHOẢNG CÁCH từ token thật cuối.
    Index 0 = token cuối, index k = cách token cuối k bước.
    """
    model.zero_grad()
    emb = model.embedding(input_ids)          # (B, T, E)
    emb.retain_grad()
    logits = model.forward_from_embeddings(emb, lengths)   # TODO: tách forward thành 2 phần
    loss_fn(logits, labels.float()).backward()

    g = emb.grad.norm(dim=-1)                 # (B, T)
    # TODO: với mỗi sample, đảo ngược đoạn [0:length] để index theo khoảng cách từ cuối,
    #       bỏ phần padding, rồi trung bình trên batch (chỉ tính vị trí có dữ liệu).
    ...
```

Vì sao đo theo **khoảng cách từ cuối** thay vì vị trí tuyệt đối? Vì review dài ngắn khác nhau; thứ quan trọng là gradient phải đi ngược bao nhiêu bước.

### 5.2. Thí nghiệm

- Model: RNN (masked) và LSTM (custom), đều **chưa train** (khởi tạo) và **sau 1 epoch**.
- Đo trên 20 batch của val set, chỉ lấy review có length ≥ 200.
- Vẽ đồ thị: trục x = khoảng cách từ token cuối (0 → 200), trục y = mean grad norm, **log scale**. 4 đường.

**Kỳ vọng định tính:** đường RNN giảm gần như tuyến tính trên log scale (tức giảm theo hàm mũ), đường LSTM giảm chậm hơn rõ rệt. Nếu không thấy vậy, kiểm tra lại masking và cách index.

### 5.3. Thí nghiệm seq_len, nhiều seed

Làm lại bảng seq_len của Lab 1, nhưng với masking và **3 seeds**, báo cáo mean ± std:

| max_len | RNN val_acc | LSTM val_acc |
|---|---|---|
| 50 | ? ± ? | ? ± ? |
| 100 | ? ± ? | ? ± ? |
| 200 | ? ± ? | ? ± ? |
| 400 | ? ± ? | ? ± ? |

> **Câu hỏi:** Với IMDb, nhiều review thể hiện cảm xúc rõ ở câu cuối. Vậy tăng `max_len` có chắc giúp ích không, hay chỉ làm bài toán khó hơn cho RNN? Đây là lý do cần Task 6.

---

## Task 6 — Synthetic long-range task

Dữ liệu thật trộn lẫn nhiều hiệu ứng. Để **cô lập đúng khả năng nhớ xa**, tự tạo một task mà đáp án chỉ nằm ở đầu chuỗi.

### 6.1. Định nghĩa

- Vocab: `PAD=0`, hai token tín hiệu `POS=1`, `NEG=2`, và 100 token nhiễu `3..102`.
- Mỗi sample: **token đầu là POS hoặc NEG** (label tương ứng), `L-1` token còn lại là nhiễu ngẫu nhiên.
- Model đọc hết chuỗi, phân loại bằng `h` cuối. Muốn đúng, nó **phải nhớ token đầu qua L-1 bước nhiễu**.

```python
# src/synthetic.py
class FirstTokenDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, seq_len, n_noise=100, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 2, (n_samples,), generator=g)
        noise = torch.randint(3, 3 + n_noise, (n_samples, seq_len), generator=g)
        noise[:, 0] = self.labels + 1          # 1 = NEG(label 0)?  -> tự quyết mapping, ghi rõ
        self.x = noise
        self.lengths = torch.full((n_samples,), seq_len)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.x[i], self.lengths[i], self.labels[i]
```

### 6.2. Thí nghiệm

- `L ∈ {10, 25, 50, 100, 200, 500}`, 10k train / 2k val mỗi L.
- Model nhỏ: `embed_dim=16`, `hidden_size=32`, 20 epochs, Adam 1e-3, clip 1.0.
- So sánh: RNN, LSTM (`forget_bias=1`), LSTM (`forget_bias=0`), GRU (`impl=pytorch`).
- Vẽ val accuracy theo L.

**Kỳ vọng định tính:** với L nhỏ, mọi model gần 100%. Khi L tăng, RNN rơi về ~50% (đoán mò) sớm nhất; LSTM trụ được xa hơn nhiều. Điểm đáng chú ý là khác biệt giữa `forget_bias=0` và `forget_bias=1` — đo xem nó xuất hiện từ L nào.

### 6.3. Mở rộng (tuỳ chọn)

- Đặt token tín hiệu ở **vị trí ngẫu nhiên** trong 10% đầu chuỗi.
- Thêm token `FLIP` ở giữa chuỗi đảo label — tương tự "not" trong ngôn ngữ thật. Model có học được không?

---

## Task 7 — Nhìn vào bên trong: gate analysis

Dùng `MyLSTM(..., return_gates=True)` trên model đã train ở Task 4.

### 7.1. In gate theo từng token

Với mỗi token, tính trung bình trên chiều hidden của `f_t` và `i_t`:

```
token        f_mean   i_mean   Δ‖c‖
this         0.71     0.38     +0.21
film         0.74     0.35     +0.09
is           0.78     0.29     +0.03
not          0.62     0.55     +0.40   ← ?
bad          0.58     0.61     +0.52   ← ?
...
```

(Số minh hoạ để thấy format — số thật của bạn sẽ khác.)

Chạy trên 3 câu sai của Lab 1 và 3 câu đúng. Ghi nhận xét vào `REPORT.md`.

### 7.2. Tìm neuron "chuyên biệt" (tuỳ chọn)

Trung bình trên chiều hidden che mất nhiều thông tin. Thử:

1. Thu `c_t` của 2.000 review val.
2. Với mỗi neuron `j`, tính tương quan giữa `c_T[j]` cuối và label.
3. Lấy top-5 neuron tương quan mạnh nhất, vẽ `c_t[j]` theo từng token của một review — xem nó "bật" ở đâu.

> **Cảnh báo diễn giải:** Gate activation cao ở một token **không chứng minh** token đó là nguyên nhân quyết định. Đây là công cụ quan sát, không phải giải thích nhân quả. Ghi kết luận một cách thận trọng.

---

## Task 8 — Ablation

Chạy bằng `ablation_run.py`, **3 seeds mỗi cấu hình**, báo cáo test acc mean ± std. Thay đổi **một yếu tố mỗi lần** so với baseline (LSTM custom, masked, forget_bias=1, uni, 1 layer):

| # | Thay đổi | Test Acc |
|---|---|---|
| A | Baseline | ? ± ? |
| B | Không masking (naive padding) | ? ± ? |
| C | `forget_bias = 0` | ? ± ? |
| D | Không gradient clipping | ? ± ? |
| E | `impl=pytorch` (sanity: phải ≈ A) | ? ± ? |
| F | GRU (pytorch) | ? ± ? |
| G | Bidirectional (pytorch) | ? ± ? |
| H | 2 layers, dropout 0.3 (pytorch) | ? ± ? |
| I | Baseline không phải deep learning: TF-IDF + LogisticRegression | ? |

Dòng I **bắt buộc**. Nếu LSTM không vượt được TF-IDF + LR một khoảng rõ ràng, điều đó nói lên rất nhiều về bài toán (và cần được ghi nhận thẳng thắn trong report).

Dòng E là sanity check: nếu A và E chênh lệch nhiều hơn độ lệch giữa các seed, có bug trong custom implementation.

Ghi thêm cột **thời gian train/epoch** cho A và E — bạn sẽ thấy vì sao cuDNN kernel quan trọng.

---

## Task 9 — Inference, API, Tests

1. `Predictor` đọc `model_cfg` từ checkpoint, tự build đúng model. **API không đổi interface**.
2. Thêm field (tuỳ chọn, không phá schema cũ) trong response: `"model": "lstm-custom-v1"`.
3. Tests mới cần có:

| File | Nội dung |
|---|---|
| `test_lstm.py` | so khớp output & gradient với `nn.LSTM`, forget bias init, shape, `return_gates` shape |
| `test_padding.py` | padding invariance cho mọi `model.type` × `impl` |
| `test_model.py` (mở rộng) | `build_model` cho từng cấu hình, backward pass, load checkpoint cũ của Lab 1 vẫn chạy |
| `test_synthetic.py` | token đầu khớp label, không có PAD, deterministic theo seed |
| `test_api.py` | giữ nguyên + chạy với checkpoint LSTM |

```bash
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/ -v
```

---

## Deliverables

- [ ] Code theo cấu trúc trên, **toàn bộ test pass** (24 cũ + mới).
- [ ] `artifacts/lstm_model.pt` (kèm `model_cfg`), `artifacts/vocab.json` (dùng lại).
- [ ] `REPORT.md` gồm:
  - Bảng Task 2.4 (RNN naive vs masked) + trả lời câu hỏi 0.1
  - Training curve và metrics test của LSTM (format giống Lab 1)
  - Bảng 12 câu hành vi, 3 model
  - Đồ thị per-position gradient (Task 5.2) + bảng seq_len nhiều seed (5.3)
  - Đồ thị synthetic task (Task 6.2)
  - Bảng gate analysis + nhận xét thận trọng (Task 7)
  - Bảng ablation đầy đủ (Task 8)
  - Phần **"Điều bất ngờ"**: ít nhất 1 kết quả khác với kỳ vọng của bạn và giải thích

---

## Câu hỏi suy nghĩ (trả lời trong REPORT.md)

1. Trong `MyLSTM`, tại sao chỉ cần **một** bias `b` trong khi PyTorch dùng hai (`bias_ih`, `bias_hh`)? Hai cách có tương đương về khả năng biểu diễn không?
2. Nếu forget gate luôn bằng 1 và input gate luôn bằng 1, cell state trở thành gì? Nó có vấn đề gì?
3. Gradient theo đường `c_t → c_{t-1}` xấp xỉ `f_t`. Nhưng gradient còn đi qua `h_{t-1}` vào các gate. Đường đó có bị vanish không? Vì sao LSTM vẫn hoạt động tốt?
4. Bidirectional LSTM cải thiện classification — tại sao lại **không dùng được** cho language model?
5. Masking (Task 2.1) và `pack_padded_sequence` cho cùng kết quả. Về hiệu năng, cái nào nhanh hơn trên GPU và vì sao?
6. Kết quả synthetic task có phản ánh đúng những gì xảy ra trên IMDb không? Khi nào một thí nghiệm synthetic "đẹp" lại gây hiểu lầm?

---

## Gợi ý thứ tự làm

```
Task 0 (đọc) → Task 1 (+test so khớp) → Task 2 (+test padding) → Task 2.4 (train lại RNN)
   → Task 3 → Task 4 → Task 6 (nhanh, nhỏ) → Task 5 → Task 7 → Task 8 → Task 9
```

Làm Task 6 trước Task 5 vì synthetic task train rất nhanh và cho feedback rõ ràng về việc implement có đúng không.
