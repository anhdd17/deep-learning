# Simple RNN Sentiment Classification

Binary sentiment classification trên IMDb Movie Reviews sử dụng Simple RNN tự implement từ đầu.

---

## Problem

Phân loại một movie review tiếng Anh thành **positive** hoặc **negative**.

```
Input : "I really enjoyed this movie. The story was fantastic."
Output: {"label": "positive", "probability": 0.94}
```

---

## Data

**Dataset:** IMDb Movie Reviews (Stanford NLP) — 50,000 reviews, balanced 50/50.

**Input:** raw text (movie review)  
**Target:** 0 = negative, 1 = positive

**Split** (từ 25,000 train reviews của IMDb):

| Split | Samples |
|-------|---------|
| Train | 20,000  |
| Val   | 2,500   |
| Test  | 2,500   |

Vocab build từ **train only** — không leak val/test.  
UNK rate trên val: **2.58%** — vocab phủ gần như toàn bộ.

---

## Architecture

```
Text
 ↓ normalize + tokenize
Tokens  ["i", "loved", "this", "movie"]
 ↓ Vocabulary encode + pad
Token IDs  [10, 452, 11, 18, 0, 0, ..., 0]   shape: (seq_len,)
 ↓ nn.Embedding(vocab_size=20000, embed_dim=64, padding_idx=0)
Embeddings   shape: (seq_len, 64)
 ↓ MySimpleRNN
x₁ ──→ h₁
       ↓
x₂ ──→ h₂        h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b)
       ↓
      ...
       ↓
x_T ──→ h_T       shape: (128,)
 ↓ nn.Linear(128, 1)
logit   shape: (1,)
 ↓ sigmoid (inference only)
probability ∈ [0, 1]
 ↓ threshold 0.5
positive / negative
```

**Parameters:**

| Component | Shape | Params |
|-----------|-------|--------|
| Embedding | (20000, 64) | 1,280,000 |
| W_xh | (64, 128) | 8,192 |
| W_hh | (128, 128) | 16,384 |
| b | (128,) | 128 |
| Linear | (128, 1) | 129 |
| **Total** | | **1,304,833** |

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 10 |
| Batch size | 64 |
| Gradient clipping | 5.0 |
| Seed | 42 |

Best checkpoint saved tại epoch có `val_loss` thấp nhất → `artifacts/model.pt`.

**Training curve:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1  | 0.694 | 51.3% | 0.693 | 50.4% |
| 4  | 0.682 | 56.2% | 0.686 | 55.8% |
| 8  | 0.557 | 72.2% | **0.648** | **69.1%** ← best |
| 10 | 0.571 | 70.8% | 0.712 | 63.1% |

Epoch 9-10 val_loss tăng lại → overfitting bắt đầu xuất hiện.

---

## Evaluation

Đánh giá trên **test set** (2,500 samples) với best checkpoint:

| Metric | Score |
|--------|-------|
| Accuracy | 69.0% |
| Precision | 70.3% |
| Recall | 67.3% |
| F1 | 68.8% |

**Confusion Matrix:**

```
              Pred Neg  Pred Pos
  Actual Neg     873       360
  Actual Pos     414       853
```

Model sai tương đương ở cả 2 chiều — không bị lệch class.

---

## Experiment — Vanishing Gradient

Train với 4 sequence lengths khác nhau, quan sát gradient norm của `W_hh` (weight truyền thông tin qua thời gian):

| seq_len | rnn_type | val_acc | grad W_xh | grad W_hh |
|---------|----------|---------|-----------|-----------|
| 10  | custom  | 64.8% | 0.438 | 0.430 |
| 20  | custom  | 64.9% | 0.618 | 0.415 |
| 50  | custom  | 51.5% | 0.220 | 0.384 |
| 100 | custom  | 53.3% | 0.096 | **0.069** ← gần chết |
| 10  | pytorch | 64.3% | 0.447 | 0.611 |
| 20  | pytorch | 65.0% | 0.718 | 0.624 |
| 50  | pytorch | 51.3% | 0.416 | 0.600 |
| 100 | pytorch | 61.0% | 0.269 | 0.547 |

**Tại sao Simple RNN gặp vấn đề với sequence dài?**

Gradient được tính bằng chain rule ngược từ timestep cuối về đầu (BPTT). Mỗi bước nhân thêm `W_hh`. Nếu eigenvalue của `W_hh < 1`, sau 100 lần nhân liên tiếp gradient co về gần 0 — model không còn học được thông tin từ các token xa.

```
seq_len=10  → grad W_hh ≈ 0.43   → model học tốt (~65%)
seq_len=100 → grad W_hh ≈ 0.07   → gradient gần như chết (~53%)
```

LSTM và GRU giải quyết vấn đề này bằng cơ chế gate — cho phép gradient "chảy" qua nhiều timestep mà không bị vanish.

---

## Inference

```python
from src.inference import Predictor
from src.config import load_config

cfg       = load_config("configs/default.yaml")
predictor = Predictor("artifacts/model.pt", "artifacts/vocab.json", cfg)

predictor.predict("I really loved this movie")
# {"label": "positive", "probability": 0.79}
```

**10 test cases:**

| Text | Expected | Predicted | OK? |
|------|----------|-----------|-----|
| "This movie was absolutely amazing, I loved every second of it!" | positive | positive | ✓ |
| "A brilliant masterpiece. Outstanding performances." | positive | positive | ✓ |
| "Terrible movie. Complete waste of time and money." | negative | negative | ✓ |
| "The worst film I have ever seen in my entire life." | negative | negative | ✓ |
| "Great!" | positive | positive | ✓ |
| "Boring." | negative | negative | ✓ |
| "I went into this film with very high expectations..." | negative | negative | ✓ |
| "I don't think this movie was good." | negative | negative | ✓ |
| "This film is not bad at all, actually quite enjoyable." | positive | negative | ✗ |
| "It started slow but the ending saved everything." | positive | positive | ✓ |
| "The special effects were great but the story was a mess." | negative | positive | ✗ |
| "Not the worst movie I have seen." | positive | negative | ✗ |

**9/12 correct.** 3 câu sai đều có pattern phủ định (`not bad`, `not the worst`) hoặc sentiment trái chiều trong cùng câu — giới hạn của Simple RNN, không phải lỗi implement.

---

## REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**POST /predict**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The movie was fantastic"}'
```

```json
{"label": "positive", "probability": 0.97}
```

**GET /health**

```json
{"status": "ok"}
```

---

## Unit Tests

```bash
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/ -v
```

```
24 passed in 0.98s
```

| File | Coverage |
|------|----------|
| `test_preprocessing.py` | normalize, tokenize, UNK, padding, truncation |
| `test_model.py` | shape check mỗi layer, backward pass, padding_idx |
| `test_api.py` | /health, /predict valid/invalid, missing field, schema |

---

## Project Structure

```
rnn-sentiment/
├── configs/default.yaml       # hyperparameters
├── data/raw/                  # IMDb CSV
├── src/
│   ├── logger.py              # setup_logging + get_logger
│   ├── config.py              # load_config từ YAML
│   ├── preprocessing.py       # normalize + tokenize
│   ├── vocabulary.py          # Vocabulary (PAD, UNK, encode, save/load)
│   ├── dataset.py             # IMDbDataset + DataLoader
│   ├── rnn.py                 # MySimpleRNN (tự implement)
│   ├── model.py               # SentimentRNN (Embedding + RNN + Linear)
│   ├── train.py               # training loop + TensorBoard logging
│   ├── evaluate.py            # metrics trên test set
│   ├── experiment.py          # vanishing gradient experiment
│   └── inference.py           # Predictor class
├── api/main.py                # FastAPI
├── tests/                     # pytest
├── artifacts/                 # model.pt, vocab.json
├── runs/                      # TensorBoard logs
├── train_run.py               # entry point training
├── experiment_run.py          # entry point experiment
└── predict.py                 # entry point inference
```
