# Lab: End-to-End Sentiment Classification với Simple RNN

## 1. Bối cảnh

Bạn được giao xây dựng một hệ thống phân loại cảm xúc của movie review.

Hệ thống nhận vào một câu review bằng tiếng Anh và dự đoán:

- positive
- negative

Ví dụ:

Input:

```
"I really enjoyed this movie. The story was fantastic."
```

Output:

```json
{
    "label": "positive",
    "probability": 0.94
}
```

Model phải được train từ dữ liệu, sau đó được đóng gói thành một API để client có thể gọi.

## 2. Mục tiêu

Sau khi hoàn thành lab, bạn phải hiểu và triển khai được toàn bộ pipeline:

```
Raw Text
   ↓
Tokenization
   ↓
Vocabulary
   ↓
Token IDs
   ↓
Embedding
   ↓
Simple RNN
   ↓
Last Hidden State
   ↓
Linear
   ↓
Logit
   ↓
Probability
   ↓
Classification
```

Đồng thời hiểu training:

```
Prediction
   ↓
Loss
   ↓
Backpropagation / BPTT
   ↓
Gradient
   ↓
Optimizer
   ↓
Update weights
```

## 3. Bài toán

### Task

Binary Text Classification

### Input

Một movie review:

```
"This movie was absolutely amazing"
```

### Target

- 0 → negative
- 1 → positive

### Output

Model phải dự đoán:

```
probability ∈ [0, 1]
```

và chuyển thành:

```
probability >= 0.5 → positive
probability <  0.5 → negative
```

## 4. Dataset

Sử dụng IMDb Movie Review Dataset hoặc một dataset sentiment tiếng Anh tương đương.

Dataset tối thiểu phải có dạng:

```csv
text,label

"I loved this movie",1
"This movie was terrible",0
...
```

Chia dataset thành:

- Train
- Validation
- Test

Không được train trên test set.

## 5. Yêu cầu 1 — Preprocessing

Xây dựng pipeline:

```
raw text
   ↓
normalize
   ↓
tokenize
   ↓
vocabulary
   ↓
token IDs
   ↓
padding/truncation
```

Ví dụ:

```
"I love this movie"

→ ["i", "love", "this", "movie"]

→ [12, 57, 8, 231]
```

Yêu cầu xử lý được:

- từ chưa xuất hiện trong vocabulary → `<UNK>`
- padding → `<PAD>`
- sequence quá dài → truncate
- sequence quá ngắn → pad

## 6. Yêu cầu 2 — Embedding

Xây dựng:

```
Token ID
    ↓
Embedding
    ↓
Vector
```

Ví dụ:

```
token_id = 57

↓

[0.12, -0.43, 0.81, ...]
```

Embedding dimension tự chọn, ví dụ: 64 hoặc 128.

## 7. Yêu cầu 3 — Tự implement Simple RNN

Đây là phần quan trọng nhất của lab.

Không được dùng `nn.RNN` ở version đầu.

Implement công thức:

$$ h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1}+b) $$

Khởi tạo:

$$ h_0 = \vec{0} $$

Model phải xử lý:

```
x₁ → h₁
x₂ → h₂
x₃ → h₃
...
xₜ → hₜ
```

và sử dụng `h_T` làm representation của toàn bộ sentence.

Bắt buộc phải hiểu:

- `x_t` là gì?
- `h_t` là gì?
- `h_0` là gì?
- `W_xh` là gì?
- `W_hh` là gì?
- Vì sao `h_t` chứa information từ các timestep trước?
- Vì sao cùng một `W_xh`, `W_hh` được dùng ở mọi timestep?

## 8. Yêu cầu 4 — Classification Head

Từ `h_T`, thực hiện:

```
h_T
 ↓
Linear
 ↓
logit
```

Sử dụng binary classification.

Training sử dụng `BCEWithLogitsLoss`. Không cần tự gọi sigmoid trước loss.

Inference:

```
logit
 ↓
sigmoid
 ↓
probability
 ↓
threshold 0.5
 ↓
positive / negative
```

## 9. Yêu cầu 5 — Training

Implement training loop:

```
for epoch:

    batch
      ↓
    forward
      ↓
    prediction
      ↓
    loss
      ↓
    backward
      ↓
    optimizer.step()
```

Theo dõi ít nhất:

- train loss
- validation loss
- train accuracy
- validation accuracy

Sử dụng optimizer như Adam.

## 10. Yêu cầu 6 — Evaluation

Sau khi training xong, đánh giá chỉ trên test set.

Report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Không cần cố đạt một accuracy cụ thể.

Mục tiêu chính là hiểu pipeline và model hoạt động như thế nào.

## 11. Yêu cầu 7 — Experiment về vấn đề của Simple RNN

Đây là phần rất quan trọng.

Train/evaluate model với các sequence length khác nhau:

- 10
- 20
- 50
- 100

Quan sát:

- performance
- loss
- gradient

Mục tiêu: tự quan sát vấn đề vanishing/exploding gradient của Simple RNN khi sequence dài.

Nếu có thể, log thêm gradient norm theo timestep/layer để nhìn hiện tượng rõ hơn.

## 12. Yêu cầu 8 — So sánh với nn.RNN

Sau khi tự implement xong `MySimpleRNN`, thay bằng `torch.nn.RNN`.

Sau đó kiểm tra:

```
Custom RNN
      vs
PyTorch nn.RNN
```

Mục tiêu không phải benchmark.

Mục tiêu là chứng minh: `nn.RNN` thực chất đang thực hiện cùng ý tưởng recurrence mà bạn vừa tự implement.

## 13. Yêu cầu 9 — Inference

Viết một module inference:

```python
predict("I really loved this movie")
```

trả về:

```json
{
    "label": "positive",
    "probability": 0.94
}
```

Test ít nhất 10 câu tự nghĩ ra.

Trong đó nên có:

- positive rõ ràng
- negative rõ ràng
- câu ngắn
- câu dài
- câu có từ phủ định
- câu model dễ nhầm

Ví dụ:

```
"I don't think this movie was good."
```

## 14. Yêu cầu 10 — REST API

Dùng FastAPI.

Tạo:

```
POST /predict
```

Request:

```json
{
    "text": "The movie was fantastic"
}
```

Response:

```json
{
    "label": "positive",
    "probability": 0.97
}
```

Thêm:

```
GET /health
```

Response:

```json
{
    "status": "ok"
}
```

## 15. Yêu cầu 11 — Unit Test

Viết test tối thiểu cho:

**Preprocessing:**

- text → tokens
- tokens → IDs
- padding
- UNK

**Model** — kiểm tra shape:

```
input
→ embedding
→ RNN
→ h_T
→ logit
```

**API** — test `POST /predict` với valid và invalid input.

## 16. Cấu trúc project

Bạn có thể tổ chức:

```
rnn-sentiment/
│
├── data/
│
├── src/
│   ├── preprocessing.py
│   ├── vocabulary.py
│   ├── embedding.py
│   ├── rnn.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── api/
│   └── main.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_rnn.py
│   ├── test_model.py
│   └── test_api.py
│
├── artifacts/
│   ├── model.pt
│   ├── vocab.json
│   └── config.json
│
├── requirements.txt
└── README.md
```

Không cần Docker/Kubernetes/MLflow/database.

## 17. README phải giải thích được

README cần có:

**Problem**
- What problem are we solving?

**Data**
- What is the input?
- What is the target?
- How is data split?

**Architecture**

```
Text
 ↓
Tokenizer
 ↓
Embedding
 ↓
Simple RNN
 ↓
Linear
 ↓
Logit
```

**Training**
- Loss
- Optimizer
- Epoch
- Batch size
- Learning rate

**Evaluation**
- Accuracy
- Precision
- Recall
- F1
- Confusion Matrix

**Experiment**

Giải thích: Tại sao Simple RNN gặp vấn đề với sequence dài?

**Inference**

Ví dụ request/response.

## 18. Definition of Done

Lab được coi là DONE khi bạn có thể tự vẽ và giải thích được:

```
                    TRAINING

Text
 ↓
Tokens
 ↓
IDs
 ↓
Embedding
 ↓
x₁ ──→ h₁
       ↓
x₂ ──→ h₂
       ↓
x₃ ──→ h₃
       ↓
      ...
       ↓
xₜ ──→ hₜ
       ↓
     Linear
       ↓
     Logit
       ↓
BCEWithLogitsLoss
       ↓
Backpropagation / BPTT
       ↓
Gradient
       ↓
Adam
       ↓
Update weights
```

và inference:

```
New Text
   ↓
Embedding
   ↓
Simple RNN
   ↓
hₜ
   ↓
Linear
   ↓
Logit
   ↓
Sigmoid
   ↓
Probability
   ↓
Positive / Negative
   ↓
FastAPI
```

### Quan trọng nhất

Bạn không cần tối ưu accuracy và cũng không cần infrastructure.

Mục tiêu của lab là sau khi hoàn thành, bạn có thể nhìn vào:

$$ h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b) $$

và nói được từng thành phần đang làm gì, nó được tạo ra ở đâu, được update như thế nào, và cuối cùng nó biến một câu text thành một prediction ra sao.
