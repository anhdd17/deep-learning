# Lab 3 — LSTM Sentiment Classification: Report

---

## 1. Kết quả training

### 1.1. Simple RNN (có masking) vs LSTM custom

| Model | Config | Epochs | Best val_loss | Test Acc | Test F1 |
|-------|--------|--------|--------------|----------|---------|
| Simple RNN (masked) | default.yaml, max_len=100 | 7 (early stop) | 0.6928 | 50.16% | 0.517 |
| **LSTM custom** | lstm.yaml, max_len=256 | 9 (early stop) | 0.3747 | **85.16%** | 0.850 |

**Nhận xét:** Simple RNN với masking chỉ đạt 50% — gần random. Có hai giải thích:
1. RNN bị vanishing gradient nặng ở max_len=100, không học được phụ thuộc dài.
2. Cần thêm epochs và tuning hơn để đánh giá fair hơn.

LSTM custom (tự implement) đạt 85.16%, xác nhận gating mechanism hoạt động đúng.

### 1.2. Trả lời câu hỏi từ Lab 1

> "Kết luận ở Lab 1 rằng các câu sai là giới hạn của Simple RNN, không phải lỗi implement — sau khi đọc 0.1, bạn có còn chắc chắn không?"

Không còn chắc chắn. Lab 1 có bug padding: PAD token vẫn được đọc sau token thật cuối, làm nhiễu hidden state. Đây là lỗi implement, không phải giới hạn kiến trúc. Sau khi fix masking, RNN vẫn yếu — nhưng lý do chính xác hơn là vanishing gradient, được xác nhận định lượng ở Task 5.

---

## 2. LSTM Implementation

### 2.1. Công thức

```
gates = x_t @ W_x + h_{t-1} @ W_h + b    # (B, 4H)
i, f, g, o = gates.chunk(4, dim=-1)

c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
h_t = sigmoid(o) * tanh(c_t)
```

### 2.2. Kiểm tra khớp với nn.LSTM

`test_matches_torch_lstm` pass với atol=1e-5 — custom implementation khớp hoàn toàn với PyTorch sau khi copy weights đúng cách (chú ý transpose và cộng hai bias).

### 2.3. Padding invariance

`test_padding_invariance` pass cho 5 combinations (rnn/lstm/gru × custom/pytorch): logit không thay đổi khi pad thêm bất kỳ số lượng PAD token nào.

---

## 3. Gradient probe — vanishing gradient theo vị trí

Đo `‖∂L/∂e_t‖` theo khoảng cách từ token thật cuối, trên val samples có length ≥ 200.

| Dist từ cuối | RNN_init | RNN_1ep | LSTM_init | LSTM_1ep |
|-------------:|:--------:|:-------:|:---------:|:--------:|
| 0 | 1.76e-03 | 1.71e-03 | 1.96e-03 | 3.29e-03 |
| 9 | 5.63e-12 | 7.00e-05 | 1.77e-04 | 2.86e-03 |
| 24 | ~0 | 3.00e-05 | 1.28e-05 | 2.32e-03 |
| 49 | ~0 | 7.63e-06 | 1.21e-07 | 1.73e-03 |
| 99 | ~0 | 3.57e-07 | 1.39e-11 | 9.32e-04 |
| 199 | ~0 | 2.18e-10 | 2.99e-20 | 2.76e-04 |

**Kết luận:** RNN (untrained) mất gradient hoàn toàn từ dist=9. Sau 1 epoch, gradient vẫn tồn tại nhưng giảm ~7 bậc độ lớn từ dist=0 đến dist=199. LSTM (1 epoch) giảm chỉ ~1 bậc — cell state giữ gradient flow hiệu quả hơn nhiều.

![Gradient probe](artifacts/gradient_probe.png)

---

## 4. Synthetic long-range memory task

Task: token đầu tiên là tín hiệu duy nhất, model phải nhớ qua L-1 bước nhiễu.

| L | RNN | LSTM (fb=1) | LSTM (fb=0) | GRU |
|--:|:---:|:-----------:|:-----------:|:---:|
| 10 | 100% | 100% | 100% | 100% |
| 25 | 100% | 100% | 100% | 100% |
| 50 | **50%** | 100% | 100% | **51%** |
| 100 | 52% | **100%** | **52%** | 50% |
| 200 | 50% | 51% | 51% | 51% |
| 500 | 52% | 51% | 52% | 50% |

**Kết luận:**
- RNN và GRU fail từ L=50 — vanishing gradient không giữ được thông tin qua nhiều bước.
- LSTM (fb=0) fail từ L=100 — forget gate khởi tạo ở 0.5 (sigmoid(0)) quên thông tin nhanh hơn.
- LSTM (fb=1) trụ đến L=100 (100%) — forget gate gần 1 ban đầu → cell state "mở" mặc định.
- Ở L=200/500 tất cả đều về ~50% với hidden=32, 20 epochs — giới hạn capacity, không phải kiến trúc.

> **Khi nào synthetic "đẹp" lại gây hiểu lầm?** Synthetic cô lập một hiệu ứng duy nhất. Trên IMDb, cảm xúc thường xuất hiện ở cuối review (token gần nhất với classifier) — RNN không cần nhớ xa để đạt accuracy vừa phải. Synthetic cho thấy *khả năng* nhớ xa, không phải *sự cần thiết* của nó trên dữ liệu thật.

![Synthetic results](artifacts/synthetic_results.png)

---

## 5. Gate analysis

Chạy trên 6 câu (3 positive rõ, 1 trường hợp negation, 1 câu khó, 1 negative rõ). Accuracy: 5/6.

**Quan sát:**

| Câu | Pred | Prob | Ghi chú |
|-----|------|------|---------|
| "absolutely wonderful...superb...moving" | positive | 87.8% | f_mean ổn định ~0.73, i_mean tăng ở từ mạnh |
| "outstanding...brilliant...recommended" | positive | 99.7% | "recommended" có i_mean cao nhất (0.646) |
| "loved...masterpiece" | positive | 94.2% | "loved" kích hoạt mạnh ngay từ đầu |
| "not bad...actually quite good" | positive | 55.1% | "bad" có Δ‖c‖=5.24 — cao nhất trong câu, model bị nhiễu |
| "expected to hate...surprisingly decent" | **negative** (sai) | 22.3% | "hate" early làm cell state lệch negative, "decent" cuối không đủ recover |
| "terrible...waste...painful" | negative | 98.9% | "waste" Δ‖c‖=4.71 — tín hiệu mạnh nhất |

**Nhận xét:** LSTM nhạy cảm với từ cảm xúc mạnh (i_mean cao, Δ‖c‖ lớn). Điểm yếu rõ: không xử lý tốt negation ("hate...but decent") vì từ tiêu cực xuất hiện sớm chiếm ưu thế.

> **Cảnh báo diễn giải:** Gate activation cao không chứng minh nhân quả — đây là quan sát tương quan.

---

## 6. Câu hỏi suy nghĩ

**Q1. Tại sao MyLSTM dùng 1 bias, PyTorch dùng 2?**

Hai cách tương đương về biểu diễn: `bias_ih + bias_hh` và `b` đều là vector `(4H,)` cộng vào gates. Chỉ tổng mới ảnh hưởng đến output. PyTorch tách ra để có thể regularize riêng từng phần, nhưng trong thực tế hiếm ai làm vậy.

**Q2. forget gate = 1, input gate = 1 thì cell state trở thành gì?**

`c_t = c_{t-1} + g_t` — cộng dồn không giới hạn (additive memory). Cell state có thể diverge theo thời gian vì không có cơ chế kiểm soát độ lớn.

**Q3. Gradient qua h_{t-1} vào các gate có bị vanish không?**

Có — đường qua `h_{t-1}` vẫn đi qua `tanh` và `sigmoid`, có thể bị vanish. Nhưng LSTM hoạt động tốt vì tồn tại *đường tắt* qua `c_t → c_{t-1}` với gradient xấp xỉ `f_t` — không qua activation phi tuyến, nên gradient flow ổn định hơn nhiều.

**Q4. Bidirectional không dùng được cho language model — tại sao?**

Language model phải predict token tiếp theo từ các token đã thấy. Bidirectional đọc cả tương lai (backward pass) → data leakage: model "biết trước" token cần predict.

**Q5. Masking vs pack_padded_sequence — cái nào nhanh hơn trên GPU?**

`pack_padded_sequence` nhanh hơn vì loại bỏ hoàn toàn tính toán trên PAD token — tổng số bước RNN giảm từ `B×T` xuống tổng `lengths`. Masking vẫn tính đủ `B×T` bước, chỉ chặn update. Trên GPU, pack hiệu quả hơn rõ vì tận dụng batching tốt hơn.

**Q6. Synthetic có phản ánh đúng IMDb không?**

Phần nào. Synthetic xác nhận LSTM nhớ xa tốt hơn RNN — điều này đúng về nguyên lý. Nhưng trên IMDb cảm xúc thường tập trung ở đầu hoặc cuối review, không yêu cầu nhớ qua hàng trăm bước nhiễu thuần túy. Synthetic "đẹp" có thể gây hiểu lầm khi ta kết luận "LSTM tốt hơn RNN trên IMDb vì nhớ xa hơn" — thực tế lợi thế có thể đến từ gating mechanism giúp chọn lọc thông tin, không chỉ từ khả năng nhớ xa.

---

## 7. Artifacts

| File | Mô tả |
|------|-------|
| `artifacts/lstm_model.pt` | LSTM custom trained, test acc=85.16% |
| `artifacts/rnn_model.pt` | RNN masked trained, test acc=50.16% |
| `artifacts/vocab.json` | Vocabulary 20,000 tokens |
| `artifacts/gradient_probe.png` | Gradient flow theo vị trí token |
| `artifacts/synthetic_results.png` | Val accuracy vs sequence length |
