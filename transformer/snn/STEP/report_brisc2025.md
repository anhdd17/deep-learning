# Báo cáo thử nghiệm: Spikformer trên tập dữ liệu BRISC2025

**Ngày:** 06/10/2026  
**Người thực hiện:** anhdd 
**Server:** gpu2 (NVIDIA Quadro RTX 5000, 16 GiB VRAM)

---

## 1. Mục tiêu

Thử nghiệm khả năng ứng dụng mô hình **Spiking Transformer (SNN)** vào bài toán phân loại khối u não trên ảnh MRI T1. Đây là bước khởi đầu trong hướng nghiên cứu so sánh các kiến trúc SNN trên dữ liệu y tế, trước khi mở rộng sang tập BraTS2023.

---

## 2. Tập dữ liệu — BRISC2025

**BRISC** (BRain tumor Image Segmentation & Classification) là tập dữ liệu MRI não T1 được chú thích bởi chuyên gia y tế, công bố năm 2025.

| Thông số | Chi tiết |
|----------|----------|
| Tổng số ảnh | 6,000 slice 2D (T1-weighted) |
| Train / Test | 5,000 / 1,000 |
| Số lớp | 4 (Glioma, Meningioma, No Tumor, Pituitary) |
| Mặt cắt | Axial, Coronal, Sagittal |
| Format | JPG, 2D slice |

**Phân bố nhãn (train):**

| Class | Số ảnh |
|-------|-------:|
| Glioma | 1,147 |
| Meningioma | 1,329 |
| No Tumor | 1,067 |
| Pituitary | 1,457 |

> Nguồn: Fateh et al., *BRISC: Annotated dataset for brain tumor segmentation and classification*, arXiv:2506.14318, 2025.

---

## 3. Mô hình — Spikformer

**Spikformer** (Ma et al., ICLR 2023) là kiến trúc Spiking Neural Network (SNN) kết hợp cơ chế Transformer, trong đó toàn bộ phép tính floating-point được thay thế bằng xung nhị phân (binary spikes).

### 3.1 Cấu trúc chính

- **SPS (Spiking Patch Splitter):** tokenization ảnh qua 4 lớp Conv-BN-LIF thay vì linear projection
- **SSA (Spiking Self-Attention):** attention mechanism với neuron LIF thay softmax
- **LIF Neuron:** Leaky Integrate-and-Fire — tích lũy điện thế, bắn xung khi vượt ngưỡng
- **Surrogate Gradient (SigmoidGrad):** xấp xỉ đạo hàm qua spike cho backpropagation

### 3.2 Cấu hình thử nghiệm

| Tham số | Giá trị |
|---------|---------|
| Backbone | spikformer\_imagenet |
| Depths | 4 blocks |
| Embed dim | 384 |
| Num heads | 8 |
| Input size | 224 × 224 |
| Tổng tham số | ~9.3M |
| Neuron | LIFNode (τ=2.0, threshold=1.0) |
| Timestep T | 2 |
| Precision | FP16 (AMP) |

### 3.3 Cấu hình huấn luyện

| Tham số | Giá trị |
|---------|---------|
| Optimizer | AdamW |
| Learning rate | 3e-4 (cosine decay) |
| Weight decay | 0.05 |
| Batch size | 8 |
| Epochs | 100 |
| Warmup | 10 epochs |
| Augmentation | RandAugment (m=7, n=2), RandomResizedCrop, ColorJitter |
| Label smoothing | 0.1 |
| Mixup | Tắt (dataset nhỏ) |

---

## 4. Kết quả

### 4.1 Độ chính xác

| Epoch | Train Loss | Eval Loss | Eval Acc@1 |
|-------|-----------|-----------|------------|
| 0 | 1.331 | 1.146 | 54.2% |
| 10 | 0.875 | 0.674 | 75.7% |
| 20 | 0.745 | 0.594 | 76.4% |
| 30 | 0.667 | 0.365 | 87.8% |
| 50 | 0.584 | 0.411 | 85.1% |
| 70 | 0.517 | 0.256 | 92.2% |
| **80** | **0.515** | **0.265** | **93.2%** ← best |
| 90 | 0.494 | 0.264 | 93.1% |
| 100 | 0.479 | 0.258 | 93.0% |

**Kết quả tốt nhất: 93.2% accuracy (epoch 80)**

### 4.2 Inference mẫu (8 ảnh test)

| Ground Truth | Dự đoán | Confidence |
|---|---|---|
| Glioma | ✗ Meningioma | 47.8% |
| Glioma | ✗ Meningioma | 67.8% |
| Meningioma | ✓ Meningioma | 55.4% |
| Meningioma | ✓ Meningioma | 55.7% |
| No Tumor | ✓ No Tumor | 92.2% |
| No Tumor | ✓ No Tumor | 89.3% |
| Pituitary | ✓ Pituitary | 75.7% |
| Pituitary | ✓ Pituitary | 67.0% |

---

## 5. Phân tích

### 5.1 Overfitting

Không có dấu hiệu overfit:
- `eval_loss < train_loss` xuyên suốt quá trình training (do label smoothing và data augmentation làm train loss cao hơn)
- Eval accuracy plateau ổn định quanh 93% từ epoch 80, không suy giảm

### 5.2 Nhận xét per-class

- **No Tumor** cho confidence cao nhất (~90%) — vùng não không có khối u có đặc trưng rõ ràng, dễ phân biệt
- **Pituitary** đạt accuracy tốt (~75% confidence) — vị trí giải phẫu đặc trưng (tuyến yên)
- **Glioma** bị nhầm thành Meningioma — đây là nhầm lẫn phổ biến trong phân loại MRI T1 đơn thuần; trong thực tế lâm sàng cần thêm chuỗi T2/FLAIR/DWI để phân biệt

### 5.3 Hạn chế

- Chỉ dùng 1 modality (T1), thiếu thông tin từ T2, FLAIR, T1-contrast
- Dataset tương đối nhỏ (5,000 ảnh training), chưa đánh giá per-class accuracy đầy đủ
- Train từ đầu (from scratch), chưa thử fine-tune từ checkpoint pretrained

---

## 6. Hướng tiếp theo

1. **Benchmark thêm model** — chạy QKFormer (NeurIPS 2024), SDT, SGLFormer trên cùng dataset để có bảng so sánh SNN architectures
2. **Mở rộng sang BraTS2023** — chuyển đổi volume 3D → 2D slice, áp dụng SNN cho bài toán segmentation khối u não với multi-modal MRI (T1c, T1n, T2f, T2w)
3. **Đánh giá năng lượng** — đo synaptic operations (SOP) để so sánh hiệu quả năng lượng của SNN so với ViT thường

---

## 7. Môi trường thực nghiệm

| Thành phần | Phiên bản |
|---|---|
| GPU | NVIDIA Quadro RTX 5000 (16 GiB, Turing) |
| CUDA Driver | 13.2 |
| PyTorch | 2.4.1+cu121 |
| spikingjelly | 0.0.0.0.14 |
| timm | 0.5.4 |
| Python | 3.10.12 |
| Framework | STEP (NeurIPS 2025) |

---

## Tài liệu tham khảo

- Zhou et al., *Spikformer: When Spiking Neural Network Meets Transformer*, ICLR 2023
- Fateh et al., *BRISC: Annotated dataset for brain tumor segmentation and classification with Swin-HafNet*, arXiv:2506.14318, 2025
- Yao et al., *STEP: A Unified Spiking Transformer Evaluation Platform*, NeurIPS 2025
