# So sánh Spikformer: CIFAR-10 vs BRISC2025

**Ngày:** 08/10/2026  
**Người thực hiện:** anhdd  
**Server:** gpu2 (NVIDIA Quadro RTX 5000, 16 GiB VRAM)

---

## 1. Tổng quan 2 thử nghiệm

| | Spikformer CIFAR-10 | Spikformer BRISC2025 |
|---|---|---|
| Backbone | spikformer_cifar | spikformer_imagenet |
| Tập dữ liệu | CIFAR-10 (ảnh tự nhiên) | BRISC2025 (MRI não T1) |
| Số lớp | 10 | 4 |
| Kích thước ảnh | 32 × 32 | 224 × 224 |
| Patch size | 4 | 16 |
| Embed dim | 384 | 384 |
| Num heads | 12 | 8 |
| Depths | 4 | 4 |
| Timestep T | 4 | 2 |
| Tổng tham số | ~9.3M | ~9.3M |
| Neuron | LIFNode (τ=2.0, threshold=1.0) | LIFNode (τ=2.0, threshold=1.0) |

---

## 2. Cấu hình huấn luyện

| | CIFAR-10 | BRISC2025 |
|---|---|---|
| Epochs | 400 (dừng ở 353) | 100 |
| Optimizer | AdamW | AdamW |
| Learning rate | 5e-4 (cosine) | 3e-4 (cosine) |
| Weight decay | 0.06 | 0.05 |
| Batch size | 128 | 8 |
| Warmup | 20 epochs | 10 epochs |
| Mixup | 0.5 (tắt ở epoch 200) | Tắt |
| Cutmix | 0.0 | 0.0 |
| RandAugment | rand-m9-n1 | m=7, n=2 |
| Random Erasing | 0.25 | — |
| Label smoothing | 0.1 | 0.1 |
| AMP (FP16) | ✓ | ✓ |

---

## 3. Kết quả

| | CIFAR-10 | BRISC2025 |
|---|---|---|
| **Accuracy đạt được** | **94.94%** | **93.20%** |
| Baseline paper | 95.41% (Spikformer ICLR 2023) | Không có |
| Gap với paper | -0.47% | — |
| Best epoch | 349 / 400 | 80 / 100 |
| Thời gian training | ~22 giờ | ~3 giờ |

---

## 4. Phân tích so sánh

### 4.1 Hiệu quả theo kích thước dữ liệu

CIFAR-10 có **50,000** ảnh training (gấp 10× BRISC2025), nhưng accuracy chỉ cao hơn ~1.7%. Điều này cho thấy Spikformer hoạt động tốt ngay cả với dataset nhỏ (~5,000 ảnh) — phù hợp cho bài toán y tế nơi dữ liệu có nhãn khan hiếm.

### 4.2 Ảnh hưởng của timestep T

- CIFAR-10 dùng **T=4**: model có 4 bước xử lý thời gian, bắt được dynamics phức tạp hơn trên ảnh tự nhiên đa dạng.
- BRISC2025 dùng **T=2**: đủ cho ảnh MRI có cấu trúc không gian rõ ràng, giảm chi phí tính toán.

Tăng T làm tăng accuracy nhưng cũng tăng thời gian inference tuyến tính theo T.

### 4.3 Lý do gap với paper CIFAR-10

Paper gốc đạt 95.41%, kết quả thực nghiệm là 94.94% (gap 0.47%). Nguyên nhân chính:

1. **`cutmix: 0.0`** — paper dùng `cutmix=0.5`, đây là augmentation đóng góp ~0.3–0.5%
2. **Training bị interrupt** — dừng ở epoch 353/400, chưa chạy hết cosine schedule
3. **CuPy không được kích hoạt** — STEP framework dùng BrainCog (pure PyTorch) thay vì SpikingJelly với CuPy; không ảnh hưởng accuracy nhưng làm training chậm ~2–4×

### 4.4 Nhầm lẫn đặc trưng theo domain

**CIFAR-10:** Nhầm lẫn chủ yếu giữa các lớp có hình dạng tương đồng (dog/cat, automobile/truck).

**BRISC2025:** Nhầm lẫn cao nhất giữa Glioma ↔ Meningioma — phù hợp với thực tế lâm sàng vì cả hai đều có thể xuất hiện dưới dạng khối tăng tín hiệu trên T1. Trong thực tế cần thêm chuỗi T2/FLAIR/DWI để phân biệt chính xác.

### 4.5 Chi phí tính toán

| | CIFAR-10 | BRISC2025 |
|---|---|---|
| Thời gian/epoch | ~3.5 phút | ~1.8 phút |
| Tổng training | ~22 giờ | ~3 giờ |
| Nguyên nhân | T=4, 50k ảnh, aug nặng | T=2, 5k ảnh, batch nhỏ |

CIFAR-10 tốn thời gian gấp ~7× BRISC2025 do dataset lớn hơn và T lớn hơn. Cả hai đều bị chậm thêm do STEP dùng BrainCog thay vì SpikingJelly+CuPy.

---

## 5. Kết luận

- Spikformer hoạt động tốt trên **cả hai domain** — ảnh tự nhiên lẫn ảnh y tế — với accuracy cạnh tranh.
- Với dataset nhỏ như BRISC2025, mô hình vẫn hội tụ nhanh và đạt accuracy cao sau 100 epoch.
- **Lần thử nghiệm tiếp theo nên bổ sung `cutmix: 0.5`** để thu hẹp gap với paper trên CIFAR-10.
- Hướng mở rộng: benchmark thêm QKFormer, SDT trên BRISC2025 để có bảng so sánh đa kiến trúc SNN trên dữ liệu y tế.
