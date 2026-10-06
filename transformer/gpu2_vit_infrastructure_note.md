# GPU2 — Note hạ tầng training và serving Vision Transformer (ViT)

Ngày ghi nhận: **06/10/2026**. Máy: `gpu2`. Tài khoản: `cs1_user6`.

Tài liệu dựa trên output người dùng cung cấp lúc 08:05; chưa chạy benchmark hoặc kiểm tra trực tiếp trên máy. Các cấu hình dưới đây là điểm khởi đầu để đo, không phải cam kết batch size, latency hay throughput.

## 1. Cấu hình và hiện trạng

| Thành phần | Ghi nhận | Ý nghĩa vận hành |
|---|---|---|
| GPU | 1 × NVIDIA Quadro RTX 5000, 16 GiB VRAM | Kiến trúc Turing, compute capability 7.5; không phải RTX 5000 Ada |
| GPU tại thời điểm kiểm tra | 6 MiB VRAM, utilization 0%, 34°C | Gần như rảnh; chỉ thấy Xorg, chưa thấy job compute |
| Driver | 595.45.04 | Phiên bản theo output cung cấp |
| CUDA trên `nvidia-smi` | 13.2 | Mức CUDA driver hỗ trợ, không phải bằng chứng Toolkit 13.2 đã được cài |
| CPU | `nproc = 12` | 12 processing units khả dụng; chưa xác định nhân vật lý/model CPU |
| RAM | 15 GiB tổng, khoảng 13 GiB available | Cần kiểm soát dataset cache, worker và bộ nhớ khi load checkpoint |
| Swap | 9.7 GiB, chưa dùng | Không thay thế RAM/VRAM; swap nhiều có thể làm job chậm mạnh |
| Filesystem chứa home | `/dev/mapper/vgubuntu-root`, mount `/` | Home cùng filesystem với root |
| Dung lượng | 877G tổng, 741G dùng, 92G available, 90% | 92G là dung lượng filesystem tại thời điểm kiểm tra, không phải quota riêng |
| Phiên đăng nhập | `gpu` trên tty7; `cs1_user6` trên pts/0 | `who` không cho biết toàn bộ job nền hoặc container |
| sudo | `sudo -v` không hiện lỗi | Xác thực thành công; xem `sudo -l` để biết phạm vi quyền |

Chưa thể kết luận RAM là điểm nghẽn chỉ vì RAM nhỏ hơn VRAM. Cần đo tải thực tế. Chưa có dữ liệu về tốc độ ổ đĩa, quota, distro/glibc, Toolkit, bản PyTorch hoặc giới hạn tài nguyên của lab.

## 2. Phạm vi phù hợp cho ViT

- Ưu tiên fine-tune pretrained ViT và thử nghiệm học thuật; bắt đầu với ViT nhỏ hoặc ViT-B/16 ở ảnh 224×224.
- Có thể thử train từ đầu ở quy mô nhỏ; pretraining ViT lớn trên dataset lớn cần nhiều compute và thời gian hơn đáng kể.
- Serving một model phân loại ảnh/embedding là hướng phù hợp để bắt đầu; số model đồng thời và SLA phải dựa trên benchmark.
- ViT-L/H, ảnh độ phân giải cao, nhiều crop hoặc backbone kèm decoder có thể vượt giới hạn dù batch nhỏ.

Với ViT chuẩn dùng patch 16×16 và một CLS token:

| Kích thước ảnh | Số token | Tỷ lệ kích thước ma trận attention so với 224 |
|---|---:|---:|
| 224×224 | 197 | 1× |
| 384×384 | 577 | khoảng 8.6× |
| 448×448 | 785 | khoảng 15.9× |

Attention chuẩn có phần tính toán tăng theo bình phương số token. Bảng không có nghĩa tổng VRAM hay tổng thời gian tăng đúng cùng tỷ lệ; backend tối ưu có thể không lưu toàn bộ ma trận attention. Đổi resolution còn cần model hỗ trợ kích thước mới và xử lý positional embedding đúng cách.

## 3. Precision và tương thích

- Training: ưu tiên **FP16 autocast + GradScaler**. Giữ tham số model ở FP32 khi dùng AMP thông thường.
- Turing không có BF16 native hoặc TF32. Không chọn BF16 làm mặc định; một số phép toán có thể không hỗ trợ.
- FlashAttention 2 trong bản CUDA chính của thư viện không hỗ trợ Turing; có triển khai riêng nhưng không cần thêm vào baseline.
- Có thể dùng PyTorch SDPA với backend tự chọn tương thích; không ép FlashAttention backend.
- Driver mới không đảm bảo mọi wheel/extension đều hỗ trợ GPU này. Kiểm tra build và chạy phép toán thực tế.
- Dùng wheel PyTorch thường không cần cài CUDA Toolkit hệ thống. Compile CUDA extension có thể cần Toolkit và compiler phù hợp.
- Bắt đầu bằng eager execution; chỉ thử `torch.compile` hay engine khác sau khi baseline đúng và đã đo.

## 4. Kiểm tra trước khi cài/chạy

```bash
hostname
uname -m
cat /etc/os-release
lscpu
nproc
free -h
df -h "$HOME"
df -i "$HOME"
nvidia-smi
command -v conda
command -v python
command -v nvcc
sudo -l
```

Không tự thay driver, Python hệ thống hoặc dịch vụ của lab. Kiểm tra lịch/quy định sử dụng GPU; GPU rảnh tại một thời điểm không đồng nghĩa được cấp riêng. Chỉ đọc log hay quản trị bằng sudo trong phạm vi được cấp.

## 5. Môi trường Python

Dùng Conda có sẵn nếu phù hợp; nếu chưa có, cài Miniconda vào home sau khi kiểm tra kiến trúc máy và đường dẫn chưa tồn tại. Không cài đè môi trường người dùng đang dùng.

Ví dụ tạo env mới, không cần sudo:

```bash
conda create -n vit python=3.12 -y
conda activate vit
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip check
```

`cu128` là lựa chọn khởi điểm; lệnh không pin version nên kết quả có thể thay đổi theo ngày cài. Sau smoke test thành công, ghi lại phiên bản thực tế và dùng bộ phiên bản đó để tái lập. Chỉ cài `timm`, `transformers` hoặc framework serving khi project cần.

Smoke test tính toán GPU và FP16 backward:

```bash
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
print('Build architectures:', torch.cuda.get_arch_list())
x = torch.randn(512, 512, device='cuda', requires_grad=True)
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss = (x @ x.T).float().square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss).item()
assert torch.isfinite(x.grad).all().item()
print('FP16 forward/backward: OK')
PY
```

Đạt smoke test chưa có nghĩa model ViT cụ thể đã chạy được. Tiếp tục thử model, loss và optimizer thật trong ít nhất một nhóm gradient accumulation đầy đủ, bao gồm optimizer step đầu tiên vì optimizer có thể cấp phát state lúc đó.

Lưu cấu hình môi trường trong thư mục project:

```bash
python -m pip freeze > requirements.lock.txt
conda env export --no-builds > environment.yml
nvidia-smi > gpu_snapshot.txt
```

## 6. Cấu hình training khởi điểm

Áp dụng để thử fine-tune ViT-B/16, ảnh 224×224, một GPU; chưa benchmark:

```python
cfg = dict(
    device="cuda",
    amp_dtype="float16",
    image_size=224,
    batch_size=8,
    grad_accum=4,              # effective batch = 32 cho nhóm đủ 4 batch
    num_workers=2,            # thử 4 sau khi đo RAM và tốc độ nạp dữ liệu
    pin_memory=True,
    grad_checkpointing=False,
    save_every_epochs=1,
    keep_last=True,
    keep_best=True,
)
```

Đây là schema đề xuất cho project, không phải API config tự hoạt động của PyTorch. Code phải đọc từng key; `amp_dtype` phải được ánh xạ sang `torch.float16`. Không phải model nào cũng có sẵn gradient checkpointing.

### Nguyên tắc training loop

1. `model.train()`; tạo `torch.amp.GradScaler("cuda")` khi training FP16.
2. DataLoader trả tensor CPU, chuyển sang GPU bằng `.to("cuda", non_blocking=True)` khi phù hợp.
3. Forward/loss trong `torch.autocast("cuda", dtype=torch.float16)`; backward qua scaler.
4. Chuẩn hóa loss theo nhóm accumulation. Với 4 microbatch bằng kích thước nhau, chia loss cho 4. Nhóm cuối thiếu batch hoặc khác số mẫu cần chuẩn hóa theo số mẫu thực; framework có thể đã xử lý nên không chia hai lần.
5. Chỉ optimizer step, scaler update và zero grad ở biên nhóm. Nếu clip gradient, unscale trước khi clip; scheduler theo step cập nhật theo optimizer update thực tế.
6. Validation dùng `model.eval()` và inference mode; không giữ graph/tensor GPU trong danh sách log.

Batch hiệu dụng = microbatch × số lần accumulation × số GPU. Gradient accumulation không làm GPU giữ đồng thời cả 32 mẫu. Nó cũng không hoàn toàn tương đương batch lớn nếu model có BatchNorm; ViT chuẩn thường dùng LayerNorm.

### Khi thiếu tài nguyên

| Hiện tượng | Điều chỉnh đầu tiên |
|---|---|
| CUDA OOM | Batch 8→4, accumulation 4→8; nếu cần 2/16 |
| Vẫn CUDA OOM | Bật checkpointing nếu model hỗ trợ; dùng model nhỏ hơn, freeze backbone hoặc giảm resolution hợp lệ |
| RAM tăng mạnh | Giảm worker, prefetch và cache; không giữ toàn dataset ở RAM |
| GPU thường chờ | Đo decode/augmentation/I/O; thử tăng worker 2→4 nếu RAM còn đủ |
| Loss/gradient NaN | Kiểm tra data/loss/LR/scaler; thử FP32 để khoanh vùng |

Không tự đổi learning rate chỉ vì thay microbatch nếu effective batch giữ nguyên. Cần chọn LR theo pretrained checkpoint, optimizer, dataset và chiến lược fine-tune; hạ tầng không quyết định LR phù hợp.

## 7. Serving ViT trên một GPU

Baseline: **một process sở hữu model GPU**, một luồng xử lý inference có queue giới hạn; bắt đầu batch 1. Không chạy nhiều server worker cùng load model lên GPU vì mỗi worker thường tạo bản sao model và CUDA context.

| Tham số | Khởi điểm |
|---|---|
| GPU model process | 1 |
| Batch inference | 1; benchmark thêm 2/4/8 nếu cần throughput |
| Số batch inference đồng thời trên GPU | 1 trước khi đo |
| Precision | FP16 autocast, đối chiếu chất lượng với FP32 |
| Chế độ model | `model.eval()` + `torch.inference_mode()` |
| Queue/timeout | Đặt giới hạn theo SLA; reject khi quá tải |
| Training cùng GPU | Ưu tiên tách thời gian khi serving cần latency ổn định |

Ví dụ lõi inference; `model` đã load đúng kiến trúc/weights, `images` đã preprocess thành tensor CPU:

```python
model = model.to("cuda").eval()

@torch.inference_mode()
def predict(images):
    images = images.to("cuda", non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(images)
    return logits.float().softmax(dim=-1).cpu()
```

Ví dụ giả định model trả logits Tensor cho classification một nhãn. Với embedding, multilabel hoặc model trả object, thay phần lấy output/postprocess tương ứng. Serving không cần GradScaler.

### Hợp đồng đầu vào và model artifact

- Giữ đúng resize/crop, interpolation, RGB, normalization và resolution của checkpoint. Validation/serving dùng transform xác định, không dùng augmentation ngẫu nhiên của training.
- Lưu model ID/architecture, weight version, class mapping, preprocessing config và package versions cùng artifact.
- Giới hạn kích thước upload, số pixel sau decode, số ảnh/request và queue để tránh request làm cạn RAM/VRAM.
- Load model một lần lúc startup; warm-up với shape/batch dự kiến rồi mới báo ready.
- Thử FP16 trên validation set để kiểm tra metric và độ ổn định so với FP32.
- Khởi đầu PyTorch eager. Chỉ cân nhắc ONNX/TensorRT sau khi đo; xác minh phiên bản engine còn hỗ trợ Turing và kiểm tra sai lệch output sau chuyển đổi.
- Nếu cần training đồng thời serving, giới hạn batch chưa đảm bảo SLA; phải đo tải hỗn hợp và còn khoảng trống VRAM cho cả hai.

## 8. Đo hiệu năng và tiêu chí nghiệm thu

Không suy ra FPS từ tên GPU. Benchmark đúng model, checkpoint, resolution và pipeline thực.

| Training | Serving |
|---|---|
| Loss hữu hạn, gradient/optimizer update hoạt động | Output đúng shape, nhãn và preprocessing |
| Peak VRAM sau optimizer step | Peak VRAM sau warm-up và dưới tải |
| RAM available, swap, GPU utilization | RAM, queue length, lỗi/OOM/timeout |
| Images/second và thời gian mỗi epoch | Throughput, latency p50/p95/p99 |
| Validation metric và khả năng resume | Metric FP16 so với FP32 |

Với PyTorch, dùng `torch.cuda.reset_peak_memory_stats()` trước vùng đo và `torch.cuda.max_memory_allocated()` sau vùng đo. `nvidia-smi` gồm cả bộ nhớ CUDA ngoài tensor allocator nên hai số có thể khác nhau. Khi đo thời gian GPU bằng đồng hồ CPU, synchronize ở biên benchmark; không thêm synchronize vào mọi request production.

Đo riêng thời gian model và end-to-end gồm decode, preprocess, queue, chuyển CPU↔GPU, postprocess. Bỏ phần warm-up khỏi thống kê steady-state. Chưa có SLA nên chưa thể chốt batch phục vụ hoặc số request/giây.

## 9. Theo dõi, dung lượng và checkpoint

```bash
watch -n 2 nvidia-smi
free -h
vmstat 1
df -h "$HOME"
du -h --max-depth=1 "$HOME"
```

Nếu process bị `Killed`, kiểm tra log OOM nếu tài khoản được phép:

```bash
dmesg -T | tail -n 100
```

`Killed` chưa đủ chứng minh OOM; có thể do giới hạn cgroup, scheduler hoặc thao tác bên ngoài. Nếu không đọc được log, nhờ quản trị xác nhận thay vì tự thay cấu hình hệ thống.

- Đọc dataset theo nhu cầu; tránh nhân bản dataset/cache qua nhiều project.
- Checkpoint training để resume cần model, optimizer, scaler, scheduler nếu có, epoch/update counter, config; thêm RNG/sampler state khi cần tái lập chặt chẽ.
- Giữ `last` cho resume và `best` theo metric validation đã chọn. Artifact serving thường chỉ cần weights và metadata cần suy luận.
- Ghi checkpoint vào file tạm rồi rename trong cùng filesystem để giảm nguy cơ mất checkpoint tốt khi ghi lỗi; cần chừa chỗ cho cả file cũ và file tạm.
- Dung lượng env/checkpoint không cố định. Theo dõi trước mỗi run; ổ root đang 90% nên đặt ngân sách dataset, cache và checkpoint cùng quản trị.
- Chỉ dọn cache của mình khi cần; kiểm tra đường dẫn cache dùng chung và tác động trước khi chạy `conda clean -a` hoặc `python -m pip cache purge`.

## 10. Checklist trước một run

- [ ] Xác nhận quyền/lịch dùng GPU và dung lượng khả dụng.
- [ ] Env hoạt động; smoke test FP16 forward/backward đạt.
- [ ] Ghi model, weights, task, resolution, dataset split, preprocessing và seed.
- [ ] Thử một nhóm accumulation đầy đủ gồm optimizer step; đo peak VRAM/RAM.
- [ ] Training config khởi điểm 8×4; giảm batch nếu cần, không coi đó là mức đã được bảo đảm.
- [ ] Validation và checkpoint `last`/`best` hoạt động; thử resume khi cần chạy dài.
- [ ] Serving load một model process; warm-up, kiểm tra FP16 và đo latency dưới tải.
- [ ] Lưu phiên bản môi trường và kết quả benchmark cùng experiment.

## Nguồn tham khảo chính thức

- [NVIDIA — GPU compute capability](https://developer.nvidia.com/cuda/gpus)
- [NVIDIA — Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
- [NVIDIA — nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [PyTorch — Các bản phát hành và lệnh cài](https://docs.pytorch.org/get-started/previous-versions/)
- [PyTorch — AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [Torchvision — VisionTransformer](https://docs.pytorch.org/vision/stable/models/vision_transformer.html)
- [FlashAttention — README và phạm vi GPU hỗ trợ](https://github.com/Dao-AILab/flash-attention)

Các ví dụ là hướng dẫn để chạy trên `gpu2`; tài liệu này không xác nhận đã cài đặt, training hoặc triển khai dịch vụ trên máy.
