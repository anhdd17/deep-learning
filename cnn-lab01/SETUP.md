# ML/DL Environment Setup Guide

Hướng dẫn cài đặt và kích hoạt môi trường cho project CNN trên macOS Apple Silicon (M1/M2/M3/M4).

---

## 1. Tổng quan — Tại sao cần env riêng?

Mỗi project ML/DL cần phiên bản thư viện khác nhau. Nếu cài tất cả vào một môi trường, chúng sẽ xung đột.

| Tình huống | Hậu quả nếu không có env riêng |
|---|---|
| Project A cần torch 1.x, Project B cần torch 2.x | Không thể cài song song |
| Cài thêm package mới | Có thể break package cũ |
| Làm việc nhóm | Mỗi người chạy version khác nhau |

**Công cụ:** Dùng **Conda** — quản lý cả Python version lẫn package, tốt nhất cho ML/DL.  
**Python version:** 3.12 — ổn định, torch 2.x tương thích đầy đủ. Tránh 3.13 (quá mới, nhiều thư viện ML chưa support).

---

## 2. Tạo và cài đặt env

### Bước 1 — Tạo env mới (chỉ làm 1 lần)

```bash
conda create -n ml python=3.12 -y
```

> Nếu đã có env `ml` rồi (như máy bạn) thì bỏ qua bước này.

### Bước 2 — Kích hoạt env

```bash
conda activate ml
```

Dấu nhắc terminal sẽ thay đổi thành `(ml)`:

```
(ml) potter@MacBook ~ %
```

### Bước 3 — Cài các package ML/DL

```bash
pip install torch torchvision pyyaml matplotlib jupyter
```

Hoặc dùng file `requirements.txt` của project:

```bash
pip install -r requirements.txt
```

**Tại sao dùng `pip` thay vì `conda install`?**  
PyTorch khuyến nghị cài qua pip trên macOS. Cài qua conda có thể lấy bản cũ hơn.

### Bước 4 — Kiểm tra cài đặt

```bash
python -c "import torch; print(torch.__version__)"
# Output mong đợi: 2.x.x
```

---

## 3. Kích hoạt env mỗi khi làm việc

### Trong Terminal

```bash
conda activate ml
```

Để thoát:

```bash
conda deactivate
```

### Trong VS Code

1. `Cmd+Shift+P` → gõ `Python: Select Interpreter`
2. Chọn: `/Users/<tên>/miniconda3/envs/ml/bin/python`

Sau đó Pylance sẽ tìm đúng thư viện, lỗi import màu đỏ sẽ mất.

### Trong Jupyter Notebook

```bash
conda activate ml
jupyter notebook
# Hoặc:
jupyter lab
```

Khi mở notebook, chọn kernel **ml** ở góc trên bên phải.  
Nếu không thấy kernel `ml`:

```bash
conda activate ml
python -m ipykernel install --user --name ml --display-name "Python (ml)"
```

Sau đó restart Jupyter và chọn lại kernel.

---

## 4. Kiểm tra GPU — Apple Silicon MPS

PyTorch hỗ trợ GPU Apple Silicon qua **MPS** (Metal Performance Shaders) — thay thế CUDA trên Mac.

```bash
conda activate ml
python -c "
import torch
print('Torch version :', torch.__version__)
print('MPS available :', torch.backends.mps.is_available())
print('MPS built     :', torch.backends.mps.is_built())
"
```

Kết quả mong đợi trên M4:

```
Torch version : 2.x.x
MPS available : True
MPS built     : True
```

**Ý nghĩa thực tế:**  
Khi training CNN, model và data được đẩy lên GPU Apple Silicon → nhanh hơn 5–10x so với CPU thuần.  
Code trong `utils.py` sẽ tự detect:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

---

## 5. Workflow hàng ngày

```
1. Mở terminal
2. conda activate ml          ← luôn làm đầu tiên
3. Mở VS Code (đã chọn đúng interpreter rồi thì không cần làm lại)
4. Code / chạy script
5. conda deactivate           ← khi xong (hoặc chỉ cần đóng terminal)
```

---

## 6. Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'torch'`

**Nguyên nhân:** VS Code hoặc terminal đang dùng sai Python (không phải env `ml`).

```bash
which python      # Kiểm tra đang dùng Python nào
conda activate ml
which python      # Phải thấy: .../miniconda3/envs/ml/bin/python
```

---

### Lỗi: Pylance vẫn báo đỏ dù đã cài torch

**Nguyên nhân:** VS Code chưa chọn đúng interpreter.

→ `Cmd+Shift+P` → `Python: Select Interpreter` → chọn env `ml`.

---

### Lỗi: `MPS available: False`

**Nguyên nhân:** Torch version quá cũ hoặc macOS cũ.

```bash
pip install --upgrade torch torchvision
```

Yêu cầu tối thiểu: macOS 12.3+, torch 1.12+.

---

### Xem danh sách env đang có

```bash
conda env list
```

### Xem các package đã cài trong env hiện tại

```bash
pip list
```

### Xóa env (nếu cần làm lại từ đầu)

```bash
conda deactivate
conda env remove -n ml
```
