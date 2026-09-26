import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Giá trị mean/std tính sẵn trên toàn bộ 50,000 ảnh train của CIFAR-10
# Dùng để normalize: pixel = (pixel - mean) / std → phân phối ≈ N(0,1)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # per channel: R, G, B
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_transforms(train: bool) -> transforms.Compose:
    """
    Train: augmentation + normalize  →  model thấy ảnh hơi khác nhau mỗi epoch
    Val:   chỉ normalize             →  đánh giá trên ảnh gốc, không biến đổi
    """
    if train:
        return transforms.Compose([
            # Pad thêm 4px mỗi bên rồi crop ngẫu nhiên về 32×32
            # → model học không phụ thuộc vị trí vật thể trong ảnh
            transforms.RandomCrop(32, padding=4),

            # Lật ngang ngẫu nhiên 50%
            # → mèo nhìn trái hay phải đều là mèo
            transforms.RandomHorizontalFlip(),

            # PIL Image (H, W, C) uint8 [0,255] → Tensor (C, H, W) float [0,1]
            transforms.ToTensor(),

            # (pixel - mean) / std → phân phối ≈ N(0,1)
            # Giúp gradient flow ổn định, hội tụ nhanh hơn
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    use_pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
    val_size: int = 5000,
) -> tuple[DataLoader, DataLoader]:
    """
    Trả về (train_loader, val_loader).

    Split 50,000 ảnh train thành 45,000 train + val_size val.
    10,000 ảnh test (train=False) được giữ riêng cho build_test_loader(),
    không đụng vào trong quá trình train/tune.
    """
    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=build_transforms(train=True),
    )
    # val dùng transform không augment để đo chính xác
    full_train_no_aug = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=build_transforms(train=False),
    )

    n = len(full_train)  # 50,000
    indices = list(range(n))
    # val lấy từ cuối để reproducible khi fix seed bên ngoài
    train_idx, val_idx = indices[:-val_size], indices[-val_size:]

    from torch.utils.data import Subset
    train_dataset = Subset(full_train,        train_idx)
    val_dataset   = Subset(full_train_no_aug, val_idx)

    workers = num_workers if num_workers > 0 else min(8, os.cpu_count() or 4)

    train_loader = _make_loader(train_dataset, batch_size, shuffle=True,  num_workers=workers)
    val_loader   = _make_loader(val_dataset,   batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader


def build_test_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """
    Trả về test_loader từ 10,000 ảnh CIFAR-10 (train=False).
    Chỉ gọi 1 lần duy nhất sau khi chọn xong model — không dùng để tune.
    """
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=build_transforms(train=False),
    )
    workers = num_workers if num_workers > 0 else min(8, os.cpu_count() or 4)
    return _make_loader(test_dataset, batch_size, shuffle=False, num_workers=workers)


# Bạn nói đúng, tôi giải thích rải rác, chưa có bức tranh tổng thể. Để tôi làm lại đầy đủ.                         
                                         
#   ---                                                                                                              
#   Training DL gồm những bước nào?                                                                                  
                                                                                                                   
#   Một vòng lặp training lặp đi lặp lại điều này:            
                                                                                                                   
#   ┌─────────────┐                                           
#   │   data.py   │  1. Lấy batch ảnh + label từ dataset                                                             
#   └──────┬──────┘                                                                                                  
#          │ (64, 3, 32, 32)                                                                                         
#   ┌──────▼──────┐                                                                                                  
#   │   model.py  │  2. Cho ảnh chạy qua CNN → ra logits (64, 10)                                                    
#   └──────┬──────┘                                                                                                  
#          │ logits                                                                                                  
#   ┌──────▼──────┐                                                                                                  
#   │    Loss     │  3. So sánh logits với label đúng → ra 1 số (loss)                                               
#   └──────┬──────┘                                                                                                  
#          │ loss                                                                                                    
#   ┌──────▼──────┐                                                                                                  
#   │   Backprop  │  4. Tính gradient: mỗi weight ảnh hưởng loss thế nào?                                            
#   └──────┬──────┘                                                                                                  
#          │ gradients                                                                                               
#   ┌──────▼──────┐                                                                                                  
#   │  Optimizer  │  5. Cập nhật weights theo gradient → model "học"                                                 
#   └──────┬──────┘                                                                                                  
#          │                                                                                                         
#          └──────────────────────────── lặp lại từ đầu                                                              
                                                                                                                   
#   data.py là bước 1 — không có nó thì không có gì để train.                                                        
                                                                                                                   
#   ---                                                                                                              
#   data.py gồm những gì?                                     
                                                                                                                   
#   data.py
#   ├── CIFAR10_MEAN, CIFAR10_STD   → hằng số normalize (tính sẵn từ dataset)                                        
#   ├── CLASSES                     → mapping: index 0→"airplane", 1→"automobile"...                                 
#   ├── build_transforms(train)     → pipeline biến ảnh thành tensor                                                 
#   │   ├── [train]  RandomCrop     → cắt ngẫu nhiên (augmentation)                                                  
#   │   ├── [train]  RandomFlip     → lật ngang (augmentation)                                                       
#   │   ├── [cả hai] ToTensor       → PIL (H,W,C) uint8 → Tensor (C,H,W) float                                       
#   │   └── [cả hai] Normalize      → pixel = (pixel - mean) / std                                                   
#   └── build_dataloaders()         → tạo DataLoader cho train và val                                                
#       ├── Dataset  = cái kho ảnh (biết lấy ảnh thứ i ra trả gì)                                                    
#       └── DataLoader = người đóng batch (gộp N ảnh, shuffle, dùng nhiều worker)                                    
                                                                                                                   
#   ---                                                                                                              
#   Luồng 1 ảnh đi qua data.py                                                                                       
                                                                                                                   
#   file ảnh trên disk                                        
#         ↓  Dataset đọc lên                                                                                         
#   PIL Image  (32, 32, 3)  — uint8  [0, 255]                                                                        
#         ↓  RandomCrop + RandomFlip  (chỉ train)                                                                    
#   PIL Image  đã biến đổi                                                                                           
#         ↓  ToTensor                                                                                                
#   Tensor     (3, 32, 32)  — float  [0.0, 1.0]                                                                      
#         ↓  Normalize                                                                                               
#   Tensor     (3, 32, 32)  — float  ≈ [-2.0, 2.0]                                                                   
#         ↓  DataLoader gom 64 ảnh lại                                                                               
#   Tensor     (64, 3, 32, 32)  ← đây mới là thứ model nhận                                                          
#   Label      (64,)             ← 64 con số nguyên 0–9                                                              
                                                                                                                   
#   ---  