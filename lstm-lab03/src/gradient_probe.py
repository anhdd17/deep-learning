import torch
import torch.nn as nn


def per_position_grad(model, input_ids, lengths, labels, loss_fn, max_positions=200):
    """
    Đo ‖∂L/∂e_t‖ trung bình theo khoảng cách từ token thật cuối.

    Returns: tensor (max_positions,) — index 0 = token cuối, index k = cách cuối k bước.
    """
    model.zero_grad()
    emb = model.embedding(input_ids)   # (B, T, E)
    emb.retain_grad()                  # intermediate tensor cần retain để có .grad sau backward
    logits = model.forward_from_embeddings(emb, lengths)
    loss_fn(logits, labels.float()).backward()

    g = emb.grad.norm(dim=-1)          # (B, T) — norm gradient tại mỗi timestep

    accum  = torch.zeros(max_positions)
    counts = torch.zeros(max_positions)

    for b in range(input_ids.shape[0]):
        L     = lengths[b].item()
        g_seq = g[b, :L]               # (L,) — chỉ token thật, bỏ pad
        g_rev = g_seq.flip(0)          # đảo: index 0 = token cuối
        take  = min(L, max_positions)
        accum[:take]  += g_rev[:take].detach().cpu()
        counts[:take] += 1

    return accum / counts.clamp(min=1)


def collect_grad_profile(model, val_loader, loss_fn, device,
                         min_length=200, max_batches=20, max_positions=200):
    """
    Chạy per_position_grad trên nhiều batch, chỉ lấy sample có length >= min_length.
    Trả về tensor (max_positions,) trung bình toàn bộ.
    """
    model.eval()
    total  = torch.zeros(max_positions)
    counts = torch.zeros(max_positions)
    n_batches = 0

    for x, lengths, y in val_loader:
        mask = lengths >= min_length
        if mask.sum() == 0:
            continue

        x, lengths, y = x[mask], lengths[mask], y[mask]
        x       = x.to(device)
        lengths = lengths.to(device)
        y       = y.to(device)

        profile = per_position_grad(model, x, lengths, y, loss_fn, max_positions)
        valid   = (profile > 0).float()
        total  += profile * valid
        counts += valid

        n_batches += 1
        if n_batches >= max_batches:
            break

    return total / counts.clamp(min=1)
