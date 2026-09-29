import torch
from src.preprocessing import tokenize


def analyze_gates(model, text, vocab, max_len):
    """
    Chạy model trên 1 câu, trả về list[dict] mỗi dict là 1 token với:
      token, f_mean, i_mean, delta_c_norm, prediction, probability

    Yêu cầu: model.rnn là MyLSTM và impl="custom".
    """
    tokens  = tokenize(text)
    ids     = vocab.encode(tokens, max_len)
    T_real  = min(len(tokens), max_len)

    x       = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([T_real], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        emb = model.embedding(x)
        _, _, gates = model.rnn(emb, lengths, return_gates=True)
        logit = model(x, lengths)
        prob  = torch.sigmoid(logit).item()

    rows = []
    c    = torch.zeros(model.rnn.hidden_size)
    for t in range(T_real):
        f     = gates["f"][0, t].cpu()
        i_g   = gates["i"][0, t].cpu()
        g     = gates["g"][0, t].cpu()
        c_new = f * c + i_g * g
        rows.append({
            "token":       tokens[t] if t < len(tokens) else "<PAD>",
            "f_mean":      round(f.mean().item(), 4),
            "i_mean":      round(i_g.mean().item(), 4),
            "delta_c":     round((c_new - c).norm().item(), 4),
        })
        c = c_new

    label = "positive" if prob >= 0.5 else "negative"
    return rows, label, round(prob, 4)


def print_gate_table(rows, label, prob, text=""):
    if text:
        print(f'\nText: "{text}"')
    print(f"Prediction: {label} ({prob:.1%})\n")
    print(f"{'token':<16} {'f_mean':>8} {'i_mean':>8} {'Δ‖c‖':>9}")
    print("-" * 46)
    for r in rows:
        print(f"{r['token']:<16} {r['f_mean']:>8.4f} {r['i_mean']:>8.4f} {r['delta_c']:>9.4f}")
