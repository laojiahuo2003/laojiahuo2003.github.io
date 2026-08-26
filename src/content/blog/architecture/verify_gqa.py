# -*- coding: utf-8 -*-
"""GQA 博文验证脚本：验证文中给出的关键数字与代码语义。
环境: conda env torch (torch 1.13, 无 SDPA 的部分用 einsum 手算等价验证)
运行: python verify_gqa.py
"""
import torch
import torch.nn.functional as F

print("=== ① KV cache 算账（图 2 与正文表格的数字）===")
def kv_per_token(n_kv, n_layers=32, d_h=128, bytes_=2):
    return 2 * n_layers * n_kv * d_h * bytes_          # B / token

for name, n_kv in [("MHA n_kv=32", 32), ("GQA n_kv=8", 8), ("MQA n_kv=1", 1)]:
    pt = kv_per_token(n_kv)
    print(f"{name}: {pt/1024:.0f} KB/token | 4096ctx 单序列 {pt*4096/1024**3:.3f} GB"
          f" | batch=8 {pt*4096*8/1024**3:.2f} GB")

print()
print("=== ② repeat_kv 手算案例（n_kv=2, n_rep=2, d=4；正文引用）===")
def repeat_kv(h, n_rep):
    B, S, H_kv, D = h.shape
    if n_rep == 1:
        return h
    x = h[:, :, :, None, :]                  # unsqueeze(2): (B,S,8,1,128)
    x = x.expand(B, S, H_kv, n_rep, D)       # 视图, 0 拷贝
    return x.reshape(B, S, H_kv * n_rep, D)  # 一次拷贝

k = torch.tensor([[[[1., 0., 0., 0.], [0., 1., 0., 0.]]]])   # (1,1,2,4)
out = repeat_kv(k, 2)
print("k 头0=[1,0,0,0] 头1=[0,1,0,0]")
print("repeat_kv(k,2) =", out[0, 0].tolist())
assert torch.equal(out[0, 0, 0], out[0, 0, 1]) and out[0, 0, 0, 0] == 1.0
assert torch.equal(out[0, 0, 2], out[0, 0, 3]) and out[0, 0, 2, 1] == 1.0
print("头0 -> 槽位[0,1], 头1 -> 槽位[2,3] ✓ 与正文手算一致")

print()
print("=== ③ GQA(repeat_kv) == MHA(共享权重铺回) 数学等价性 ===")
def attn(q, k, v):                            # (B,S,H,D) -> (B,S,H*D)
    B, S, H, D = q.shape
    q_, k_, v_ = (t.transpose(1, 2) for t in (q, k, v))    # (B,H,S,D)
    scores = q_ @ k_.transpose(-1, -2) / D**0.5
    a = torch.softmax(scores, dim=-1) @ v_
    return a.transpose(1, 2).reshape(B, S, H * D)

torch.manual_seed(0)
B, S, Hq, Hkv, D, xdim = 2, 5, 8, 4, 16, 64
x = torch.randn(B, S, xdim)
Wq = torch.randn(xdim, Hq * D)
Wk_kv = torch.randn(Hkv, xdim, D)             # 每组一份的 K 投影 (4 头)
Wv_kv = torch.randn(Hkv, xdim, D)

q = (x @ Wq).view(B, S, Hq, D)
k_gqa = torch.einsum("bsd,hde->bshe", x, Wk_kv)
v_gqa = torch.einsum("bsd,hde->bshe", x, Wv_kv)

# 路径 A: 共享权重按组铺回 8 头 -> 普通 MHA
Wk_mha = Wk_kv[[h // (Hq // Hkv) for h in range(Hq)]]      # (8,64,16)
Wv_mha = Wv_kv[[h // (Hq // Hkv) for h in range(Hq)]]
outA = attn(q, torch.einsum("bsd,hde->bshe", x, Wk_mha),
               torch.einsum("bsd,hde->bshe", x, Wv_mha))

# 路径 B: repeat_kv 补头 -> 同一个 attention
outB = attn(q, repeat_kv(k_gqa, Hq // Hkv), repeat_kv(v_gqa, Hq // Hkv))
print("outA == outB:", torch.allclose(outA, outB, atol=1e-5),
      "| max diff:", (outA - outB).abs().max().item())

print()
print("=== ④ 参数量（正文引用）===")
print(f"W_q/W_o: {4096*4096/1e6:.1f} M | W_k/W_v GQA(4096x1024): {4096*1024/1e6:.1f} M"
      f" | MHA(4096x4096): {4096*4096/1e6:.1f} M")
print(f"每层 W_k+W_v: MHA {(2*4096*4096)/1e6:.1f}M -> GQA {(2*4096*1024)/1e6:.1f}M"
      f" | 32 层 fp16 差: {(2*4096*4096-2*4096*1024)*32*2/1024**3:.2f} GB")
