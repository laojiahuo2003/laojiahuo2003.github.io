# -*- coding: utf-8 -*-
"""SwiGLU 博文验证脚本：验证文中全部数字。"""
import torch
import torch.nn.functional as F

print("=== ① SiLU 关键数字（正文 §2）===")
x = torch.tensor([-1.2785, -1.0, 0.0, 1.0, 2.0])
y = F.silu(x)
for xi, yi in zip(x.tolist(), y.tolist()):
    print(f"SiLU({xi:+.4f}) = {yi:+.4f}")
# 最小值位置数值搜索
grid = torch.linspace(-3, 0, 30001)
vals = F.silu(grid)
imin = vals.argmin().item()
print(f"最小值: x={grid[imin]:.4f}, SiLU={vals[imin]:.4f} (正文: -1.28, -0.278)")

print()
print("=== ② 参数账（正文 §4）===")
old = 2 * 4096 * 16384
new = 3 * 4096 * 11008
print(f"老 FFN 2x4096x16384 = {old/1e6:.1f} M")
print(f"SwiGLU 3x4096x11008 = {new/1e6:.1f} M")
print(f"差 {(new/old-1)*100:.2f}%  (正文: 0.8%)")
print(f"16384*2/3 = {16384*2/3:.1f} -> 256 倍数取整 = {((16384*2//3)//256+1)*256}")

print()
print("=== ③ 手算案例（正文 §5）===")
xt = torch.tensor([[1., 2.]])
W_gate = torch.tensor([[1., 0.], [0., 1.]])
W_up   = torch.tensor([[1., 1.], [1., 2.]])
W_down = torch.tensor([[1., 0.], [0., 1.]])

gate = xt @ W_gate          # [1,2]
up   = xt @ W_up            # [3,5]
print("sigma(1) =", torch.sigmoid(torch.tensor(1.)).item())   # 0.731
print("sigma(2) =", torch.sigmoid(torch.tensor(2.)).item())   # 0.881
silu_gate = F.silu(gate)
print("门路 SiLU([1,2]) =", silu_gate.tolist())               # [0.731, 1.762]
mul = silu_gate * up
print("相乘 =", mul.tolist())                                  # [2.193, 8.808]
out = mul @ W_down
print("输出 =", out.tolist())                                  # [2.193, 8.808]
assert abs(out[0,0].item() - 2.1930) < 5e-4
assert abs(out[0,1].item() - 8.8080) < 5e-4
print("✓ 与正文手算一致")

print()
print("=== ④ 两代 FFN 实现对照（正文 §6）===")
import torch.nn as nn

class FFN_ReLU(nn.Module):
    def __init__(self, dim=8, hidden=32):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

class FFN_SwiGLU(nn.Module):
    def __init__(self, dim=8, hidden=21):          # 32*2/3≈21.3
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj   = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

torch.manual_seed(0)
x = torch.randn(2, 5, 8)
a, b = FFN_ReLU(), FFN_SwiGLU()
pa = sum(p.numel() for p in a.parameters())
pb = sum(p.numel() for p in b.parameters())
print(f"FFN_ReLU   参数 {pa}  输出 {tuple(a(x).shape)}")
print(f"FFN_SwiGLU 参数 {pb}  输出 {tuple(b(x).shape)}  (参数比 {pb/pa:.2f})")
print("✓ 两代实现前向形状一致, 参数量约打平")
