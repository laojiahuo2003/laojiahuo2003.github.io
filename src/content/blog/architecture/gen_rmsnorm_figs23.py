# -*- coding: utf-8 -*-
"""
RMSNorm 博客配图 · 图 2：前向数据流（张量形状标注）+ 图 3：公式↔代码对照
复用 gen_rmsnorm_figs.py 的引擎与 fit_box 自适应布局。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_rmsnorm_figs import Page, C, nid, fit_box, text_w, write_page

# ================================================================ 图 2：前向数据流
p2 = Page("2 · RMSNorm 前向数据流", w=900, h=760)

p2.add(x=60, y=24, w=780, h=36, label="前向数据流：两行核心代码的形状之旅",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

flow = [
    ("x  (B, S, d)", C["input"], "输入：batch×序列×维度"),
    ("x.pow(2)  →  x²   (B, S, d)", C["attn"], "逐元素平方"),
    (".mean(dim=-1, keepdim=True)", C["attn"], "最后一维归约 → (B, S, 1)"),
    ("+ eps 后 rsqrt", C["norm"], "1 / sqrt(mean(x²)+ε)   (B, S, 1)"),
    ("x · 上述结果  广播相乘", C["attn"], "(B, S, 1) 广播回 (B, S, d)"),
    ("× self.weight (γ)", C["norm"], "逐维缩放，输出 (B, S, d)"),
]
nodes = []
y = 90
for label, fill, note in flow:
    w = max(fit_box(label, 12)[0], 300)
    n = p2.add(x=220, y=y, w=w, h=44, label=label, fill=fill, fs=12)
    # 右侧注释（小字、无框）
    p2.add(x=220 + w + 20, y=y + 6, w=fit_box(note, 10.5)[0], h=32,
           label=note, fill=C["white"], fs=10.5)._stroke = "#FFFFFF"
    if nodes:
        p2.edge(nodes[-1], n, exit_=(0.5, 1), entry=(0.5, 0))
    nodes.append(n)
    y += 66

# γ 侧栏（虚线连到最后一行）
gamma = p2.add(x=60, y=90, w=130, h=60, label="γ 可学习\n初始化全 1",
               fill=C["gray"], fs=11)
p2.edge(gamma, nodes[-1], exit_=(0.5, 1), entry=(0, 0.5), dashed=True,
        color="#8E24AA")

# 底部要点框（每行独立测量宽度，整体框取最大行）
notes = [
    "keepdim=True 保持 (B,S,1)，广播的关键",
    "rsqrt = 融合算子：一次完成 1/sqrt，fp16 下更稳",
    "γ 初始为全 1：训练开始时不扰动初始表征",
]
nw = max(fit_box(s, 11.5)[0] for s in notes)
p2.add(x=(900 - nw) / 2, y=y + 10, w=nw, h=len(notes) * 11.5 * 1.9 + 24,
       label="\n".join(notes), fill=C["panel"], fs=11.5)

# ================================================================ 图 3：公式 ↔ 代码对照
p3 = Page("3 · 公式与代码对照", w=900, h=700)

p3.add(x=60, y=24, w=780, h=36, label="一行公式，两行代码",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

# 公式框
p3.add(x=60, y=84, w=780, h=52,
       label="y_i = γ_i · x_i / sqrt( (1/d)·Σ_j x_j² + ε )",
       fill=C["note"], fs=14, bold=True)

# 代码框（等宽逐行）
code_lines = [
    "class RMSNorm(nn.Module):",
    "    def __init__(self, dim, eps=1e-5):",
    "        self.eps = eps",
    "        self.weight = nn.Parameter(torch.ones(dim))   # γ",
    "",
    "    def forward(self, x):                            # x: (B,S,d)",
    "        return x * torch.rsqrt(                      # ②÷√(ms+ε)",
    "            x.pow(2).mean(-1, keepdim=True) + self.eps)  # ①ms=mean(x²)",
    "               ) * self.weight                       # ③×γ",
]
code_w = max(fit_box(s, 12)[0] for s in code_lines if s) + 40
ch = len(code_lines) * 12 * 1.7 + 24
p3.add(x=(900 - code_w) / 2, y=160, w=code_w, h=ch,
       label="\n".join(code_lines), fill="#F5F7FA", fs=12)

# 底部三个信息卡（每卡内容独立测量）
cards = [
    ("pre-norm 放置", "归一化在子层【前】\n\nx = x + Attn(Norm(x))\nx = x + FFN(Norm(x))", C["attn"]),
    ("真实超参", "Llama-2-7B: d=4096, ε=1e-5\nQwen3: d=4096, ε=1e-6\nDeepSeek-V3: d=7168, ε=1e-6", C["gray"]),
    ("敢去掉 β 的理由", "β 的自由度在大规模下\n收益≈0，且是量化时\n激活异常的惯犯", C["panel"]),
]
cw = 235
gap = (900 - 120 - cw * 3) / 2
for i, (title, body, color) in enumerate(cards):
    x0 = 60 + i * (cw + gap)
    card = p3.add(x=x0, y=420, w=cw, h=170, label=body, fill=color, fs=11)
    p3.add(x=x0, y=392, w=cw, h=26, label=title, fill=C["norm"], fs=12,
           bold=True)

p3.add(x=60, y=620, w=780, h=40,
       label="最后一层后还有一个 final RMSNorm 再接 lm_head——手写时最容易漏",
       fill="#FFF9C4", fs=11.5)

# ================================================================ 生成 + 验证
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
write_page(p2, os.path.join(OUT_DIR, "rmsnorm_2_forward.drawio"))
write_page(p3, os.path.join(OUT_DIR, "rmsnorm_3_code.drawio"))
