# -*- coding: utf-8 -*-
"""
生成 RMSNorm 博客配图（draw.io XML → 本地 CLI 导出 PNG）
3 页：① LayerNorm vs RMSNorm 对比 ② RMSNorm 前向计算流 ③ 代码-公式对照
"""
import os
import sys

# 引擎（Page/Node/Edge + 布局检查）在 llm-architectures/docs/gen_drawio.py 里。
# 该文件顶层会执行全部页面定义，所以只 exec 它的「引擎定义区」（Page 1 之前），
# 拿到 Page / C / nid 三个符号，无副作用。
_ENGINE = r"C:\Users\ljh\Desktop\code\llm-architectures\docs\gen_drawio.py"
_src = open(_ENGINE, encoding="utf-8").read()
_cut = _src.index("# ================================================================"
                  " Page 1")
_ns = {"__name__": "drawio_engine"}
exec(compile(_src[:_cut], "drawio_engine", "exec"), _ns)
Page, C, nid = _ns["Page"], _ns["C"], _ns["nid"]

STROKE = "#5A6B7F"
INK = "#1A2332"

# ================================================================ Page 1: 对比图
p1 = Page("1 · LayerNorm vs RMSNorm", w=860, h=620)

p1.add(x=60, y=30, w=740, h=40, label="同一个向量，两种归一化",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

# 输入向量示例
vin = p1.add(x=80, y=100, w=280, h=56, label="x = [3.0, 1.0, -2.0, 4.0]",
             fill=C["input"], fs=13, bold=True)

# LayerNorm 分支（左）
p1.add(x=60, y=190, w=160, h=32, label="LayerNorm", fill=C["norm"], fs=13, bold=True)
ln1 = p1.add(x=60, y=232, w=160, h=34, label="① 减均值 μ", fill=C["white"], fs=11)
ln2 = p1.add(x=60, y=278, w=160, h=34, label="② 除标准差 σ", fill=C["white"], fs=11)
ln3 = p1.add(x=60, y=324, w=160, h=34, label="③ × γ + β", fill=C["white"], fs=11)
for a, b in [(ln1, ln2), (ln2, ln3)]:
    p1.edge(a, b, exit_=(0.5, 1), entry=(0.5, 0))
p1.edge(vin, ln1, exit_=(0.25, 1), entry=(0.5, 0))

ln_out = p1.add(x=60, y=396, w=160, h=60,
                label="μ=1.5  σ≈2.29\n→ [-0.22, -0.66, -1.53, 0.66]",
                fill="#FFF3E0", fs=10)
p1.edge(ln3, ln_out, exit_=(0.5, 1), entry=(0.5, 0))

# RMSNorm 分支（右）
p1.add(x=560, y=190, w=160, h=32, label="RMSNorm", fill=C["norm"], fs=13, bold=True)
rn1 = p1.add(x=560, y=232, w=160, h=34, label="① 求均方根 RMS", fill=C["white"], fs=11)
rn2 = p1.add(x=560, y=278, w=160, h=34, label="② x / RMS", fill=C["white"], fs=11)
rn3 = p1.add(x=560, y=324, w=160, h=34, label="③ × γ", fill=C["white"], fs=11)
for a, b in [(rn1, rn2), (rn2, rn3)]:
    p1.edge(a, b, exit_=(0.5, 1), entry=(0.5, 0))
p1.edge(vin, rn1, exit_=(0.75, 1), entry=(0.5, 0))

rn_out = p1.add(x=560, y=396, w=160, h=60,
                label="RMS≈2.74\n→ [1.10, 0.37, -0.73, 1.46]",
                fill="#FFF3E0", fs=10)
p1.edge(rn3, rn_out, exit_=(0.5, 1), entry=(0.5, 0))

# 中间公式对比（两侧各留 40px 给步骤框：框宽160 起 x=60/x=560，中间可用 240~540）
p1.add(x=250, y=232, w=280, h=96,
       label="LayerNorm: (x−μ)/σ · γ + β\n\nRMSNorm:  x / RMS(x) · γ\n\n"
             "RMS(x) = √( (1/d)·Σxᵢ² )",
       fill=C["note"], fs=12)

p1.add(x=60, y=490, w=740, h=90,
       label="省掉的东西：μ（一次求和归约）、β（一份参数+一次加法）\n"
             "省掉的理由：归一化真正起作用的是「缩放不变性」，中心化贡献很小\n"
             "收益：少一次完整的数据依赖归约 → 训练/推理都更快；参数少一半",
       fill=C["panel"], fs=12)

# ================================================================ Page 2: 前向计算流
p2 = Page("2 · RMSNorm 前向与反向", w=860, h=640)

p2.add(x=60, y=30, w=740, h=40, label="RMSNorm 前向数据流（张量形状标注）",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

flow = [
    ("x  (B, S, d)", C["input"], 34),
    ("x.pow(2)  → x²  (B, S, d)", C["attn"], 34),
    (".mean(dim=-1, keepdim=True)  →  (B, S, 1)", C["attn"], 34),
    ("+ eps 后 rsqrt  →  1/√(mean(x²)+ε)  (B, S, 1)", C["norm"], 40),
    ("x · 1/√(...)  广播相乘  →  (B, S, d)", C["attn"], 34),
    ("× γ (逐维缩放)  →  (B, S, d)", C["norm"], 34),
]
nodes = []
y = 90
for label, fill, h in flow:
    n = p2.add(x=200, y=y, w=460, h=h, label=label, fill=fill, fs=12)
    if nodes:
        p2.edge(nodes[-1], n, exit_=(0.5, 1), entry=(0.5, 0))
    nodes.append(n)
    y += h + 24

p2.add(x=60, y=100, w=120, h=300, label="γ 可学习\n初始化为全 1",
       fill=C["gray"], fs=11)
p2.edge(p2.nodes[-1], nodes[5], exit_=(0.5, 1), entry=(0, 0.5), dashed=True,
        color="#8E24AA")

p2.add(x=60, y=440, w=740, h=160,
       label="反向传播要点（autograd 自动做，但值得知道）：\n"
             "∂L/∂xᵢ = γᵢ/√(ms+ε) · (∂L/∂yᵢ − xᵢ·Σⱼ(∂L/∂yⱼ·γⱼ·xⱼ)/(d·(ms+ε)))\n"
             "（ms = mean(x²)）——梯度不只走「直接路径」，还有一项通过 ms 回头\n\n"
             "数值细节：eps 在 rsqrt 里防除零；fp16 训练时归一化通常保持 fp32 计算",
       fill=C["panel"], fs=11)

# ================================================================ Page 3: 代码对照
p3 = Page("3 · 公式 ↔ 代码对照", w=860, h=560)

p3.add(x=60, y=30, w=740, h=40, label="一行公式，两行代码",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

p3.add(x=60, y=100, w=740, h=64,
       label="公式：  yᵢ = γᵢ · xᵢ / √( (1/d)·Σⱼ xⱼ² + ε )",
       fill=C["note"], fs=14, bold=True)

code = """class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ

    def forward(self, x):                            # x: (B,S,d)
        x_norm = x * torch.rsqrt(                    # ② 除以 √(ms+ε)
            x.pow(2).mean(-1, keepdim=True) + self.eps)  # ① ms=mean(x²)
        return x_norm * self.weight                  # ③ × γ"""

p3.add(x=60, y=190, w=740, h=190, label=code, fill="#F5F7FA", fs=12)
# 代码框右边注释箭头
p3.add(x=60, y=400, w=230, h=110,
       label="Llama 系列全部把归一化\n放在子层【前】(pre-norm)\n\n"
             "x = x + Attn(Norm(x))\nx = x + FFN(Norm(x))",
       fill=C["attn"], fs=11)
p3.add(x=320, y=400, w=230, h=110,
       label="真实模型超参：\n\nLlama-2-7B: d=4096, ε=1e-5\n"
             "Qwen3:      d=4096, ε=1e-6\nDeepSeek-V3: d=7168, ε=1e-6",
       fill=C["gray"], fs=11)
p3.add(x=580, y=400, w=220, h=110,
       label="为什么敢去掉 β？\n\nβ 的一份加法自由度\n在大规模下收益≈0，\n还干扰量化",
       fill=C["panel"], fs=11)

# ================================================================ 输出（单页一个文件）
import xml.sax.saxutils as sx

def write_page(page, path):
    problems = page.check()
    if problems:
        print(f"[{page.name}] 布局问题:")
        for p in problems:
            print("  ", p)
    else:
        print(f"[{page.name}] 布局检查通过 ✓")
    cells = []
    for n in page.nodes:
        geom = (f'<mxGeometry x="{n.x}" y="{n.y}" width="{n.w}" '
                f'height="{n.h}" as="geometry"/>')
        cells.append(f'<mxCell id="{n.id}" value="{sx.escape(n.label)}" '
                     f'style="{n.style()}" vertex="1" parent="{n.parent}">'
                     f'{geom}</mxCell>')
    for e in page.edges:
        st = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
              f"strokeColor={e.color};strokeWidth={e.width}")
        if e.dashed:
            st += ";dashed=1"
        if e.exit_:
            st += f";exitX={e.exit_[0]};exitY={e.exit_[1]};exitDx=0;exitDy=0"
        if e.entry:
            st += f";entryX={e.entry[0]};entryY={e.entry[1]};entryDx=0;entryDy=0"
        lbl = f' value="{sx.escape(e.label)}"' if e.label else ""
        cells.append(f'<mxCell id="{nid("e")}"{lbl} style="{st}" edge="1" '
                     f'parent="1" source="{e.src}" target="{e.dst}">'
                     f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    xml = (f'<mxfile host="app.diagrams.net"><diagram id="{nid("d")}" '
           f'name="{page.name}"><mxGraphModel dx="900" dy="700" grid="1" '
           f'gridSize="10" page="1" pageScale="1" pageWidth="{page.w}" '
           f'pageHeight="{page.h}"><root><mxCell id="0"/>'
           f'<mxCell id="1" parent="0"/>' + "".join(cells) +
           '</root></mxGraphModel></diagram></mxfile>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    print("已生成:", path)


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
write_page(p1, os.path.join(OUT_DIR, "rmsnorm_1_compare.drawio"))
write_page(p2, os.path.join(OUT_DIR, "rmsnorm_2_forward.drawio"))
write_page(p3, os.path.join(OUT_DIR, "rmsnorm_3_code.drawio"))
