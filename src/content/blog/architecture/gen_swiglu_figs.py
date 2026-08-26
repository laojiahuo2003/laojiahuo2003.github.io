# -*- coding: utf-8 -*-
"""SwiGLU 博客配图：三张
  1. ReLU / GELU / SiLU 曲线关键点对比
  2. SwiGLU 门控结构（双路）
  3. 参数账：2 矩阵 vs 3 矩阵 + 2/3 规则（11008 的来历）
沿用 gen_gqa_figs.py 的引擎与套路。
"""
import os
import xml.sax.saxutils as sx

_ENGINE = r"C:\Users\ljh\Desktop\code\llm-architectures\docs\gen_drawio.py"
_src = open(_ENGINE, encoding="utf-8").read()
_cut = _src.index("# ================================================================"
                  " Page 1")
_ns = {"__name__": "drawio_engine"}
exec(compile(_src[:_cut], "drawio_engine", "exec"), _ns)
Page, C, nid = _ns["Page"], _ns["C"], _ns["nid"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams["font.family"] = ["Microsoft YaHei"]
_fp = FontProperties(family="Microsoft YaHei")
_fig = plt.figure(figsize=(10, 8))
_renderer = _fig.canvas.get_renderer()
_dpi = _fig.dpi


def text_w(s: str, fontsize: float) -> float:
    t = _fig.text(0, 0, s, fontproperties=_fp, fontsize=fontsize)
    bb = t.get_window_extent(renderer=_renderer)
    t.remove()
    return bb.width / _dpi * 96.0


def fit_box(label, fs, min_w=120, pad=28, line_h=None):
    lines = label.split("\n")
    w = max(text_w(ln, fs) for ln in lines) + pad * 2
    h = len(lines) * (line_h or fs * 1.85) + 14
    return max(w, min_w), h


STROKE = "#5A6B7F"


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
        _lbl = sx.escape(n.label).replace(chr(10), "&#xa;")
        cells.append(f'<mxCell id="{n.id}" value="{_lbl}" '
                     f'style="{n.style()}" vertex="1" parent="{n.parent}">'
                     f'{geom}</mxCell>')
    for e in page.edges:
        st = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
              f"strokeColor={e.color};strokeWidth={e.width}")
        if e.dashed:
            st += ";dashed=1"
        if not e.arrow:
            st += ";endArrow=none"
        if e.exit_:
            st += f";exitX={e.exit_[0]};exitY={e.exit_[1]};exitDx=0;exitDy=0"
        if e.entry:
            st += f";entryX={e.entry[0]};entryY={e.entry[1]};entryDx=0;entryDy=0"
        _elbl = sx.escape(e.label).replace(chr(10), "&#xa;") if e.label else ""
        lbl = f' value="{_elbl}"' if e.label else ""
        cells.append(f'<mxCell id="{nid("e")}"{lbl} style="{st}" edge="1" '
                     f'parent="1" source="{e.src}" target="{e.dst}">'
                     f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    xml = (f'<mxfile host="app.diagrams.net"><diagram id="{nid("d")}" '
           f'name="{page.name}"><mxGraphModel dx="900" dy="700" grid="1" '
           f'gridSize="10" page="1" pageScale="1" pageWidth="{page.w}" '
           f'pageHeight="{page.h}"><root><mxCell id="0"/>'
           f'<mxCell id="1" parent="0"/>'
           # 折线(网格/轴/曲线)先输出垫底,框和文字后画压在上面
           + "".join(getattr(page, "_extra_cells", [])) + "".join(cells) +
           '</root></mxGraphModel></diagram></mxfile>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    print("已生成:", path)


def polyline(page, pts, color, width=2.2, dashed=False):
    """自由折线:必须用 edge + sourcePoint/targetPoint + Array waypoints,
    style 里的 points=[...] 不会被 draw.io 渲染(踩过的坑)。"""
    style = (f"endArrow=none;html=1;rounded=0;strokeColor={color};"
             f"strokeWidth={width};")
    if dashed:
        style += "dashed=1;"
    way = "".join(f'<mxPoint x="{x}" y="{y}"/>'
                  for x, y in pts[1:-1])
    cell = (f'<mxCell id="{nid("pl")}" value="" style="{style}" '
            f'edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
            f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/>'
            f'<Array as="points">{way}</Array>'
            f'</mxGeometry></mxCell>')
    page._extra_cells = getattr(page, "_extra_cells", [])
    page._extra_cells.append(cell)


import math

p1 = Page("1 · ReLU GELU SiLU", w=1000, h=560)

p1.add(x=40, y=16, w=920, h=32,
       label="ReLU → GELU → SiLU：越来越圆，差异全在负半轴",
       fill=C["white"], fs=16, bold=True)._stroke = "#FFFFFF"

# ---- 绘图区：x∈[-4,2] → X [70,920]，y∈[-0.6,2.4] → Y [90,440] ----
PX0, PX1, PY0, PY1 = 70, 920, 90, 440
XMIN, XMAX, YMIN, YMAX = -4.0, 2.0, -0.6, 2.4


def to_px(xv, yv):
    px = PX0 + (xv - XMIN) / (XMAX - XMIN) * (PX1 - PX0)
    py = PY0 + (YMAX - yv) / (YMAX - YMIN) * (PY1 - PY0)
    return round(px, 1), round(py, 1)


def silu(x):
    return x / (1 + math.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


# ---- 网格（先画，垫底）----
for gx in (-3, -2, -1, 1, 2):                       # 竖网格（0 是轴）
    x_px = to_px(gx, 0)[0]
    polyline(p1, [(x_px, PY0), (x_px, PY1)], "#ECEFF1", width=1.0)
for gy in (2, 1.5, 1, 0.5, -0.5):                   # 横网格（0 是轴）
    y_px = to_px(0, gy)[1]
    polyline(p1, [(PX0, y_px), (PX1, y_px)], "#ECEFF1", width=1.0)

# ---- 坐标轴 ----
polyline(p1, [to_px(XMIN, 0), to_px(XMAX, 0)], "#78909C", width=1.4)
polyline(p1, [to_px(0, YMIN), to_px(0, YMAX)], "#78909C", width=1.4)

# ---- 刻度 ----
TK = dict(fill="none", fs=9)
for gx in (-3, -2, -1, 1, 2):
    x_px, _ = to_px(gx, 0)
    p1.add(x=x_px - 9, y=375, w=20, h=18, label=str(gx), **TK)
for gy, lbl in ((2, "2"), (1, "1"), (-0.5, "-0.5")):
    _, y_px = to_px(0, gy)
    p1.add(x=608, y=y_px - 9, w=26, h=18, label=lbl, **TK)
p1.add(x=620, y=375, w=14, h=18, label="0", **TK)
p1.add(x=928, y=432, w=14, h=16, label="x", **TK)
p1.add(x=610, y=70, w=24, h=16, label="f(x)", **TK)

# ---- 三条曲线（61 点采样）----
xs = [(XMIN + i * (XMAX - XMIN) / 60) for i in range(61)]
polyline(p1, [to_px(v, silu(v)) for v in xs], "#1E88E5", width=3.0)
polyline(p1, [to_px(v, gelu(v)) for v in xs], "#7E57C2", width=3.0)
polyline(p1, [to_px(v, max(0.0, v)) for v in xs], "#E53935", width=3.0)

# ---- 左上标签堆（ReLU/GELU/SiLU 各一，配引线指向曲线特征点）----
LBL = [
    ("ReLU · max(0, x)", "负区一刀切 0，神经元死了不复活", "#E53935",
     to_px(-2.8, 0)),                                   # → 平坦死区段
    ("GELU · x·Φ(x)", "平滑过渡，浅坑（BERT / GPT-2）", "#7E57C2",
     to_px(-0.75, gelu(-0.75))),                        # → 浅坑
    ("SiLU · x·σ(x)", "坑更深（最低 -0.278），处处光滑\nLlama config 里的 silu",
     "#1E88E5", to_px(-1.28, -0.278)),                  # → 最低点
]
ly = 118
for title, desc, colr, target in LBL:
    n_lines = desc.count("\n") + 1
    h = 20 + n_lines * 17
    p1.add(x=96, y=ly, w=300, h=h,
           label=f"{title}\n{desc}", fill=C["white"], fs=11)._stroke = colr
    ly += h + 18

# ---- SiLU 最低点标记 ----
mx, my = to_px(-1.28, -0.278)
p1.add(x=mx - 4, y=my - 4, w=8, h=8, label="", fill="#1E88E5",
       shape="ellipse")._stroke = "#1E88E5"
p1.add(x=mx + 10, y=my + 2, w=54, h=14, label="-0.278", fill="none",
       fs=9)

# ---- 死区标注（负半轴下方空白处）----
p1.add(x=140, y=424, w=150, h=16, label="死区：x<0 恒为 0", fill="none",
       fs=9)

# ---- 右上：重合说明 ----
p1.add(x=690, y=112, w=190, h=36, label="正半轴三线重合\n都趋近 y = x",
       fill=C["white"], fs=10)._stroke = "#FFFFFF"

# ---- 底部收尾横幅 ----
p1.add(x=70, y=488, w=850, h=38,
       label="共同点：都是「一条路」——输入 x 直接过函数。真正的变化不在函数曲线，在结构：装门（见下图）",
       fill=C["panel"], fs=11)

# ================================================================
p2 = Page("2 · SwiGLU 双路结构", w=1000, h=560)

p2.add(x=40, y=18, w=920, h=34, label="SwiGLU：一路做菜，一路决定上多少",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

xin = p2.add(x=60, y=250, w=110, h=52, label="x\n(B, S, 4096)", fill=C["input"],
             fs=11, bold=True)
gate = p2.add(x=280, y=130, w=190, h=52,
              label="W_gate（门路）\n4096 → 11008", fill=C["ffn"], fs=11)
up = p2.add(x=280, y=350, w=190, h=52,
            label="W_up（内容路）\n4096 → 11008", fill=C["ffn"], fs=11)
silu_n = p2.add(x=530, y=130, w=150, h=52, label="SiLU\n(连续阀门 0~1)",
                fill=C["attn"], fs=11)
mul = p2.add(x=560, y=246, w=130, h=52, label="⊙ 逐元素相乘", fill=C["res"],
             fs=11, bold=True)
down = p2.add(x=760, y=246, w=180, h=52, label="W_down\n11008 → 4096",
              fill=C["ffn"], fs=11)
out = p2.add(x=760, y=352, w=180, h=44, label="输出 (B, S, 4096)",
             fill=C["out"], fs=11)

p2.edge(xin, gate, exit_=(0.5, 1), entry=(0.5, 0), label="")
p2.edge(xin, up, exit_=(0.5, 0), entry=(0.5, 1))
p2.edge(gate, silu_n, exit_=(1, 0.5), entry=(0, 0.5))
p2.edge(silu_n, mul, exit_=(0.5, 1), entry=(0.5, 0))
p2.edge(up, mul, exit_=(1, 0.5), entry=(0, 0.5))
p2.edge(mul, down, exit_=(1, 0.5), entry=(0, 0.5))
p2.edge(down, out, exit_=(0.5, 1), entry=(0.5, 0), label="")

# 老结构对照（底部）
old = p2.add(x=60, y=470, w=880, h=56,
             label="老 FFN 对照：x → W₁(4096→16384) → ReLU → W₂(16384→4096)　　一条路，没有门",
             fill=C["gray"], fs=11)

gate_note = "门开多大跟内容走：同一个维度，这个词开 0.9，那个词开 0.1"
gw, gh = fit_box(gate_note, 11)
p2.add(x=280, y=70, w=gw, h=gh, label=gate_note, fill=C["white"],
       fs=11)._stroke = "#FFFFFF"

# ================================================================
# 图 3：参数账（2 矩阵 vs 3 矩阵 + 2/3 规则）
# ================================================================
p3 = Page("3 · 参数账", w=1000, h=520)

p3.add(x=40, y=18, w=920, h=34, label="多一个矩阵，宽度缩到 2/3 找平",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

# 左：老结构参数条（两个矩阵 16384 高）
bar_x = 90
p3.add(x=bar_x, y=110, w=90, h=200, label="W₁\n4096\n×16384", fill=C["ffn"],
       fs=11)
p3.add(x=bar_x + 110, y=110, w=90, h=200, label="W₂\n16384\n×4096", fill=C["ffn"],
       fs=11)
p3.add(x=bar_x - 10, y=80, w=230, h=24, label="老 FFN：2 × 4096 × 16384 = 134.2 M",
       fill=C["white"], fs=11)._stroke = "#FFFFFF"

# 右：SwiGLU（三个矩阵 11008 高 ≈ 200*2/3 ≈ 133）
h2 = 200 * 2 / 3
bx2 = 500
p3.add(x=bx2, y=110 + (200 - h2), w=90, h=h2, label="W_gate\n4096\n×11008",
       fill=C["attn"], fs=10)
p3.add(x=bx2 + 110, y=110 + (200 - h2), w=90, h=h2, label="W_up\n4096\n×11008",
       fill=C["attn"], fs=10)
p3.add(x=bx2 + 220, y=110 + (200 - h2), w=90, h=h2, label="W_down\n11008\n×4096",
       fill=C["attn"], fs=10)
p3.add(x=bx2 - 10, y=80, w=330, h=24,
       label="SwiGLU：3 × 4096 × 11008 = 135.3 M（+0.8%）",
       fill=C["white"], fs=11)._stroke = "#FFFFFF"

# 底部：2/3 规则推导
rule = ("找平公式：3 × d × (2/3 · d_ff) = 2 × d × d_ff　→　中间宽度 = 原宽度 × 2/3\n"
        "Llama2-7B 实算：4096 × 4 = 16384 → × 2/3 = 10922.7 → 向上取整到 256 倍数 = 11008")
rw, rh = fit_box(rule, 12)
p3.add(x=(1000 - rw) / 2, y=370, w=rw, h=rh, label=rule, fill=C["note"], fs=12)

pnote = "config 里那行 intermediate_size: 11008 —— 不是玄学，是一道四则运算"
pw2, ph2 = fit_box(pnote, 11)
p3.add(x=(1000 - pw2) / 2, y=448, w=pw2, h=ph2, label=pnote, fill=C["panel"],
       fs=11)

# ================================================================
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

write_page(p1, os.path.join(OUT_DIR, "swiglu_1_activations.drawio"))
write_page(p2, os.path.join(OUT_DIR, "swiglu_2_gate.drawio"))
write_page(p3, os.path.join(OUT_DIR, "swiglu_3_params.drawio"))
print("（图 1 含折线 cell，需目检曲线渲染）")
