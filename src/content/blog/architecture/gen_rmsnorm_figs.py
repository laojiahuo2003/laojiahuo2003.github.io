# -*- coding: utf-8 -*-
"""
RMSNorm 博客配图 v2：内容驱动布局
核心改进：框的宽度由「文字实际渲染宽度」决定（用 matplotlib 字体度量估算），
不再固定框宽硬塞——解决「字挤在一起」的问题。
公式框用等宽估算 + 加 30% 余量。
"""
import os
import xml.sax.saxutils as sx

# ---------------- 引擎（从 llm-architectures/docs/gen_drawio.py 提取）----------------
_ENGINE = r"C:\Users\ljh\Desktop\code\llm-architectures\docs\gen_drawio.py"
_src = open(_ENGINE, encoding="utf-8").read()
_cut = _src.index("# ================================================================"
                  " Page 1")
_ns = {"__name__": "drawio_engine"}
exec(compile(_src[:_cut], "drawio_engine", "exec"), _ns)
Page, C, nid = _ns["Page"], _ns["C"], _ns["nid"]

# ---------------- 文字宽度测量（matplotlib 真实字体度量）----------------
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
    """文字在 fontsize(pt) 下的宽度，单位换算成 drawio 坐标(px @96dpi)。
    draw.io 默认 1pt 字号在画布上约 1.33px；一个中文字符宽约等于字号 pt。
    我们用 matplotlib 精确测量，再按 pt→px (×96/72) 转换。"""
    t = _fig.text(0, 0, s, fontproperties=_fp, fontsize=fontsize)
    bb = t.get_window_extent(renderer=_renderer)
    t.remove()
    return bb.width / _dpi * 96.0  # px


def fit_box(label: str, fs: float, min_w=120, pad=56, line_h=None):
    """按内容计算需要的 (w, h)。多行 label 用 \\n 分行，逐行取最大宽度。"""
    lines = label.split("\n")
    w = max(text_w(ln, fs) for ln in lines) + pad * 2
    h = len(lines) * (line_h or fs * 1.85) + 16
    return max(w, min_w), h


print("测量校准: 'RMSNorm' @12pt ≈", round(text_w("RMSNorm", 12)), "px (预期~60)")


def add_fit(page, x, y, label, fill, fs=12, bold=False, **kw):
    """内容自适应框：宽度=最长行+padding，高度=行数×行高"""
    w, h = fit_box(label, fs)
    h = kw.pop("h_override", h)  # 允许手动加高
    return page.add(x=x, y=y, w=w, h=h, label=label, fill=fill, fs=fs,
                    bold=bold, **kw), w, h


# ================================================================ 图 1：对比图 v2
p1 = Page("1 · LayerNorm vs RMSNorm", w=980, h=680)

p1.add(x=60, y=24, w=860, h=36, label="同一个向量，两种归一化",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

# 输入向量（居中放置）
vin, vw, vh = add_fit(p1, 330, 80, "x = [3.0,  1.0,  -2.0,  4.0]", C["input"],
                      fs=13, bold=True)

# ---- 左右两条流程（每行拆成「步骤」+「计算」两行，宽度自适应，避免挤在一行）----
steps_ln = [
    ("① 减均值 μ", "x − 1.50"),
    ("② 除标准差 σ", "÷ 2.29"),
    ("③ 仿射变换", "× γ + β"),
]
steps_rn = [
    ("① 求均方根 RMS", "RMS(x) = sqrt(mean(x²)) = 2.74"),
    ("② 除以 RMS", "x ÷ 2.74"),
    ("③ 缩放", "× γ"),
]

# 预计算两条流程的最大框宽（每框两行文字，取两行较长者），保证左右对称
def step_box_w(s):
    return max(fit_box(s[0], 11)[0], fit_box(s[1], 11)[0]) + 30

col_w = max(max(step_box_w(s) for s in steps_ln),
            max(step_box_w(s) for s in steps_rn), 210)

x_left, x_right = 60, 980 - 60 - col_w
y = 170
heads = []
for x0, name, color in [(x_left, "LayerNorm", C["norm"]),
                        (x_right, "RMSNorm", C["norm"])]:
    h = p1.add(x=x0, y=y, w=col_w, h=34, label=name, fill=color, fs=14, bold=True)
    heads.append(h)

y = 224
boxes_ln, boxes_rn = [], []
for (s1, calc1), (s2, calc2) in zip(steps_ln, steps_rn):
    b1 = p1.add(x=x_left, y=y, w=col_w, h=46,
                label=f"{s1}\n{calc1}", fill=C["white"], fs=11)
    b2 = p1.add(x=x_right, y=y, w=col_w, h=46,
                label=f"{s2}\n{calc2}", fill=C["white"], fs=11)
    boxes_ln.append(b1); boxes_rn.append(b2)
    if y == 224:
        p1.edge(heads[0], b1, exit_=(0.5, 1), entry=(0.5, 0))
        p1.edge(heads[1], b2, exit_=(0.5, 1), entry=(0.5, 0))
    else:
        p1.edge(boxes_ln[-2], b1, exit_=(0.5, 1), entry=(0.5, 0))
        p1.edge(boxes_rn[-2], b2, exit_=(0.5, 1), entry=(0.5, 0))
    y += 64

# 输出（两行，宽度与列对齐）
y_out = y + 6
out_ln = p1.add(x=x_left, y=y_out, w=col_w, h=52,
                label="μ=1.50  σ=2.29\n[ 0.65, -0.22, -1.53,  1.09 ]",
                fill="#FFF3E0", fs=11)
out_rn = p1.add(x=x_right, y=y_out, w=col_w, h=52,
                label="RMS=2.74\n[ 1.10,  0.37, -0.73,  1.46 ]",
                fill="#FFF3E0", fs=11)
p1.edge(boxes_ln[-1], out_ln, exit_=(0.5, 1), entry=(0.5, 0))
p1.edge(boxes_rn[-1], out_rn, exit_=(0.5, 1), entry=(0.5, 0))

# ---- 中间公式（按最长行自适应宽度；列变宽后中间空间可能不足，则压缩表述）----
mid_x = x_left + col_w + 24
mid_w = x_right - 24 - mid_x
mid_label = "LayerNorm\n(x−μ)/σ · γ + β\n\nRMSNorm\nx / RMS(x) · γ"
# 若中间区域放不下（含 padding），把公式框改为「竖排窄版」
if fit_box("x / RMS(x) · γ", 12)[0] + 40 > mid_w:
    mid_label = "LayerNorm\n(x−μ)/σ·γ+β\n\nRMSNorm\nx/RMS·γ"
p1.add(x=mid_x, y=224, w=mid_w, h=110, label=mid_label, fill=C["note"], fs=12)

# 输入到两条流程的箭头
p1.edge(vin, heads[0], exit_=(0.25, 1), entry=(0.5, 0))
p1.edge(vin, heads[1], exit_=(0.75, 1), entry=(0.5, 0))

# 底部总结（拆成三行短句，每行自适应测量，保证不超宽）
summary_lines = [
    "省掉：μ（一次归约）· β（一份参数）· σ²（二次归约）",
    "依据：归一化的价值在缩放不变性，中心化贡献很小",
    "收益：更快 · 参数少一半 · fp16 更稳",
]
sw = max(fit_box(s, 12)[0] for s in summary_lines)
sh = len(summary_lines) * 12 * 1.85 + 24
p1.add(x=(980 - sw) / 2, y=y_out + 100, w=sw, h=sh,
       label="\n".join(summary_lines), fill=C["panel"], fs=12)

# ================================================================ 导出函数（单文件）
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
           f'<mxCell id="1" parent="0"/>' + "".join(cells) +
           '</root></mxGraphModel></diagram></mxfile>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    print("已生成:", path)


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
write_page(p1, os.path.join(OUT_DIR, "rmsnorm_1_compare.drawio"))
