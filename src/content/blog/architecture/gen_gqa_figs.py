# -*- coding: utf-8 -*-
"""
GQA 博客配图：三张图
  1. MHA vs MQA vs GQA 演进对比（8 Q 头缩样示意）
  2. KV cache 算账（真实规格 32 层 · head_dim 128 · fp16）
  3. 实现流程（q/k/v 通路 + repeat_kv 形状流转）
沿用 rmsnorm/rope 的生成套路：引擎 + 文字宽度测量 + 布局检查。
"""
import os
import xml.sax.saxutils as sx

# ---------------- 引擎（同 gen_rmsnorm_figs.py）----------------
_ENGINE = r"C:\Users\ljh\Desktop\code\llm-architectures\docs\gen_drawio.py"
_src = open(_ENGINE, encoding="utf-8").read()
_cut = _src.index("# ================================================================"
                  " Page 1")
_ns = {"__name__": "drawio_engine"}
exec(compile(_src[:_cut], "drawio_engine", "exec"), _ns)
Page, C, nid = _ns["Page"], _ns["C"], _ns["nid"]

# ---------------- 文字宽度测量 ----------------
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


def fit_box(label: str, fs: float, min_w=120, pad=28, line_h=None):
    lines = label.split("\n")
    w = max(text_w(ln, fs) for ln in lines) + pad * 2
    h = len(lines) * (line_h or fs * 1.85) + 14
    return max(w, min_w), h


print("测量校准: 'GQA' @12pt ≈", round(text_w("GQA", 12)), "px (预期~38)")

STROKE = "#5A6B7F"
INK = "#1A2332"


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
        st = f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor={e.color};strokeWidth={e.width}"
        if getattr(e, "straight", False):
            pass  # 直线：不加 orthogonalEdgeStyle
        else:
            st = "edgeStyle=orthogonalEdgeStyle;" + st
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
           f'<mxCell id="1" parent="0"/>' + "".join(cells) +
           "".join(getattr(page, "_straight_cells", [])) +
           '</root></mxGraphModel></diagram></mxfile>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    print("已生成:", path)


def straight(page, src, dst, color=STROKE, width=1.4, label=""):
    """直线边（共享连线用直线更清晰）"""
    st = f"rounded=1;html=1;strokeColor={color};strokeWidth={width}"
    if label:
        st += f';fontSize=10;labelBackgroundColor=#FFFFFF'
    _elbl = sx.escape(label).replace(chr(10), "&#xa;") if label else ""
    lbl = f' value="{_elbl}"' if label else ""
    cells = (f'<mxCell id="{nid("e")}"{lbl} style="{st}" edge="1" '
             f'parent="1" source="{src.id}" target="{dst.id}">'
             f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    page._straight_cells = getattr(page, "_straight_cells", [])
    page._straight_cells.append(cells)
    return None


OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# 图 1：MHA vs MQA vs GQA（8 Q 头缩样示意）
# ================================================================
p1 = Page("1 · MHA vs MQA vs GQA", w=1200, h=470)

p1.add(x=40, y=18, w=1120, h=34, label="三种注意力，变的只是 KV 头的数量",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

# ---- 面板参数 ----
PANEL_W = 356
QB_W, QB_H, QB_GAP = 34, 30, 6          # Q 头小方块
NQ = 8
QROW_W = NQ * QB_W + (NQ - 1) * QB_GAP  # 314
panel_xs = [40, 40 + PANEL_W + 26, 40 + 2 * (PANEL_W + 26)]
Y_PANEL = 66
q_y = Y_PANEL + 96 + 14
kv_y = q_y + QB_H + 78
info_y = kv_y + 34 + 22

C_MHA = C["attn"]      # 绿
C_MQA = C["res"]       # 红
C_G1, C_G2 = C["attn"], C["ffn"]  # GQA 两组：绿 / 紫


def q_row(px):
    """放 8 个 Q 头方块，返回节点列表"""
    qs = []
    x0 = px + (PANEL_W - QROW_W) / 2
    for i in range(NQ):
        qs.append(p1.add(x=x0 + i * (QB_W + QB_GAP), y=q_y, w=QB_W, h=QB_H,
                         label=f"q{i}", fill=C["input"], fs=10))
    return qs


panels = [
    ("MHA\nMulti-Head\n每 Q 头一份 KV", C_MHA, 8),
    ("MQA\nMulti-Query\n全体共享一份 KV", C_MQA, 1),
    ("GQA\nGrouped-Query\n每组一份 KV", C_G1, 2),
]
infos = [
    "KV 头：8（= Q 头数）\nKV cache：基准",
    "KV 头：1（全部共享）\nKV cache：1/8",
    "KV 头：2（每组 4 个 Q）\nKV cache：1/4",
]

for (title, colr, n_kv), px, info in zip(panels, panel_xs, infos):
    p1.add(x=px, y=Y_PANEL, w=PANEL_W, h=60, label=title, fill=colr,
           fs=13, bold=True)
    p1.add(x=px + 10, y=q_y - 26, w=QROW_W, h=20,
           label="Q 头 × 8（三种都一样）", fill=C["white"], fs=10
           )._stroke = "#FFFFFF"
    qs = q_row(px)
    x0 = px + (PANEL_W - QROW_W) / 2
    if n_kv == 8:                       # MHA：一一对应
        for i, q in enumerate(qs):
            kv = p1.add(x=x0 + i * (QB_W + QB_GAP), y=kv_y, w=QB_W, h=34,
                        label="K·V", fill=colr, fs=10)
            straight(p1, q, kv, color=STROKE, width=1.2)
    elif n_kv == 1:                     # MQA：8 线汇聚
        kv = p1.add(x=px + PANEL_W / 2 - 32, y=kv_y, w=64, h=34,
                    label="K·V", fill=colr, fs=11, bold=True)
        for q in qs:
            straight(p1, q, kv, color=STROKE, width=1.2)
    else:                               # GQA：每组 4 Q → 一份 KV
        grp = [qs[:4], qs[4:]]
        for gi, g in enumerate(grp):
            gc = C_G1 if gi == 0 else C_G2
            cx = (g[0].x + g[-1].x + QB_W) / 2
            kv = p1.add(x=cx - 26, y=kv_y, w=52, h=34, label="K·V",
                        fill=gc, fs=10)
            for q in g:
                straight(p1, q, kv, color=STROKE, width=1.2)
    iw, ih = fit_box(info, 11)
    p1.add(x=px + (PANEL_W - iw) / 2, y=info_y, w=iw, h=ih, label=info,
           fill=C["note"], fs=11)

sum_label = ("KV cache 与 KV 头数成正比：Q 头一个不少，KV 头 8 → 2 → 1 份\n"
             "省的是显存与带宽，伤的是每个 Q 头「专用」K/V 的表达力——GQA 停在折中点")
sw, sh = fit_box(sum_label, 12)
p1.add(x=(1200 - sw) / 2, y=info_y + 66, w=sw, h=sh, label=sum_label,
       fill=C["panel"], fs=12)

# ================================================================
# 图 2：KV cache 算账（32 层 · head_dim=128 · fp16）
# ================================================================
p2 = Page("2 · KV cache 算账", w=980, h=470)

p2.add(x=40, y=18, w=900, h=34, label="省下的 KV cache，是每个 token 都要背的显存",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

f1 = "每 token KV cache = 2 (K 和 V) × n_layers × n_kv_heads × d_head × 2 B (fp16)\n" \
     "= 2 × 32 × n_kv_heads × 128 × 2 B = 16 KB × n_kv_heads"
fw, fh = fit_box(f1, 12)
p2.add(x=(980 - fw) / 2, y=64, w=fw, h=fh, label=f1, fill=C["note"], fs=12)

rows = [
    ("MHA\nn_kv=32", C["attn"], 512, "512 KB / token　→　4096 上下文单序列 2.0 GB"),
    ("GQA\nn_kv=8",  C["ffn"],   128, "128 KB / token　→　4096 上下文单序列 512 MB"),
    ("MQA\nn_kv=1",  C["res"],    16, "16 KB / token　→　4096 上下文单序列 64 MB"),
]
BAR_X, BAR_MAXW, BAR_H = 250, 300, 34
by = 160
for name, colr, kb, val in rows:
    lw, lh = fit_box(name, 11, pad=16)
    p2.add(x=60, y=by + (BAR_H - lh) / 2, w=110, h=lh, label=name,
           fill=colr, fs=11, bold=True)
    bw = max(BAR_MAXW * kb / 512, 14)
    p2.add(x=BAR_X, y=by, w=bw, h=BAR_H, label="", fill=colr)
    vw, vh = fit_box(val, 11)
    p2.add(x=BAR_X + BAR_MAXW + 24, y=by + (BAR_H - vh) / 2, w=vw, h=vh,
           label=val, fill=C["white"], fs=11)._stroke = "#FFFFFF"
    by += BAR_H + 20

note = ("decode 阶段每生成一个 token，都要把整个 KV cache 从显存搬到 SM 一遍：\n"
        "KV cache 减到 1/r，显存占用和带宽压力一起减到 1/r——解码提速的来源\n"
        "batch = 8 · 4096 上下文：16 GB → 4 GB → 512 MB（A100-40G 单卡装得下与否的分界）")
nw, nh = fit_box(note, 11)
p2.add(x=(980 - nw) / 2, y=by + 8, w=nw, h=nh, label=note,
       fill=C["panel"], fs=11)

# ================================================================
# 图 3：实现流程（q/k 通路 + repeat_kv）
# ================================================================
p3 = Page("3 · 实现：算之前把 KV 头补回来", w=1140, h=560)

p3.add(x=40, y=18, w=1060, h=34, label="实现：KV 头少没关系，算之前补回来",
       fill=C["white"], fs=15, bold=True)._stroke = "#FFFFFF"

Y_Q, Y_KV = 84, 250
x_in = 60
bw1 = 118

xin = p3.add(x=x_in, y=Y_Q + 26, w=bw1, h=52, label="x\n(B, S, 4096)",
             fill=C["input"], fs=11, bold=True)
qproj = p3.add(x=x_in + 170, y=Y_Q, w=bw1 + 30, h=52,
               label="W_q · RoPE\n4096 × 4096", fill=C["ffn"], fs=11)
qout = p3.add(x=x_in + 360, y=Y_Q, w=bw1 + 30, h=52,
              label="q: view\n(B, S, 32, 128)", fill=C["attn"], fs=11)
p3.edge(xin, qproj, exit_=(1, 0.5), entry=(0, 0.5))
p3.edge(qproj, qout, exit_=(1, 0.5), entry=(0, 0.5))

xink = p3.add(x=x_in, y=Y_KV + 26, w=bw1, h=52, label="x（同一个）",
              fill=C["input"], fs=11, bold=True)
kproj = p3.add(x=x_in + 170, y=Y_KV, w=bw1 + 30, h=52,
               label="W_k · RoPE\n4096 × 1024", fill=C["ffn"], fs=11)
kout = p3.add(x=x_in + 360, y=Y_KV, w=bw1 + 30, h=52,
              label="k: view\n(B, S, 8, 128)", fill=C["attn"], fs=11)
p3.edge(xink, kproj, exit_=(1, 0.5), entry=(0, 0.5))
p3.edge(kproj, kout, exit_=(1, 0.5), entry=(0, 0.5))
vnote = "v 通路同 k（W_v · 无 RoPE）"
vw2, vh2 = fit_box(vnote, 10)
p3.add(x=x_in, y=Y_KV + 92, w=vw2, h=vh2, label=vnote, fill=C["white"],
       fs=10)._stroke = "#FFFFFF"

rk = p3.add(x=x_in + 360, y=Y_KV + 108, w=bw1 + 90, h=52,
            label="repeat_kv(k, 4)\nexpand + reshape（无拷贝→一次拼接）",
            fill="#FFE0B2", fs=11, bold=True)
p3.edge(kout, rk, exit_=(0.5, 1), entry=(0.5, 0))
kbig = p3.add(x=x_in + 640, y=Y_KV, w=bw1 + 50, h=52,
              label="(B, S, 32, 128)\n8 份 KV 各自复制 4 份",
              fill=C["attn"], fs=11)
p3.edge(rk, kbig, exit_=(1, 0.5), entry=(0, 0.5))

sdpa = p3.add(x=x_in + 640, y=Y_Q + 6, w=bw1 + 50, h=56,
              label="attention\nq · kᵀ / √d → softmax → · v",
              fill=C["out"], fs=11, bold=True)
p3.edge(qout, sdpa, exit_=(1, 0.5), entry=(0, 0.5))
p3.edge(kbig, sdpa, exit_=(0.5, 0), entry=(0.5, 1))
outp = p3.add(x=x_in + 880, y=Y_Q + 6, w=150, h=56,
              label="merge → W_o\n(B, S, 4096)", fill=C["ffn"], fs=11)
p3.edge(sdpa, outp, exit_=(1, 0.5), entry=(0, 0.5))

# repeat_kv 三步展开（拆两行，避免超宽）
t3 = ("repeat_kv 展开：k (B, S, 8, 128) —unsqueeze(2)→ (B, S, 8, 1, 128)\n"
      "—expand(n_rep=4)→ (B, S, 8, 4, 128)（视图，0 拷贝）—reshape→ (B, S, 32, 128)（一次拷贝，4 份连续）")
tw, th = fit_box(t3, 11)
p3.add(x=(1140 - tw) / 2, y=448, w=tw, h=th, label=t3, fill=C["note"], fs=11)

pnote = "参数量：W_q / W_o 各 4096×4096 = 16.8 M；W_k / W_v 各 4096×1024 = 4.2 M（MHA 时是 16.8 M，也一并省下 3/4）"
pw2, ph2 = fit_box(pnote, 11)
p3.add(x=(1140 - pw2) / 2, y=506, w=pw2, h=ph2, label=pnote,
       fill=C["panel"], fs=11)

# ================================================================
write_page(p1, os.path.join(OUT_DIR, "gqa_1_evolution.drawio"))
write_page(p2, os.path.join(OUT_DIR, "gqa_2_kvcache.drawio"))
write_page(p3, os.path.join(OUT_DIR, "gqa_3_code.drawio"))
