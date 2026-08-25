# -*- coding: utf-8 -*-
"""
draw.io 图排版验证器：用 draw.io 导出的 SVG 中的 image fallback 尺寸
（= 文字真实渲染尺寸，含自动换行）判断每个框内文字是否溢出。
用法: verify_drawio.py <file.drawio>
"""
import re
import subprocess
import sys
import tempfile
import os

DRAWIO = r"D:\draw.io\draw.io.exe"


def verify(drawio_path: str) -> int:
    svg_path = os.path.join(tempfile.gettempdir(),
                            "verify_" + os.path.basename(drawio_path) + ".svg")
    r = subprocess.run([DRAWIO, "-x", "-f", "svg", "-o", svg_path, drawio_path],
                       capture_output=True, timeout=120)
    svg = open(svg_path, encoding="utf-8").read()
    chunks = re.split(r'(?=<g data-cell-id="n\d+">)', svg)
    bad = 0
    for ch in chunks:
        m = re.match(r'<g data-cell-id="(n\d+)">', ch)
        if not m:
            continue
        rm = re.search(r'<rect x="([\d.-]+)" y="([\d.-]+)" width="([\d.]+)" '
                       r'height="([\d.]+)"', ch)
        tm = re.search(r'<g transform="translate\(([\d.-]+),\s*([\d.-]+)\)"', ch)
        im = re.search(r'<image x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" '
                       r'height="([\d.]+)"', ch)
        if not (rm and tm and im):
            continue
        tx, ty = float(tm.group(1)), float(tm.group(2))
        rx, ry, rw, rh = (float(rm.group(k)) for k in range(1, 5))
        ix, iy, iw, ih = (float(im.group(k)) for k in range(1, 5))
        box_l, box_t = tx + rx, ty + ry
        over_r = (ix + iw) - (box_l + rw)
        over_b = (iy + ih) - (box_t + rh)
        over_l = box_l - ix
        over_t = box_t - iy
        if over_r > 1 or over_b > 1 or over_l > 1 or over_t > 1:
            bad += 1
            print(f"⚠️ {m.group(1)} 文字溢出框: 框{rw:.0f}x{rh:.0f} "
                  f"文字{iw:.0f}x{ih:.0f} "
                  f"(超右{max(0,over_r):.0f} 超下{max(0,over_b):.0f} "
                  f"超左{max(0,over_l):.0f} 超上{max(0,over_t):.0f})")
    os.remove(svg_path)
    if bad == 0:
        print(f"[PASS] {drawio_path} 全部文字在框内")
    else:
        print(f"[FAIL] {drawio_path} 有 {bad} 个框文字溢出")
    return bad


if __name__ == "__main__":
    sys.exit(1 if verify(sys.argv[1]) else 0)
