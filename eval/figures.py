"""Рисунки для рукописи (PIL, без внешних зависимостей).

Fig. 1 — схема конвейера; Fig. 2 — качество поиска по ретриверам и индексам
с 95% доверительными интервалами.

Запуск: python eval/figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DPI = 300
W, H = 1500, 1000
BG = "white"
INK = "#1a1a1a"
MUTED = "#5a5a5a"
ACCENT = "#2f5d8a"
ACCENT2 = "#8aa6c1"
GREY = "#e8ecef"

FONT_DIR = Path("C:/Windows/Fonts")
REG = FONT_DIR / "arial.ttf"
BOLD = FONT_DIR / "arialbd.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = BOLD if bold else REG
    return ImageFont.truetype(str(path), size)


def box(draw, xy, text, fill=GREY, outline=ACCENT, radius=14, size=26, bold=False):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=3)
    f = font(size, bold)
    x0, y0, x1, y1 = xy
    lines = text.split("\n")
    total = sum(draw.textbbox((0, 0), line, font=f)[3] for line in lines)
    y = y0 + (y1 - y0 - total) / 2
    for line in lines:
        w = draw.textbbox((0, 0), line, font=f)[2]
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=f, fill=INK)
        y += draw.textbbox((0, 0), line, font=f)[3] + 4


def arrow(draw, start, end, fill=MUTED, width=3):
    draw.line([start, end], fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    import math

    ang = math.atan2(y1 - y0, x1 - x0)
    size = 14
    for sign in (1, -1):
        draw.line(
            [
                (x1, y1),
                (
                    x1 - size * math.cos(ang - sign * math.pi / 7),
                    y1 - size * math.sin(ang - sign * math.pi / 7),
                ),
            ],
            fill=fill,
            width=width,
        )


def fig_pipeline(path: str) -> None:
    W2, H2 = 1800, 900
    img = Image.new("RGB", (W2, H2), BG)
    d = ImageDraw.Draw(img)

    def panel_title(y, letter, text):
        d.text((60, y), letter, font=font(32, True), fill=ACCENT)
        d.text((110, y + 2), text, font=font(30, True), fill=INK)

    panel_title(50, "A", "Indexing (once per corpus)")

    y, h = 140, 120
    labels = [
        "CDISC\ndocuments\nPDF + XLSX",
        "Text\nextraction",
        "Cleaning\n7 steps",
        "Chunking\n+ section path",
        "Embedding\n384-d",
    ]
    x = 60
    boxes = []
    for label in labels:
        boxes.append((x, y, x + 300, y + h))
        x += 340
    for (xy, label) in zip(boxes, labels):
        box(d, xy, label, size=24)
    for i in range(len(boxes) - 1):
        arrow(d, (boxes[i][2] + 8, y + h / 2), (boxes[i + 1][0] - 8, y + h / 2))

    box(d, (boxes[-1][0], y + h + 60, boxes[-1][2], y + h + 180), "SQLite index\nchunks + locators",
        fill="#dce6ef", size=24)
    arrow(d, (boxes[-1][0] + 150, y + h + 8), (boxes[-1][0] + 150, y + h + 54))

    panel_title(450, "B", "Answering (per question)")

    y2, h2 = 540, 120
    box(d, (60, y2 + 70, 360, y2 + 70 + h2), "Question\n(laboratory wording)", size=24)
    box(d, (440, y2, 740, y2 + h2), "Lexical search\nBM25", fill="#eef2f6", size=24)
    box(d, (440, y2 + 150, 740, y2 + 150 + h2), "Dense search\ncosine", fill="#eef2f6", size=24)
    box(d, (820, y2 + 70, 1120, y2 + 70 + h2), "Fusion\nRRF, k = 60", fill="#dce6ef", size=24)
    box(d, (1200, y2 + 70, 1500, y2 + 70 + h2), "Reranking\nbge-reranker-base", size=24)
    box(d, (1580, y2 + 70, 1780, y2 + 70 + h2), "Answer with\ncitations", fill="#dce6ef", size=22)

    arrow(d, (368, y2 + 130), (432, y2 + 60))
    arrow(d, (368, y2 + 130), (432, y2 + 210))
    arrow(d, (748, y2 + 60), (812, y2 + 120))
    arrow(d, (748, y2 + 210), (812, y2 + 150))
    arrow(d, (1128, y2 + 130), (1192, y2 + 130))
    arrow(d, (1508, y2 + 130), (1572, y2 + 130))

    d.text((60, H2 - 80), "Both search paths use the same chunks, the same locators and the same report format,",
           font=font(22), fill=MUTED)
    d.text((60, H2 - 46), "so their comparison isolates the retriever.", font=font(22), fill=MUTED)

    img.save(path, dpi=(DPI, DPI))


def fig_retrieval(path: str, numbers: dict) -> None:
    retrievers = ["bm25", "dense", "hybrid"]
    indexes = ["raw", "clean", "clean+tuned"]
    colors = {"bm25": "#9fb3c8", "dense": ACCENT2, "hybrid": ACCENT}

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 170, W - 80, 120, H - 170

    d.text((70, 44), "B", font=font(30, True), fill=ACCENT)
    d.text((110, 48), "Hit@5 by retriever and index (95% CI)", font=font(28, True), fill=INK)

    # оси
    lo, hi = 0.5, 0.9
    d.line([(left, top), (left, bottom)], fill=INK, width=2)
    d.line([(left, bottom), (right, bottom)], fill=INK, width=2)
    for v in [0.5, 0.6, 0.7, 0.8, 0.9]:
        y = bottom - (v - lo) / (hi - lo) * (bottom - top)
        d.line([(left - 10, y), (left, y)], fill=INK, width=2)
        d.text((left - 90, y - 14), f"{v:.1f}", font=font(24), fill=INK)
        d.line([(left, y), (right, y)], fill="#dddddd", width=1)

    group_w = (right - left) / len(indexes)
    bar_w = group_w / 4.2
    for gi, idx in enumerate(indexes):
        gx = left + gi * group_w + group_w / 2
        label_w = d.textbbox((0, 0), idx, font=font(24))[2]
        d.text((gx - label_w / 2, bottom + 18), idx, font=font(24), fill=INK)
        for ri, ret in enumerate(retrievers):
            key = f"{ret}/{idx}"
            if key not in numbers:
                continue
            value = numbers[key]["hit5"]
            ci = numbers[key].get("hit5_ci") or [value, value]
            x = gx + (ri - 1) * bar_w - bar_w * 0.36
            y = bottom - (value - lo) / (hi - lo) * (bottom - top)
            d.rectangle([(x, y), (x + bar_w * 0.72, bottom)], fill=colors[ret], outline=INK)
            d.text((x, y - 34), f"{value:.3f}", font=font(20), fill=INK)
            cx = x + bar_w * 0.36
            for edge in ci:
                ey = bottom - (edge - lo) / (hi - lo) * (bottom - top)
                d.line([(cx - 14, ey), (cx + 14, ey)], fill=INK, width=2)
            y_lo = bottom - (ci[0] - lo) / (hi - lo) * (bottom - top)
            y_hi = bottom - (ci[1] - lo) / (hi - lo) * (bottom - top)
            d.line([(cx, y_hi), (cx, y_lo)], fill=INK, width=2)

    # легенда
    lx = right - 520
    ly = top - 40
    for i, ret in enumerate(retrievers):
        x = lx + i * 170
        d.rectangle([(x, ly), (x + 34, ly + 22)], fill=colors[ret], outline=INK)
        d.text((x + 44, ly - 2), ret.upper() if ret == "bm25" else ret.capitalize(),
               font=font(22), fill=INK)

    img.save(path, dpi=(DPI, DPI))


def main() -> None:
    numbers = json.loads(Path("article/numbers.json").read_text(encoding="utf-8"))
    Path("article").mkdir(exist_ok=True)
    fig_pipeline("article/fig1_pipeline.png")
    fig_retrieval("article/fig2_retrieval.png", numbers)
    print("wrote article/fig1_pipeline.png, article/fig2_retrieval.png")


if __name__ == "__main__":
    main()
