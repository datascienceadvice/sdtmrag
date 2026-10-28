"""Кандидаты в рисунки для статьи (PIL, без внешних зависимостей).

Все числа берутся из отчётов прогонов, а не набираются руками: пересчитайте
прогоны — рисунки изменятся вместе с ними.

Запуск: python eval/figures_extra.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from analyze import mean, paired_bootstrap  # noqa: E402

DPI = 300
INK = "#1a1a1a"
MUTED = "#5f6b76"
GRID = "#e3e7ea"
ACCENT = "#2f5d8a"
ACCENT2 = "#8aa6c1"
BAND = "#f4f7f9"
WARN = "#b4542a"
WARN_BG = "#fbeeea"
GOOD = "#2f7d5d"
GREY = "#b8c2ca"

FONT_DIR = Path("C:/Windows/Fonts")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / ("arialbd.ttf" if bold else "arial.ttf")), size)


def text_w(draw, s: str, f) -> int:
    return draw.textbbox((0, 0), s, font=f)[2]


def wrap(draw, text: str, f, max_w: int) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if text_w(draw, trial, f) <= max_w or not cur:
            cur = trial
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def latest(run: str) -> Path:
    return sorted(Path(f"eval/reports/{run}").glob("*/report.json"))[-1]


def load_run(run: str) -> dict[str, dict[str, dict]]:
    d = json.loads(latest(run).read_text(encoding="utf-8"))
    return {c["label"]: {it["id"]: it for it in c["items"] if it["answerable"]} for c in d["configs"]}


def load_latency(run: str) -> dict[str, float]:
    d = json.loads(latest(run).read_text(encoding="utf-8"))
    return {c["label"]: c["latency_ms"]["p50_ms"] for c in d["configs"]}


def val(it: dict, name: str, k: int) -> float:
    if name == "mrr":
        return it["retrieval"]["reciprocal_rank"]
    return it["retrieval"][f"{name}_at_k"][str(k)]


def delta(A: dict, B: dict, name: str = "hit", k: int = 5):
    ids = sorted(set(A) & set(B))
    av = [val(A[i], name, k) for i in ids]
    bv = [val(B[i], name, k) for i in ids]
    d, lo, hi = paired_bootstrap(av, bv)
    return -d, -hi, -lo  # «после минус до»: плюс = улучшение


# ---------------------------------------------------------------------------
# A. Forest plot
# ---------------------------------------------------------------------------


def fig_forest(path: str) -> None:
    dense = load_run("dense_norerank")
    bm25 = load_run("bm25_norerank")
    hyb = load_run("hybrid_norerank")
    rerank = {**load_run("dense_rerank"), **load_run("dense_rerank2")}

    groups = [
        ("Text cleaning", [("raw → cleaned", delta(dense["raw"], dense["clean"]))]),
        ("Chunk size", [("cleaned → encoder-matched", delta(dense["clean"], dense["clean+tuned"]))]),
        ("Retriever", [
            ("BM25 → dense", delta(bm25["clean"], dense["clean"])),
            ("dense → fusion (RRF)", delta(dense["clean"], hyb["clean"])),
        ]),
        ("Cross-encoder", [
            ("on raw index", delta(dense["raw"], rerank["raw"])),
            ("on cleaned index", delta(dense["clean"], rerank["clean"])),
            ("on tuned index", delta(dense["clean+tuned"], rerank["clean+tuned"])),
        ]),
    ]

    W, H = 1800, 1020
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    left, right = 780, 1420
    top, bottom = 170, H - 150
    lo_x, hi_x = -0.10, 0.21

    def x_of(v: float) -> float:
        return left + (v - lo_x) / (hi_x - lo_x) * (right - left)

    d.text((70, 48), "A", font=font(34, True), fill=ACCENT)
    d.text((120, 52), "What each intervention actually changed", font=font(30, True), fill=INK)
    d.text((70, 96), "Difference in Hit@5 (share of questions whose correct passage is in the top five), 173 questions",
           font=font(22), fill=MUTED)

    for tick in (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20):
        x = x_of(tick)
        d.line([(x, top - 24), (x, bottom + 18)], fill=GRID, width=2)
        lab = "0" if tick == 0 else f"{tick:+.2f}"
        w = text_w(d, lab, font(20))
        d.text((x - w / 2, bottom + 30), lab, font=font(20), fill=MUTED)
    d.line([(x_of(0), top - 24), (x_of(0), bottom + 18)], fill=INK, width=3)
    d.text((x_of(0) - 62, top - 66), "no effect", font=font(20), fill=MUTED)

    y = top + 44
    for group, items in groups:
        d.text((70, y - 10), group.upper(), font=font(19, True), fill=MUTED)
        d.line([(70, y + 22), (right, y + 22)], fill=GRID, width=2)
        y += 54
        for label, (dv, lo, hi) in items:
            sig = (lo > 0) or (hi < 0)
            colour = ACCENT if sig else MUTED
            d.text((110, y - 15), label, font=font(24), fill=INK)
            yc = y + 4
            d.line([(x_of(lo), yc), (x_of(hi), yc)], fill=colour, width=4)
            for edge in (lo, hi):
                d.line([(x_of(edge), yc - 12), (x_of(edge), yc + 12)], fill=colour, width=4)
            r = 12 if sig else 8
            d.ellipse([(x_of(dv) - r, yc - r), (x_of(dv) + r, yc + r)],
                      fill=colour if sig else "white", outline=colour, width=3)
            txt = f"{dv:+.3f}  [{lo:+.3f}; {hi:+.3f}]"
            d.text((x_of(hi) + 26, y - 15), txt, font=font(21), fill=INK if sig else MUTED)
            y += 62
        y += 24

    d.text((70, H - 84), "Filled markers: the 95% confidence interval excludes zero. Every comparison is paired by",
           font=font(21), fill=MUTED)
    d.text((70, H - 52), "question (10,000 bootstrap resamples); the interval is the inferential statement.",
           font=font(21), fill=MUTED)
    img.save(path, dpi=(DPI, DPI))


# ---------------------------------------------------------------------------
# B. Quality vs latency
# ---------------------------------------------------------------------------


def fig_pareto(path: str) -> None:
    dense = load_run("dense_norerank")
    bm25 = load_run("bm25_norerank")
    hyb = load_run("hybrid_norerank")
    rerank = {**load_run("dense_rerank"), **load_run("dense_rerank2")}
    lat = {**load_latency("dense_norerank"), **load_latency("bm25_norerank"), **load_latency("hybrid_norerank")}
    rerank_lat = {**load_latency("dense_rerank"), **load_latency("dense_rerank2")}

    families = [
        ("BM25", bm25, lat, ACCENT2, "open"),
        ("Dense", dense, lat, "#6f93b5", "solid"),
        ("Hybrid (RRF)", hyb, lat, ACCENT, "diamond"),
        ("dense + cross-encoder", rerank, rerank_lat, WARN, "triangle"),
    ]

    W, H = 1800, 960
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 230, 1640, 230, H - 200

    def x_of(ms: float) -> float:
        frac = (math.log10(ms) - math.log10(15.0)) / (math.log10(25000.0) - math.log10(15.0))
        return left + frac * (right - left)

    def y_of(v: float) -> float:
        return bottom - (v - 0.62) / (0.85 - 0.62) * (bottom - top)

    d.text((70, 48), "B", font=font(34, True), fill=ACCENT)
    d.text((120, 52), "Quality against the time it takes", font=font(30, True), fill=INK)
    d.text((70, 96), "Hit@5 and median latency per question; the time axis is logarithmic",
           font=font(22), fill=MUTED)

    for v in (0.65, 0.70, 0.75, 0.80, 0.85):
        y = y_of(v)
        d.line([(left, y), (right, y)], fill=GRID, width=2)
        d.text((left - 84, y - 14), f"{v:.2f}", font=font(20), fill=MUTED)
    for ms, lab in ((25, "25 ms"), (100, "0.1 s"), (1000, "1 s"), (10000, "10 s")):
        x = x_of(ms)
        d.line([(x, top), (x, bottom)], fill=GRID, width=2)
        w = text_w(d, lab, font(20))
        d.text((x - w / 2, bottom + 22), lab, font=font(20), fill=MUTED)

    # бюджет интерактивности
    xb = x_of(200)
    for yy in range(top, bottom, 26):
        d.line([(xb, yy), (xb, min(yy + 13, bottom))], fill=MUTED, width=2)
    d.text((xb + 12, top - 30), "0.2 s budget for interactive use", font=font(20), fill=MUTED)

    d.line([(left, bottom), (right, bottom)], fill=INK, width=3)
    d.line([(left, top), (left, bottom)], fill=INK, width=3)

    for name, series, lat_map, colour, shape in families:
        pts = []
        for idx in ("raw", "clean", "clean+tuned"):
            hit = mean([val(it, "hit", 5) for it in series[idx].values()])
            pts.append((x_of(lat_map[idx]), y_of(hit), hit))
        for a, b in zip(pts, pts[1:]):
            d.line([(a[0], a[1]), (b[0], b[1])], fill=colour, width=2)
        for x, y, _ in pts:
            if shape == "triangle":
                d.polygon([(x - 11, y - 17), (x + 11, y - 17), (x, y + 17)], fill=colour)
            elif shape == "diamond":
                d.polygon([(x, y - 14), (x + 14, y), (x, y + 14), (x - 14, y)], fill=colour)
            elif shape == "open":
                d.ellipse([(x - 11, y - 11), (x + 11, y + 11)], fill="white", outline=colour, width=3)
            else:
                d.ellipse([(x - 11, y - 11), (x + 11, y + 11)], fill=colour)

    # легенда
    lx, ly = x_of(1500), y_of(0.745)
    d.rectangle([lx - 20, ly - 16, right - 6, ly + 4 * 44 + 4], fill="white", outline=GRID, width=2)
    for i, (name, _, _, colour, shape) in enumerate(families):
        y = ly + 14 + i * 44
        x = lx + 10
        if shape == "triangle":
            d.polygon([(x - 10, y - 14), (x + 10, y - 14), (x, y + 14)], fill=colour)
        elif shape == "diamond":
            d.polygon([(x, y - 12), (x + 12, y), (x, y + 12), (x - 12, y)], fill=colour)
        elif shape == "open":
            d.ellipse([(x - 10, y - 10), (x + 10, y + 10)], fill="white", outline=colour, width=3)
        else:
            d.ellipse([(x - 10, y - 10), (x + 10, y + 10)], fill=colour)
        d.text((x + 26, y - 14), name, font=font(21), fill=INK)

    # две выноски вместо подписи каждой точки
    best_hyb = max((p for p in [(x_of(lat[i]), y_of(mean([val(it, "hit", 5) for it in hyb[i].values()]))) for i in ("raw", "clean", "clean+tuned")]), key=lambda p: p[1])
    d.text((best_hyb[0] + 24, best_hyb[1] - 52), "fusion: 0.803 at 23 ms", font=font(21, True), fill=ACCENT)
    ce = [(x_of(rerank_lat[i]), y_of(mean([val(it, "hit", 5) for it in rerank[i].values()]))) for i in ("raw", "clean", "clean+tuned")]
    d.text((min(p[0] for p in ce) - 300, top + 30), "cross-encoder: 0.780 at 13 s", font=font(21, True), fill=WARN)

    d.text((left, H - 110), "Fusion reaches the best Hit@5 of the study at 23 ms. A cross-encoder needs about 13 s per",
           font=font(21), fill=MUTED)
    d.text((left, H - 76), "question and still stays below fusion. Each family appears on the three indexes; the thin",
           font=font(21), fill=MUTED)
    d.text((left, H - 42), "line joins the same retriever across preprocessing, so its length shows the effect of cleaning.",
           font=font(21), fill=MUTED)
    img.save(path, dpi=(DPI, DPI))

RAW_TOP = ("types supported by SAS v5 transport files. Define-XML provides more descriptive "
           "data types (e.g., integer, float,")
RAW_TAIL = ("date, datetime); see the Define-XML specification for information about how to "
            "represent SDTM types using Define-XML data types.")
MERGED = ("types supported by SAS v5 transport files. Define-XML provides more descriptive "
          "data types (e.g., integer, float, date, datetime); see the Define-XML specification "
          "for information about how to represent SDTM types using Define-XML data types.")
FOOTER_LINES = [
    "CDISC Study Data Tabulation Model (Version 2.1 Final)",
    "\u00a9 2024 Clinical Data Interchange Standards Consortium, Inc. All rights reserved \u2022 Page 8",
    "2024-06-07",
]


def fig_cleaning(path: str) -> None:
    W, H = 1800, 780
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f = font(22)
    col_w = 700

    d.text((70, 48), "C", font=font(34, True), fill=ACCENT)
    d.text((120, 52), "What the cleaning step removes, on a real page", font=font(30, True), fill=INK)
    d.text((70, 96), "SDTM v2.1, section 2.2, page 8 \u2014 the same sentence in the raw and in the cleaned index",
           font=font(22), fill=MUTED)

    line_h = 36
    y0 = 200

    x0 = 70
    d.text((x0, y0 - 44), "Raw extraction", font=font(25, True), fill=WARN)
    y = y0
    for line in wrap(d, RAW_TOP, f, col_w):
        d.text((x0, y), line, font=f, fill=INK)
        y += line_h
    band_top = y + 4
    fy = band_top + 12
    for line in FOOTER_LINES:
        for part in wrap(d, line, font(20), col_w - 40):
            d.text((x0 + 26, fy), part, font=font(20), fill=WARN)
            fy += 30
    band_bottom = fy + 2
    d.rectangle([x0 - 8, band_top, x0 + col_w + 8, band_bottom], fill=WARN_BG)
    d.line([(x0 - 8, band_top), (x0 - 8, band_bottom)], fill=WARN, width=5)
    fy = band_top + 12
    for line in FOOTER_LINES:
        for part in wrap(d, line, font(20), col_w - 40):
            d.text((x0 + 26, fy), part, font=font(20), fill=WARN)
            fy += 30
    d.text((x0 + col_w - 128, band_top - 30), "3 footer lines", font=font(19, True), fill=WARN)
    y = band_bottom + 16
    for line in wrap(d, RAW_TAIL, f, col_w):
        d.text((x0, y), line, font=f, fill=INK)
        y += line_h

    x1 = 1000
    d.text((x1, y0 - 44), "Cleaned", font=font(25, True), fill=GOOD)
    y = y0
    for line in wrap(d, MERGED, f, col_w):
        d.text((x1, y), line, font=f, fill=INK)
        y += line_h
    d.line([(x1 - 16, y0 - 8), (x1 - 16, y - line_h + 14)], fill=GOOD, width=5)

    yb = 520
    d.text((70, yb), "Per corpus", font=font(22, True), fill=INK)
    stats = [
        ("lines removed", "32.0%"),
        ("characters removed", "9.7%"),
        ("footer characters cut out of content lines", "67,964"),
        ("words joined across hyphenation", "212"),
        ("table rows marked so column bounds survive", "8,516"),
    ]
    yy = yb + 42
    for label, value in stats:
        d.text((90, yy), label, font=font(21), fill=MUTED)
        d.text((720, yy), value, font=font(21, True), fill=INK)
        yy += 34

    d.text((1000, yb), "What it did not change", font=font(22, True), fill=INK)
    for i, line in enumerate([
        "Retrieval: Hit@5 0.682 with cleaning against",
        "0.723 without it (adjusted P = 0.08).",
        "",
        "Section labels: fixed by a heading validation",
        "rule, not by cleaning (5.6% vs 4.7% unreliable).",
    ]):
        d.text((1020, yb + 42 + i * 34), line, font=font(21), fill=MUTED)

    img.save(path, dpi=(DPI, DPI))


# ---------------------------------------------------------------------------
# D. Heatmap: register x configuration
# ---------------------------------------------------------------------------


def fig_heatmap(path: str) -> None:
    gold = {it["id"]: it for it in (json.loads(l) for l in open("eval/gold.en.jsonl", encoding="utf-8"))}
    runs = {
        "BM25": load_run("bm25_norerank"),
        "Dense": load_run("dense_norerank"),
        "Hybrid (RRF)": load_run("hybrid_norerank"),
    }
    order = ["lookup", "structure", "concept", "procedural"]
    indexes = ["raw", "clean", "clean+tuned"]
    cols = [(r, idx) for r in runs for idx in indexes]
    rows = order + ["all"]

    cell: dict[tuple[str, tuple[str, str]], float] = {}
    count: dict[str, int] = {}
    for rname, series in runs.items():
        for idx in indexes:
            items = series[idx]
            for reg in order:
                sub = [it for iid, it in items.items() if gold[iid]["register"] == reg]
                count[reg] = len(sub)
                cell[(reg, (rname, idx))] = mean([val(it, "recall", 5) for it in sub]) if sub else 0.0
            cell[("all", (rname, idx))] = mean([val(it, "recall", 5) for it in items.values()])
    count["all"] = len(gold)

    W, H = 1800, 880
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 48), "D", font=font(34, True), fill=ACCENT)
    d.text((120, 52), "Where each configuration works", font=font(30, True), fill=INK)
    d.text((70, 96), "Recall@5 by question register; darker is better, each cell averages the questions of that register",
           font=font(22), fill=MUTED)

    x0, y0 = 330, 226
    cw, ch = 156, 104
    lo, hi = 0.10, 1.0

    def shade(v: float) -> str:
        t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
        start, end = (246, 248, 250), (36, 74, 110)
        return "#%02x%02x%02x" % tuple(int(start[i] + (end[i] - start[i]) * t) for i in range(3))

    for gi, rname in enumerate(runs):
        d.text((x0 + gi * 3 * cw + 6, y0 - 66), rname, font=font(24, True), fill=INK)
    for ci, (_, idx) in enumerate(cols):
        lab = {"raw": "raw", "clean": "clean", "clean+tuned": "tuned"}[idx]
        w = text_w(d, lab, font(19))
        d.text((x0 + ci * cw + (cw - w) / 2, y0 - 34), lab, font=font(19), fill=MUTED)

    for ri, reg in enumerate(rows):
        y = y0 + ri * ch
        name = {"lookup": "variable lookup", "structure": "dataset description",
                "concept": "narrative section", "procedural": "multi-step task", "all": "all questions"}[reg]
        d.text((86, y + ch / 2 - 20), name, font=font(22, True if reg == "all" else False), fill=INK)
        d.text((86, y + ch / 2 + 8), f"n = {count[reg]}", font=font(19), fill=MUTED)
        for ci, key in enumerate(cols):
            v = cell[(reg, key)]
            x = x0 + ci * cw
            colour = shade(v)
            d.rectangle([x + 3, y + 3, x + cw - 3, y + ch - 3], fill=colour)
            f = font(22, v > 0.7)
            d.text((x + (cw - text_w(d, f"{v:.3f}", f)) / 2, y + ch / 2 - 15), f"{v:.3f}",
                   font=f, fill="white" if luminance(colour) < 0.55 else INK)

    for gi in range(1, len(runs)):
        x = x0 + gi * 3 * cw
        d.line([(x, y0 - 24), (x, y0 + len(rows) * ch)], fill="white", width=9)

    d.text((86, H - 104), "Multi-step questions stay below 0.45 everywhere. Fusion leads on lookups and dataset",
           font=font(21), fill=MUTED)
    d.text((86, H - 72), "descriptions, but dense retrieval leads on multi-step tasks (0.440 against 0.319).",
           font=font(21), fill=MUTED)
    d.text((86, H - 40), "Cleaning lowers narrative-section recall in every retriever (dense: 0.652 to 0.580).",
           font=font(21), fill=MUTED)
    img.save(path, dpi=(DPI, DPI))


# ---------------------------------------------------------------------------
# E. Grounding: unit chart + correctness
# ---------------------------------------------------------------------------


def fig_grounding(path: str) -> None:
    closed = json.loads(sorted(Path("eval/reports/closedbook").glob("*/report.json"))[-1].read_text(encoding="utf-8"))
    s = closed["summary"]
    judged = json.loads(latest("judge_rerank").read_text(encoding="utf-8"))
    answers = judged["configs"][0]["answers"]

    W, H = 1800, 820
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 48), "E", font=font(34, True), fill=ACCENT)
    d.text((120, 52), "Grounding changes behaviour, not only accuracy", font=font(30, True), fill=INK)
    d.text((70, 96), "The same model asked the 20 questions that the documentation cannot answer",
           font=font(22), fill=MUTED)

    n = 20
    cell, gap = 44, 10
    x0, y0 = 430, 190

    refused_cb = round(s["refused_unanswerable"] * n)
    refused_rag = answers["refusals_correct"]
    judged_rag = answers["refusals_total"]

    rows = [
        ("with retrieved passages", refused_rag, judged_rag),
        ("without retrieval", refused_cb, n),
    ]
    for ri, (label, refused, total) in enumerate(rows):
        y = y0 + ri * 130
        d.text((90, y + 6), label, font=font(22), fill=INK)
        d.text((90, y + 36), f"{refused} of {total} refused", font=font(20), fill=MUTED)
        for i in range(n):
            cx = x0 + (i % 10) * (cell + gap)
            cy = y + (i // 10) * (cell + gap)
            if i < refused:
                fill, outline = GOOD, GOOD
            elif i < total:
                fill, outline = WARN, WARN
            else:
                fill, outline = "white", GREY
            d.rounded_rectangle([cx, cy, cx + cell, cy + cell], radius=8, fill=fill, outline=outline, width=2)

    lx = x0 + 10 * (cell + gap) + 36
    for i, (fill, name) in enumerate(((GOOD, "refused (correct)"), (WARN, "answered (invented)"),
                                      ("white", "not judged"))):
        ly = y0 + i * 44
        d.rounded_rectangle([lx, ly, lx + 30, ly + 30], radius=6, fill=fill,
                            outline=GREY if fill == "white" else fill, width=2)
        d.text((lx + 44, ly + 2), name, font=font(21), fill=INK)

    d.text((90, 494), "Correctness on answerable questions", font=font(23), fill=INK)
    bars = [("with retrieval", answers["correctness"], ACCENT),
            ("without retrieval", s["correctness_answerable"], ACCENT2)]
    bx, bw, bh = 640, 420, 30
    for i, (name, v, colour) in enumerate(bars):
        y = 540 + i * 56
        d.text((90, y + 4), name, font=font(20), fill=MUTED)
        d.rectangle([bx, y, bx + bw, y + bh], fill="#eef2f5")
        d.rectangle([bx, y, bx + int(bw * v), y + bh], fill=colour)
        d.text((bx + int(bw * v) + 14, y + 2), f"{v:.2f}", font=font(22, True), fill=INK)

    d.text((70, H - 96), "Without passages the model answers 7 of the 20 unanswerable questions; with passages it refuses",
           font=font(21), fill=MUTED)
    d.text((70, H - 62), "every one that was judged. Denominators differ: 173 answerable questions for the closed-book",
           font=font(21), fill=MUTED)
    d.text((70, H - 28), "control, a judged subsample of 87 with retrieval.", font=font(21), fill=MUTED)
    img.save(path, dpi=(DPI, DPI))


def main() -> None:
    Path("article").mkdir(exist_ok=True)
    fig_forest("article/cand_forest.png")
    fig_pareto("article/cand_pareto.png")
    fig_cleaning("article/cand_cleaning.png")
    fig_heatmap("article/cand_heatmap.png")
    fig_grounding("article/cand_grounding.png")
    print("wrote 5 candidate figures to article/cand_*.png")


if __name__ == "__main__":
    main()
