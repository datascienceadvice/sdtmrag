"""Публикационные таблицы из отчётов прогонов.

Читает `report.json` (в них есть результаты по каждому вопросу), присоединяет
регистр и страту вопроса из `eval/gold.en.jsonl` и считает все числа, которые
попадут в статью: метрики с доверительными интервалами, парные сравнения,
стратифицированные разбивки.

Запуск:
    python eval/tables.py --dir eval/reports/dense_norerank \
                          --dir eval/reports/bm25_norerank \
                          --dir eval/reports/hybrid_norerank \
                          --out article/tables.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import bootstrap_ci, fmt_ci, mcnemar, mean, paired_bootstrap  # noqa: E402


def latest_report(directory: str) -> Path:
    cands = sorted(Path(directory).glob("*/report.json"))
    if not cands:
        raise SystemExit(f"no report.json under {directory}")
    return cands[-1]


def load_gold(path: str = "eval/gold.en.jsonl") -> dict:
    return {it["id"]: it for it in (json.loads(l) for l in open(path, encoding="utf-8"))}


def item_value(it: dict, name: str, k: int | None) -> float:
    r = it["retrieval"]
    if name == "hit":
        return r["hit_at_k"][str(k)]
    if name == "recall":
        return r["recall_at_k"][str(k)]
    if name == "ndcg":
        return r["ndcg_at_k"][str(k)]
    return r["reciprocal_rank"]


def answerable(cfg: dict) -> list[dict]:
    return [it for it in cfg["items"] if it["answerable"]]


def load_runs(dirs: list[str], gold: dict) -> dict[tuple[str, str], dict]:
    runs = {}
    for d in dirs:
        rep = json.loads(latest_report(d).read_text(encoding="utf-8"))
        for cfg in rep["configs"]:
            key = (rep.get("retriever") or cfg.get("retriever", "dense"), cfg["label"])
            runs[key] = {"cfg": cfg, "report": rep}
    return runs


def table_composition(gold: dict) -> str:
    rows = [
        ("lookup", "tab", "single", "variable definition"),
        ("structure", "tab", "single", "dataset description"),
        ("concept", "prose", "single", "narrative section"),
        ("procedural", "mix", "multi", "multi-step task"),
        ("out-of-scope", "none", "none", "refusal"),
    ]
    lines = ["| Register | Questions | Locators | Reference target |", "|---|---:|---|---|"]
    for reg, _, _, desc in rows:
        n = sum(1 for it in gold.values() if it["register"] == reg)
        multi = sum(1 for it in gold.values() if it["register"] == reg and len(it["expected"]) > 1)
        loc = "1-3" if reg == "procedural" else ("none" if reg == "out-of-scope" else "1")
        lines.append(f"| {reg} | {n} | {loc} | {desc} |")
    total = len(gold)
    ans = sum(1 for it in gold.values() if it["answerable"])
    lines.append(f"| **total** | **{total}** | | **{ans} answerable, {total - ans} refusal** |")
    return "\n".join(lines)


def table_metrics(runs: dict, keys: list[tuple[str, str]], ks: list[int], title: str) -> str:
    head = "| Retriever | Index | Hit@5 | Recall@5 | nDCG@5 | MRR |"
    lines = [f"**{title}**", "", head, "|---|---|---:|---:|---:|---:|"]
    for key in keys:
        if key not in runs:
            continue
        cfg = runs[key]["cfg"]
        its = answerable(cfg)
        if not its:
            continue
        cells = []
        for name, k in [("hit", 5), ("recall", 5), ("ndcg", 5), ("mrr", None)]:
            vals = [item_value(it, name, k) for it in its]
            lo, hi = bootstrap_ci(vals)
            cells.append(fmt_ci(mean(vals), lo, hi))
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("_Values are mean [95% bootstrap CI] over answerable questions._")
    return "\n".join(lines)


def table_paired(runs: dict, a: tuple, b: tuple, label: str) -> str:
    if a not in runs or b not in runs:
        return f"_(skipped {label}: run missing)_"
    A = {it["id"]: it for it in answerable(runs[a]["cfg"])}
    B = {it["id"]: it for it in answerable(runs[b]["cfg"])}
    ids = sorted(set(A) & set(B))
    lines = [
        f"**{label}** (n={len(ids)})",
        "",
        "| Metric | " + f"{a[1]}" + " | " + f"{b[1]}" + " | Difference [95% CI] | P |",
        "|---|---:|---:|---|---:|",
    ]
    for name, k, title in [
        ("hit", 5, "Hit@5"),
        ("recall", 5, "Recall@5"),
        ("ndcg", 5, "nDCG@5"),
        ("mrr", None, "MRR"),
    ]:
        av = [item_value(A[i], name, k) for i in ids]
        bv = [item_value(B[i], name, k) for i in ids]
        d, lo, hi = paired_bootstrap(av, bv)
        if name in ("hit", "recall"):
            _, _, p = mcnemar(av, bv)
            p_s = f"{p:.4f}"
        else:
            p_s = "—"
        lines.append(
            f"| {title} | {mean(av):.3f} | {mean(bv):.3f} | "
            f"{d:+.3f} [{lo:+.3f}; {hi:+.3f}] | {p_s} |"
        )
    return "\n".join(lines)


def table_stratified(runs: dict, key: tuple, gold: dict, by: str) -> str:
    if key not in runs:
        return f"_(skipped: {key} missing)_"
    its = answerable(runs[key]["cfg"])
    groups: dict[str, list[dict]] = {}
    for it in its:
        g = gold[it["id"]].get(by) or "other"
        groups.setdefault(g, []).append(it)
    lines = [
        f"**{by} strata ({key[0]} / {key[1]})**",
        "",
        f"| {by} | n | Recall@5 | nDCG@5 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in sorted(groups):
        sub = groups[g]
        r = [item_value(it, "recall", 5) for it in sub]
        n = [item_value(it, "ndcg", 5) for it in sub]
        m = [item_value(it, "mrr", None) for it in sub]
        lines.append(f"| {g} | {len(sub)} | {mean(r):.3f} | {mean(n):.3f} | {mean(m):.3f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--out", default="article/tables.md")
    ap.add_argument("--json", default="article/numbers.json")
    ap.add_argument("--gold", default="eval/gold.en.jsonl")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    runs = load_runs(args.dir, gold)
    keys = sorted(runs)

    sections = ["## Table 1. Benchmark composition", "", table_composition(gold), ""]

    no_rerank = [k for k in keys if not runs[k]["cfg"]["rerank"]]
    with_rerank = [k for k in keys if runs[k]["cfg"]["rerank"]]
    sections += [
        "## Table 2. Retrieval quality without the cross-encoder",
        "",
        table_metrics(runs, no_rerank, [5], "No reranking"),
        "",
    ]
    if with_rerank:
        sections += [
            "## Table 3. Retrieval quality with the cross-encoder",
            "",
            table_metrics(runs, with_rerank, [5], "With bge-reranker-base"),
            "",
        ]

    primary = ("dense", "clean")
    if primary not in runs:
        primary = keys[0] if keys else ("dense", "clean")
    sections += ["## Table 4. Stratified results", ""]
    for by in ("stratum", "register"):
        sections += [table_stratified(runs, primary, gold, by), ""]

    sections += ["## Table 5. Paired comparisons", ""]
    for a, b, label in [
        (("dense", "raw"), ("dense", "clean"), "Dense: cleaning effect"),
        (("dense", "clean"), ("dense", "clean+tuned"), "Dense: chunk tuning effect"),
        (("bm25", "clean"), ("dense", "clean"), "Lexical vs dense (same index)"),
        (("dense", "clean"), ("hybrid", "clean"), "Dense vs hybrid (same index)"),
    ]:
        sections += [table_paired(runs, a, b, label), ""]

    out = "\n".join(sections)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(out, encoding="utf-8")

    numbers = {}
    for key in keys:
        its = answerable(runs[key]["cfg"])
        entry = {"n": len(its), "latency_p50": runs[key]["cfg"]["latency_ms"]["p50_ms"]}
        for name, k, label in (("hit", 5, "hit5"), ("recall", 5, "recall5"),
                               ("ndcg", 5, "ndcg5"), ("mrr", None, "mrr")):
            vals = [item_value(it, name, k) for it in its]
            lo, hi = bootstrap_ci(vals)
            entry[label] = mean(vals)
            entry[label + "_ci"] = [lo, hi]
        numbers["/".join(key)] = entry
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
