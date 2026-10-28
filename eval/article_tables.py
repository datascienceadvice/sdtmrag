"""Собирает таблицы статьи (4 таблицы) из отчётов прогонов.

Отличается от `eval/tables.py` тем, что пишет готовый к вставке в рукопись
фрагмент в стиле Annals of Laboratory Medicine: подпись сверху, номера
таблиц, знак разности всегда «новое минус старое».
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import bootstrap_ci, mean, mcnemar, paired_bootstrap  # noqa: E402

GOLD = "eval/gold.en.jsonl"


def latest(run: str) -> str:
    return sorted(glob.glob(f"eval/reports/{run}/*/report.json"))[-1]


def load(run: str) -> dict:
    pattern = f"eval/reports/{run}/*/report.json"
    if not glob.glob(pattern):
        return {}
    d = json.loads(Path(latest(run)).read_text(encoding="utf-8"))
    return {c["label"]: {it["id"]: it for it in c["items"] if it["answerable"]}
            for c in d["configs"]}


def load_many(*runs: str) -> dict:
    """Объединяет прогоны, разложенные по разным каталогам (по метке индекса)."""
    out: dict = {}
    for run in runs:
        out.update(load(run))
    return out


def val(it: dict, name: str, k: int | None) -> float:
    r = it["retrieval"]
    if name == "mrr":
        return r["reciprocal_rank"]
    return r[f"{name}_at_k"][str(k)]


def cell(values: list[float]) -> str:
    v = mean(values)
    lo, hi = bootstrap_ci(values)
    return f"{v:.3f} [{lo:.3f}-{hi:.3f}]"


def main() -> int:
    gold = {it["id"]: it for it in (json.loads(l) for l in open(GOLD, encoding="utf-8"))}
    dense = load("dense_norerank")
    bm25 = load("bm25_norerank")
    hyb = load("hybrid_norerank")
    rer = load_many("dense_rerank", "dense_rerank2")

    out: list[str] = []

    # Таблица 1. Состав набора
    out += [
        "**Table 1. Composition of the benchmark**",
        "",
        "| Register | Questions | Locators | Target of the reference answer |",
        "|---|---:|---:|---|",
    ]
    descr = {
        "lookup": "a variable definition",
        "structure": "a dataset description",
        "concept": "a narrative section",
        "procedural": "a multi-step task",
        "out-of-scope": "none",
    }
    locs = {"lookup": "1 + alias", "structure": "1 + alias", "concept": "1",
            "procedural": "2-3", "out-of-scope": "none"}
    for reg in ("lookup", "structure", "concept", "procedural", "out-of-scope"):
        n = sum(1 for it in gold.values() if it["register"] == reg)
        out.append(f"| {reg} | {n} | {locs[reg]} | {descr[reg]} |")
    total = len(gold)
    ans = sum(1 for it in gold.values() if it["answerable"])
    out.append(f"| **Total** | **{total}** | | **{ans} answerable, {total - ans} for refusal** |")
    out.append("")

    # Таблица 2. Качество поиска
    rows = [
        ("BM25", "raw", bm25["raw"]),
        ("BM25", "cleaned", bm25["clean"]),
        ("BM25", "cleaned, tuned chunks", bm25["clean+tuned"]),
        ("Dense", "raw", dense["raw"]),
        ("Dense", "cleaned", dense["clean"]),
        ("Dense", "cleaned, tuned chunks", dense["clean+tuned"]),
        ("Dense + cross-encoder", "raw", rer.get("raw")),
        ("Dense + cross-encoder", "cleaned", rer.get("clean")),
        ("Dense + cross-encoder", "cleaned, tuned chunks", rer.get("clean+tuned")),
        ("Hybrid (RRF)", "raw", hyb["raw"]),
        ("Hybrid (RRF)", "cleaned", hyb["clean"]),
        ("Hybrid (RRF)", "cleaned, tuned chunks", hyb["clean+tuned"]),
    ]
    out += [
        "**Table 2. Retrieval quality (173 answerable questions)**",
        "",
        "| Retriever | Index | Hit@5 | Recall@5 | nDCG@5 | MRR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, index, cfg in rows:
        if not cfg:
            continue
        cells = [cell([val(it, m, k) for it in cfg.values()])
                 for m, k in (("hit", 5), ("recall", 5), ("ndcg", 5), ("mrr", None))]
        out.append(f"| {name} | {index} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Values are mean [95% bootstrap confidence interval]. Rows without the cross-encoder "
               "use plain dense or lexical retrieval; the cross-encoder rows apply bge-reranker-base "
               "to 20 candidates of 256 tokens on the same index.")
    out.append("")

    # Таблица 3. Стратификация
    out += [
        "**Table 3. Retrieval quality by stratum and register (dense retrieval, cleaned index)**",
        "",
        "| Group | n | Recall@5 | nDCG@5 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for by in ("stratum", "register"):
        groups: dict[str, list[dict]] = {}
        for iid, it in dense["clean"].items():
            groups.setdefault(gold[iid].get(by) or "other", []).append(it)
        for g in sorted(groups):
            sub = groups[g]
            r = mean([val(it, "recall", 5) for it in sub])
            n = mean([val(it, "ndcg", 5) for it in sub])
            m = mean([val(it, "mrr", None) for it in sub])
            label = f"{by}: {g}"
            out.append(f"| {label} | {len(sub)} | {r:.3f} | {n:.3f} | {m:.3f} |")
    out.append("")

    # Таблица 4. Парные сравнения
    out += [
        "**Table 4. Paired comparisons**",
        "",
        "| Comparison | Metric | Before | After | Difference [95% CI] | P |",
        "|---|---|---:|---:|---|---:|",
    ]
    pairs = [
        ("Cleaning: raw vs cleaned (dense)", dense["raw"], dense["clean"]),
        ("Chunk tuning: cleaned vs tuned (dense)", dense["clean"], dense["clean+tuned"]),
        ("Lexical vs dense (cleaned index)", bm25["clean"], dense["clean"]),
        ("Fusion: dense vs hybrid (cleaned index)", dense["clean"], hyb["clean"]),
        ("Reranking: dense vs dense + cross-encoder (cleaned)", dense["clean"], rer["clean"]),
    ]
    for label, A, B in pairs:
        ids = sorted(set(A) & set(B))
        for name, k, title in (("hit", 5, "Hit@5"), ("recall", 5, "Recall@5"),
                               ("ndcg", 5, "nDCG@5"), ("mrr", None, "MRR")):
            av = [val(A[i], name, k) for i in ids]
            bv = [val(B[i], name, k) for i in ids]
            d, lo, hi = paired_bootstrap(av, bv)
            # подпись таблицы обещает «после минус до»: меняем знак и границы
            d, lo, hi = -d, -hi, -lo
            if name in ("hit", "recall"):
                p = mcnemar(av, bv)[2]
                p_s = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            else:
                p_s = "—"
            out.append(f"| {label} | {title} | {mean(av):.3f} | {mean(bv):.3f} | "
                       f"{d:+.3f} [{lo:+.3f}; {hi:+.3f}] | {p_s} |")
    out.append("")
    out.append("Differences are After minus Before; the confidence interval is the inferential "
               "statement for continuous metrics, and the exact McNemar test is given for binary "
               "hit rates. Holm adjustment over the four primary comparisons gives adjusted "
               "P = 0.002 for fusion (Hit@5) and 0.078 for cleaning (Hit@5). The cross-encoder is "
               "compared on the cleaned index here; its effect on the other two indexes is "
               "reported in the text.")
    out.append("")

    Path("article/tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
