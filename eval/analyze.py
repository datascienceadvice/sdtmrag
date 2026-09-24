#!/usr/bin/env python3
"""Пост-обработка отчётов `sdtmrag eval`: доверительные интервалы и парные тесты.

Отчёт `report.json` содержит результат по каждому вопросу, поэтому метрики можно
пересчитать с оценкой неопределённости — это важно при наборе в ~90 вопросов,
где разница в несколько процентных пунктов может быть случайной.

Пример:
    python eval/analyze.py eval/reports/<run>/report.json
    python eval/analyze.py --pair clean:clean+tuned eval/reports/<run>/report.json
"""
import argparse
import json
import math
import random
import sys

BOOTSTRAP = 2000
SEED = 20240924


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ids(path):
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def answerable(cfg, only_ids=None):
    out = []
    for it in cfg["items"]:
        if not it.get("answerable"):
            continue
        if only_ids is not None and it["id"] not in only_ids:
            continue
        out.append(it)
    return out


def metric(it, name, k):
    r = it["retrieval"]
    if name == "mrr":
        return r["reciprocal_rank"]
    if name == "hit":
        return r["hit_at_k"][str(k)]
    if name == "recall":
        return r["recall_at_k"][str(k)]
    if name == "precision":
        return r["precision_at_k"][str(k)]
    if name == "ndcg":
        return r["ndcg_at_k"][str(k)]
    raise ValueError(name)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def bootstrap_ci(values, iters=BOOTSTRAP, seed=SEED):
    """95% доверительный интервал для среднего (перцентильный bootstrap)."""
    if not values:
        return 0.0, 0.0
    rnd = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += values[rnd.randrange(n)]
        means.append(s / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def mcnemar(a_hits, b_hits):
    """Точный тест МакНемара для парных бинарных наблюдений."""
    b = sum(1 for x, y in zip(a_hits, b_hits) if x > 0.5 and y <= 0.5)
    c = sum(1 for x, y in zip(a_hits, b_hits) if x <= 0.5 and y > 0.5)
    n = b + c
    if n == 0:
        return b, c, 1.0
    # двусторонний точный биномиальный тест при p = 0.5
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def paired_bootstrap(a, b, iters=BOOTSTRAP, seed=SEED):
    """95% ДИ для разности средних (парный bootstrap)."""
    if not a or len(a) != len(b):
        return 0.0, 0.0, 0.0
    rnd = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            i = rnd.randrange(n)
            s += a[i] - b[i]
        diffs.append(s / n)
    return mean([x - y for x, y in zip(a, b)]), percentile(diffs, 0.025), percentile(diffs, 0.975)


def fmt_ci(v, lo, hi):
    return f"{v:.3f} [{lo:.3f}; {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="пути к report.json")
    ap.add_argument("--pair", action="append", default=[],
                    help="сравнить две конфигурации в формате A:B (можно несколько раз)")
    ap.add_argument("--ks", default="1,5,10")
    ap.add_argument("--lang", default=None, help="ограничить разбивку языком (en/ru)")
    ap.add_argument("--only-gold", default=None,
                    help="сравнивать только вопросы из указанного gold-файла "
                         "(позволяет сопоставлять прогоны на разных подвыборках)")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x]
    only_ids = load_ids(args.only_gold) if args.only_gold else None

    for path in args.reports:
        rep = load(path)
        print(f"\n## {path}")
        print(f"вопросов: {rep['gold_items']}, k: {rep['ks']}")
        if only_ids is not None:
            print(f"ограничение подвыборкой {args.only_gold}: {len(only_ids)} id")
        print()
        header = "| Конфигурация | n | " + " | ".join(
            [f"Hit@{k}" for k in ks] + [f"Recall@{k}" for k in ks] + [f"nDCG@{k}" for k in ks] + ["MRR"]
        ) + " |"
        print(header)
        print("|---" * (2 + len(ks) * 3 + 1) + "|")
        for cfg in rep["configs"]:
            its = answerable(cfg, only_ids)
            if args.lang:
                its = [it for it in its if it["lang"] == args.lang]
            cells = []
            for name, kk in [("hit", k) for k in ks] + [("recall", k) for k in ks] + \
                            [("ndcg", k) for k in ks] + [("mrr", None)]:
                vals = [metric(it, name, kk) for it in its]
                v = mean(vals)
                lo, hi = bootstrap_ci(vals)
                cells.append(fmt_ci(v, lo, hi))
            print(f"| {cfg['label']} | {len(its)} | " + " | ".join(cells) + " |")

        for spec in args.pair:
            a_label, b_label = spec.split(":", 1)
            by_label = {c["label"]: c for c in rep["configs"]}
            if a_label not in by_label or b_label not in by_label:
                print(f"\n(пропуск пары {spec}: нет такой конфигурации)")
                continue
            A = {it["id"]: it for it in answerable(by_label[a_label], only_ids)}
            B = {it["id"]: it for it in answerable(by_label[b_label], only_ids)}
            ids = sorted(set(A) & set(B))
            if args.lang:
                ids = [i for i in ids if A[i]["lang"] == args.lang]
            print(f"\n### Парное сравнение: {a_label} против {b_label} (n={len(ids)})")
            print("| Метрика | " + a_label + " | " + b_label + " | разность [95% ДИ] | p (МакНемар) |")
            print("|---|---:|---:|---|---:|")
            for name, kk in [("hit", k) for k in ks] + [("recall", k) for k in ks] + [("ndcg", 5), ("mrr", None)]:
                av = [metric(A[i], name, kk) for i in ids]
                bv = [metric(B[i], name, kk) for i in ids]
                d, lo, hi = paired_bootstrap(av, bv)
                label = "MRR" if name == "mrr" else f"{name.title()}@{kk}"
                if name in ("hit",):
                    _, _, p = mcnemar(av, bv)
                    p_s = f"{p:.4f}"
                else:
                    p_s = "—"
                print(f"| {label} | {mean(av):.3f} | {mean(bv):.3f} | "
                      f"{d:+.3f} [{lo:+.3f}; {hi:+.3f}] | {p_s} |")


if __name__ == "__main__":
    sys.exit(main())
