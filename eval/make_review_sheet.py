#!/usr/bin/env python3
"""Готовит таблицу для экспертной проверки ответов системы.

Главное возражение рецензента к работе такого типа — «эталонный набор и оценка
сделаны авторами, без участия предметных специалистов». Скрипт превращает это в
конкретную задачу: выгружает вопросы, ответы системы и эталонные ответы в CSV,
который 2–3 специалиста по клинической лабораторной диагностике заполняют
независимо.

Пример:
    python eval/make_review_sheet.py \\
        eval/reports/judge/<дата>/report.json \\
        --gold eval/gold.en.jsonl --gold eval/gold.ru.jsonl \\
        --out eval/review_sheet.csv --limit 25
"""
import argparse
import csv
import json
import sys


def load_gold(paths):
    gold = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    gold[item["id"]] = item
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="report.json прогона с --judge")
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--out", default="eval/review_sheet.csv")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--every", type=int, default=2,
                    help="брать каждый N-й вопрос, чтобы покрыть все темы и языки")
    args = ap.parse_args()

    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)
    gold = load_gold(args.gold)

    if not report["configs"]:
        sys.exit("в отчёте нет конфигураций")
    cfg = report["configs"][0]
    items = [it for it in cfg["items"] if it.get("answer")]
    items = items[:: args.every][: args.limit]

    columns = [
        "id", "lang", "topic", "question", "system_answer",
        "reference_answer", "found_location",
        # заполняют эксперты
        "correctness_0_2", "completeness_0_2", "citation_ok_0_1",
        "would_use_in_practice_0_1", "comment",
    ]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for it in items:
            g = gold.get(it["id"], {})
            w.writerow({
                "id": it["id"],
                "lang": it["lang"],
                "topic": it.get("topic") or "",
                "question": it["question"],
                "system_answer": (it.get("answer") or "").replace("\n", " ").strip(),
                "reference_answer": (g.get("reference_answer") or "").strip(),
                "found_location": it.get("top1") or "",
                "correctness_0_2": "",
                "completeness_0_2": "",
                "citation_ok_0_1": "",
                "would_use_in_practice_0_1": "",
                "comment": "",
            })

    print(f"записано {len(items)} вопросов в {args.out}")
    print("инструкция для эксперта — в eval/examples/review_instructions.md")


if __name__ == "__main__":
    main()
