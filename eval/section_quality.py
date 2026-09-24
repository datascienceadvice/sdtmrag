#!/usr/bin/env python3
"""Оценка достоверности метаданных «раздел» в индексах.

Метаданные раздела используются при цитировании и в разбивке метрик, поэтому
важно, чтобы в них не попадали строки таблиц и оглавления. Скрипт считает долю
недостоверных значений раздела в каждом индексе — это одна из ключевых метрик
качества подготовки корпуса.

Признаки недостоверного значения (любой из трёх):
  * точечные выноски — строка оглавления (`1.1 PURPOSE ......... 5`);
  * два и более разрыва из двух и более пробелов — схлопнутая строка таблицы;
  * токен из трёх и более цифр — строка данных (`456 | CP | 0001`).

Пример:
    python eval/section_quality.py data/index_legacy.sqlite data/index_clean.sqlite
"""
import re
import sqlite3
import sys

DOT_LEADER = re.compile(r"\.{3,}")
MULTISPACE = re.compile(r"\s{2,}")
NUM_TOKEN = re.compile(r"^\d{3,}$")


def classify(section):
    """Возвращает причину недостоверности или None."""
    if not section:
        return "нет раздела"
    if DOT_LEADER.search(section):
        return "точечные выноски (оглавление)"
    if len(MULTISPACE.findall(section)) >= 2:
        return "несколько разрывов из 2+ пробелов (таблица)"
    if any(NUM_TOKEN.match(t) for t in section.split()):
        return "токен из 3+ цифр (строка данных)"
    return None


def main(stores):
    print("| Индекс | Чанков PDF с разделом | Уникальных значений | Недостоверных | Доля |")
    print("|---|---:|---:|---:|---:|")
    for store in stores:
        con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT section FROM chunks WHERE kind = 'pdf' AND section IS NOT NULL"
        ).fetchall()
        con.close()
        total = len(rows)
        bad = 0
        reasons = {}
        unique = set()
        for (section,) in rows:
            unique.add(section)
            reason = classify(section)
            if reason:
                bad += 1
                reasons[reason] = reasons.get(reason, 0) + 1
        share = 100.0 * bad / max(1, total)
        print(f"| `{store}` | {total} | {len(unique)} | {bad} | {share:.1f}% |")
        for reason, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"|   в том числе: {reason} | | | {n} | |")
    print()
    print("Примеры недостоверных значений (первые 5 на индекс):")
    for store in stores:
        con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT DISTINCT section FROM chunks WHERE kind = 'pdf' AND section IS NOT NULL"
        ).fetchall()
        con.close()
        bad = [s for (s,) in rows if classify(s)]
        print(f"  {store}:")
        for s in sorted(bad)[:5]:
            print(f"    {s[:110]!r}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/index_clean.sqlite"])
