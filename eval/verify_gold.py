"""Независимая проверка эталонного набора.

Выполняется после сборки и проверяет три свойства, без которых метрики
недействительны:

1. каждый локатор срабатывает во **всех** сравниваемых индексах — иначе вопрос
   неотвечаем в одной из ветвей эксперимента по причине, не связанной с
   качеством поиска;
2. эталонный ответ **дословно** присутствует в тексте фрагмента, на который
   указывает локатор, — то есть ответ извлечён, а не написан автором по памяти;
3. вопросы вне области документации не имеют лексически подходящего фрагмента,
   который можно было бы принять за ответ.

Запуск: python eval/verify_gold.py [индекс ...]
"""
from __future__ import annotations

import glob
import json
import re
import sqlite3
import sys

MAP = {
    "\u00a0": " ", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-", "\u2026": "...", "\u00ad": "", "\ufeff": "",
    "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2022": "-", "\u00b7": "-", "\u25aa": "-", "\u25cf": "-", "\u25a0": "-",
    "\u25a1": "-", "\u25cb": "-",
}


def norm(s: str) -> str:
    s = "".join(MAP.get(c, c) for c in s)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_gold(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def chunk_matches(row, loc) -> bool:
    source, page, section, sheet, row_no, text = row
    if source.lower() != loc["source"].lower():
        return False
    if "sheet" in loc and sheet != loc["sheet"]:
        return False
    if "row" in loc and row_no != loc["row"]:
        return False
    if "page" in loc and page != loc["page"]:
        return False
    if "section" in loc and not (section and norm(loc["section"]) in norm(section)):
        return False
    if "contains" in loc and norm(loc["contains"]) not in norm(text):
        return False
    return True


def matches(rows, loc) -> bool:
    return any(chunk_matches(r, loc) for r in rows)


def find_chunk(rows, loc):
    for r in rows:
        if chunk_matches(r, loc):
            return r
    return None


def core_answer(reference: str) -> str:
    """Определение без метаданных строки рабочей книги."""
    for line in reference.split("\n"):
        line = line.strip()
        for key in ("CDISC Notes: ", "Definition: ", "Structure: "):
            if line.startswith(key):
                return line[len(key):].strip()
    return " ".join(reference.split())


def reference_parts(item: dict) -> list[str]:
    ref = item["reference_answer"]
    if len(item["expected"]) == 1:
        return [ref]
    parts = [p.strip() for p in ref.split("\n\n")]
    return [re.sub(r"^\[\d+\]\s*", "", p) for p in parts]


def main() -> int:
    stores = sys.argv[1:] or sorted(glob.glob("data/index*.sqlite"))
    gold = load_gold("eval/gold.en.jsonl")
    answerable = [it for it in gold if it["answerable"]]
    unanswerable = [it for it in gold if not it["answerable"]]
    print(f"gold items: {len(gold)} ({len(answerable)} answerable, {len(unanswerable)} out-of-scope)\n")

    problems = 0
    for store in stores:
        con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT source, page, section, sheet, row, text FROM chunks"
        ).fetchall()
        con.close()

        missing = []
        for it in answerable:
            for loc in it["expected"]:
                if not matches(rows, loc):
                    missing.append((it["id"], loc.get("contains") or loc))

        not_verbatim = []
        if store.endswith("index_clean.sqlite"):
            for it in answerable:
                parts = reference_parts(it)
                for k, loc in enumerate(it["expected"]):
                    if it.get("any_of") and k > 0:
                        # Альтернативный локатор указывает на тот же факт в другом
                        # документе. Требуем, чтобы его фраза была началом
                        # эталонного определения: значит, документ говорит то же
                        # самое, а не просто содержит похожие слова.
                        core = norm(core_answer(parts[0]))
                        if norm(loc["contains"]) not in core:
                            not_verbatim.append(f"{it['id']}#{k}")
                        continue
                    ref = parts[k] if k < len(parts) else parts[0]
                    found = find_chunk(rows, loc)
                    if found is None or norm(ref) not in norm(found[5]):
                        not_verbatim.append(f"{it['id']}#{k}")

        status = "OK" if not missing and not not_verbatim else f"{len(missing)} hidden, {len(not_verbatim)} non-verbatim"
        print(f"{store:38s} chunks={len(rows):6d}  {status}")
        for iid, loc in missing[:10]:
            print(f"    locator not found: {iid}: {str(loc)[:90]}")
        for iid in not_verbatim[:10]:
            print(f"    reference not verbatim in source: {iid}")
        problems += len(missing) + len(not_verbatim)

    # 3. вопросы вне области: ищем фрагменты, лексически близкие к вопросу
    con = sqlite3.connect("file:data/index_clean.sqlite?mode=ro", uri=True)
    rows = con.execute("SELECT source, page, section, sheet, row, text FROM chunks").fetchall()
    con.close()
    suspicious = []
    for it in unanswerable:
        words = [w for w in re.findall(r"[a-z]{5,}", it["question"].lower())]
        for source, page, section, sheet, row, text in rows:
            low = norm(text)
            hits = sum(1 for w in words if w in low)
            if hits >= 6:
                suspicious.append((it["id"], source, page, hits))
                break
    print()
    print(f"out-of-scope items with >=6 shared 5-letter terms: {len(suspicious)}/{len(unanswerable)}")
    for iid, source, page, hits in suspicious:
        print(f"    {iid}: {source} p.{page} ({hits} terms)")

    ok = problems == 0
    print()
    print("locator/verbatim check:", "PASS" if ok else f"FAIL ({problems} problems)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
