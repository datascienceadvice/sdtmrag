"""Проверка, что каждый локатор «золотого» набора находится в каждом индексе.

Локатор, не совпавший ни с одним чанком, делает вопрос неотвечаемым в этом
индексе и несправедливо занижает метрики. Проверка выполняется для всех
сравниваемых индексов, а не только для базового.
"""
import json, re, sqlite3, sys, glob

MAP = {
    "\u00a0": " ", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-", "\u2026": "...", "\u00ad": "", "\ufeff": "",
    "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff",
}


def norm(s):
    s = "".join(MAP.get(c, c) for c in s)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_gold(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def matches(rows, loc):
    for r in rows:
        source, kind, page, section, sheet, row, text = r
        if source != loc["source"]:
            continue
        if "sheet" in loc and sheet != loc["sheet"]:
            continue
        if "row" in loc and row != loc["row"]:
            continue
        if "page" in loc and page != loc["page"]:
            continue
        if "section" in loc:
            if not section or norm(loc["section"]) not in norm(section):
                continue
        if "contains" in loc and norm(loc["contains"]) not in norm(text):
            continue
        return True
    return False


stores = sys.argv[1:] or sorted(glob.glob("data/index*.sqlite"))
gold = load_gold("eval/gold.en.jsonl") + load_gold("eval/gold.ru.jsonl")
answerable = [it for it in gold if it["answerable"]]
print(f"gold items: {len(gold)} ({len(answerable)} answerable)\n")

bad_total = 0
for store in stores:
    con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT source, kind, page, section, sheet, row, text FROM chunks"
    ).fetchall()
    n_chunks = len(rows)
    con.close()
    missing = []
    for it in answerable:
        for loc in it["expected"]:
            if not matches(rows, loc):
                missing.append((it["id"], loc))
    status = "OK" if not missing else f"{len(missing)} MISSING"
    print(f"{store:36s} chunks={n_chunks:6d}  {status}")
    for iid, loc in missing[:20]:
        print(f"    {iid}: {loc}")
    bad_total += len(missing)

print()
print("locator check:", "PASS" if bad_total == 0 else f"FAIL ({bad_total} missing)")
sys.exit(0 if bad_total == 0 else 1)
