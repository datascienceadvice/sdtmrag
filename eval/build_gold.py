"""Сборка англоязычного эталонного набора из спецификации `gold_spec.py`.

Два принципа, которые этот скрипт обеспечивает механически:

1. Эталонный ответ — **дословный фрагмент** источника, а не пересказ автора.
   Он берётся из текста чанка по локатору, поэтому его нельзя «придумать».
2. Локатор обязан срабатывать **во всех сравниваемых индексах** (raw, clean,
   clean+tuned). Иначе вопрос был бы неотвечаем в одной из ветвей эксперимента
   по причине, не связанной с качеством поиска.

Локатор для PDF задаётся парой «страница + характерная фраза», а не заголовком
раздела: заголовки разделов сами являются объектом вмешательства (их точность
меняется от 82.7% недостоверных до 4.6%), поэтому опираться на них при
сравнении «до/после очистки» нельзя.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

from gold_spec import SPEC

CANON = "data/index_clean.sqlite"
INDEXES = [
    "data/index_raw.sqlite",
    "data/index_clean.sqlite",
    "data/index_clean_tuned.sqlite",
]
OUT = "eval/gold.en.jsonl"

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


class Chunk:
    __slots__ = ("source", "kind", "page", "section", "sheet", "row", "text", "body")

    def __init__(self, source, kind, page, section, sheet, row, text):
        self.source = source
        self.kind = kind
        self.page = page
        self.section = section
        self.sheet = sheet
        self.row = row
        self.text = text
        self.body = strip_header(text)


def strip_header(text: str) -> str:
    """Убирает служебную первую строку чанка ([Section]/[Sheet])."""
    lines = text.split("\n")
    if lines and (lines[0].startswith("[Section]") or lines[0].startswith("[Sheet]")):
        return "\n".join(lines[1:])
    return text


def load(path: str) -> list[Chunk]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT source, kind, page, section, sheet, row, text FROM chunks"
    ).fetchall()
    con.close()
    return [Chunk(*r) for r in rows]


def find_xlsx(chunks: list[Chunk], source: str, sheet: str, row: int) -> Chunk:
    for c in chunks:
        if c.source == source and c.sheet == sheet and c.row == row:
            return c
    raise LookupError(f"no chunk for {source} / {sheet} / row {row}")


def find_section(chunks: list[Chunk], source: str, section: str) -> list[Chunk]:
    want = norm(section)
    out = [c for c in chunks if c.source == source and c.section and want in norm(c.section)]
    if not out:
        near = sorted({c.section for c in chunks if c.source == source and c.section})
        raise LookupError(
            f"no chunk with section containing {section!r} in {source}; "
            f"candidates:\n    " + "\n    ".join(near[:60])
        )
    return out


TABLEISH = re.compile(r"\s{3,}")


def body_lines(chunk: Chunk) -> list[tuple[str, bool]]:
    """Строки тела чанка и признак «строка начинает предложение»."""
    out: list[tuple[str, bool]] = []
    prev = ""
    for k, raw in enumerate(chunk.body.split("\n")):
        s = raw.strip()
        clean = k == 0 or prev == "" or prev.rstrip().endswith((".", "!", "?", ":", ";"))
        out.append((s, clean))
        prev = s
    return out


def candidate_lines(chunk: Chunk) -> list[tuple[int, str, bool]]:
    strict: list[tuple[int, str, bool]] = []
    relaxed: list[tuple[int, str, bool]] = []
    for k, (s, clean) in enumerate(body_lines(chunk)):
        if len(s) < 45 or "|" in s:
            continue
        relaxed.append((k, s, clean))
        if re.match(r"^[A-Z0-9]", s):
            strict.append((k, s, clean))
    return strict or relaxed


def score_line(line: str, clean: bool, kws: list[str]) -> int:
    low = line.lower()
    hits = sum(1 for k in kws if k.lower() in low)
    prose = 0
    if line.rstrip().endswith((".", ":", ";")):
        prose += 2
    if len(line) >= 100:
        prose += 1
    if re.search(r"\b(the|is|are|should|must|may|will)\b", low):
        prose += 1
    # граница предложения слева важнее длины: обрывок с середины фразы
    # нельзя использовать как эталонный ответ
    return hits * 3 + prose + (4 if clean else 0)


def anchors(excerpt: str) -> list[str]:
    """Фразы-якоря, от длинной к короткой.

    Длинная фраза точнее указывает на фрагмент, короткая выживает, даже если в
    «сыром» индексе внутрь предложения вклеен колонтитул. Порядок именно такой:
    resolve() берёт первую *уникальную* по корпусу фразу, иначе «A findings
    domain that contains data related» указывал бы сразу на пять доменов.
    """
    words = excerpt.split()
    out: list[str] = []
    for target in (140, 110, 90, 70, 55, 40):
        acc: list[str] = []
        for w in words:
            acc.append(w)
            if len(" ".join(acc)) >= target:
                break
        s = " ".join(acc)
        if len(s) >= 40 and s not in out:
            out.append(s)
    return out


def extend_to_sentence(body: str, pos: int) -> str:
    """Достраивает фрагмент по *строкам тела* вокруг строки pos: вперёд до
    конца предложения и назад, пока начало не станет границей предложения.
    В PDF фраза часто начинается на предыдущей строке и обрывается на полуслове."""
    raw = body.split("\n")
    excerpt = raw[pos]
    j = pos
    while (
        j + 1 < len(raw)
        and len(excerpt) < 420
        and not excerpt.rstrip().endswith((".", "!", "?"))
        and "|" not in raw[j + 1]
    ):
        excerpt = excerpt + "\n" + raw[j + 1]
        j += 1
    while (
        pos - 1 >= 0
        and len(excerpt) < 420
        and not re.match(r"^[A-Z0-9]", excerpt)
        and "|" not in raw[pos - 1]
        and raw[pos - 1].strip()
    ):
        excerpt = raw[pos - 1] + "\n" + excerpt
        pos -= 1
    return excerpt.strip()


def excerpt_candidates(chunk: Chunk, kws: list[str], phrase: str | None = None):
    """Перебирает (эталонный фрагмент, фраза-якорь) от лучшего к худшему.

    Если задана phrase (локатор `pin`), первым идёт строка, в которой эта фраза
    действительно находится: иначе отбор по ключевым словам может взять
    соседнюю строку того же чанка и эталон перестанет отвечать на вопрос.
    """
    lines = candidate_lines(chunk)
    if not lines:
        return
    ranked = sorted(
        ((score_line(s, clean, kws), len(s), k) for k, s, clean in lines),
        key=lambda t: (-t[0], -t[1]),
    )
    order: list[int] = []
    if phrase:
        want = [w for w in norm(phrase).split() if len(w) > 3]
        best_k, best_hits = None, 0
        for k, s, _ in lines:
            have = set(norm(s).split())
            hits = sum(1 for w in want if w in have)
            if hits > best_hits:
                best_k, best_hits = k, hits
        if best_k is not None and best_hits >= 3:
            order.append(best_k)
    for _, _, k in ranked:
        if k not in order:
            order.append(k)

    seen: set[str] = set()
    tried = 0
    for pos in order:
        if tried >= 12:
            break
        tried += 1
        excerpt = extend_to_sentence(chunk.body, pos)
        if excerpt not in chunk.body or excerpt in seen:
            continue
        if not re.match(r"^[A-Z0-9]", excerpt):
            continue
        seen.add(excerpt)
        yield excerpt, anchors(excerpt)


def matches_locator(chunks: list[Chunk], loc: dict) -> bool:
    want_src = loc["source"]
    want_page = loc.get("page")
    want_sheet = loc.get("sheet")
    want_row = loc.get("row")
    want_sec = norm(loc["section"]) if loc.get("section") else None
    want_contains = norm(loc["contains"]) if loc.get("contains") else None
    for c in chunks:
        if not c.source.lower() == want_src.lower():
            continue
        if want_page is not None and c.page != want_page:
            continue
        if want_sheet is not None and c.sheet != want_sheet:
            continue
        if want_row is not None and c.row != want_row:
            continue
        if want_sec is not None and not (c.section and want_sec in norm(c.section)):
            continue
        if want_contains is not None and want_contains not in norm(c.text):
            continue
        return True
    return False


def parse_locator(spec: str) -> dict:
    parts = [p.strip() for p in spec.split("|")]
    kind = parts[0]
    if kind == "xlsx":
        return {"source": parts[1], "sheet": parts[2], "row": int(parts[3])}
    if kind in ("sec", "page"):
        return {
            "kind": kind,
            "source": parts[1],
            "target": parts[2],
            "keywords": [k.strip() for k in parts[3].split(",") if k.strip()] if len(parts) > 3 else [],
        }
    if kind == "pin":
        return {"kind": "pin", "source": parts[1], "target": parts[2]}
    raise ValueError(f"bad locator spec: {spec}")


_CANON_CACHE: list[str] | None = None


def _canon_norm(canon: list[Chunk]) -> list[str]:
    """Нормированный корпус считается один раз: проверка уникальности якоря
    иначе стоит O(n^2) и превращает сборку в многочасовую."""
    global _CANON_CACHE
    if _CANON_CACHE is None:
        _CANON_CACHE = [norm(c.text) for c in canon]
    return _CANON_CACHE


def resolve(spec: str, canon: list[Chunk], indexes: list[list[Chunk]]) -> tuple[dict, str, int | None]:
    """Возвращает (локатор для gold-файла, дословный эталонный фрагмент)."""
    loc = parse_locator(spec)
    if "sheet" in loc:
        c = find_xlsx(canon, loc["source"], loc["sheet"], loc["row"])
        for idx in indexes:
            find_xlsx(idx, loc["source"], loc["sheet"], loc["row"])
        return loc, c.body.strip(), None

    kws = loc["keywords"] if "keywords" in loc else []
    if loc["kind"] == "sec":
        candidates = find_section(canon, loc["source"], loc["target"])
    elif loc["kind"] == "pin":
        phrase = loc["target"]
        candidates = [
            c for c in canon if c.source == loc["source"] and norm(phrase) in norm(c.text)
        ][:4]
        if not candidates:
            raise LookupError(f"phrase not found in {loc['source']}: {phrase!r}")
        kws = [w for w in phrase.split() if len(w) > 4][:8]
    else:
        page = int(loc["target"])
        candidates = [c for c in canon if c.source == loc["source"] and c.page == page]
        if not candidates:
            raise LookupError(f"no chunks on page {page} of {loc['source']}")

    prefer = loc.get("target") if loc["kind"] == "pin" else None
    canon_norm = _canon_norm(canon)
    last_err = None
    for chunk in candidates:
        for excerpt, anchor_list in excerpt_candidates(chunk, kws, prefer):
            valid = []
            for a in anchor_list:
                locator = {"source": loc["source"], "contains": a}
                if not all(matches_locator(idx, locator) for idx in indexes):
                    continue
                n = sum(1 for t in canon_norm if norm(a) in t)
                valid.append((n, a, locator))
            if not valid:
                last_err = f"no anchor survives in every index for {excerpt[:60]!r}"
                continue
            # предпочитаем уникальную по корпусу фразу, но только среди якорей
            # *этого* фрагмента: иначе выбор уезжает на соседнюю строку
            valid.sort(key=lambda t: t[0])
            _, _, locator = valid[0]
            return locator, excerpt, chunk.page
    raise LookupError(f"cannot build a treatment-invariant locator for {spec}: {last_err}")



def core_answer(reference: str) -> str:
    """Самое информативное предложение ответа — определение, а не метаданные строки."""
    for line in reference.split("\n"):
        line = line.strip()
        for key in ("CDISC Notes: ", "Definition: ", "Structure: "):
            if line.startswith(key):
                return line[len(key):].strip()
    return " ".join(reference.split())


def alias_locator(
    core: str,
    exclude: set[str],
    canon: list[Chunk],
    indexes: list[list[Chunk]],
) -> dict | None:
    """Локатор на тот же факт в другом документе.

    Определение переменной из рабочей книги почти всегда продублировано в
    таблице спецификации внутри PDF. Если засчитывать только строку рабочей
    книги, конфигурация, которая лучше находит прозаическую версию, штрафуется
    за то, что нашла правильный ответ.
    """
    probe = core[:120]
    if len(probe) < 40:
        return None
    others = [
        c for c in canon
        if c.source not in exclude and norm(probe) in norm(c.text)
    ]
    if not others:
        return None
    src = others[0].source
    words = core.split()
    for target in (120, 90, 70, 55, 40):
        acc: list[str] = []
        for w in words:
            acc.append(w)
            if len(" ".join(acc)) >= target:
                break
        anchor = " ".join(acc)
        if len(anchor) < 40:
            continue
        loc = {"source": src, "contains": anchor}
        if all(matches_locator(idx, loc) for idx in indexes):
            return loc
    return None

def main() -> int:
    canon = load(CANON)
    indexes = [load(p) for p in INDEXES]

    items: list[dict] = []
    errors: list[str] = []
    for entry in SPEC:
        iid, register, topic, question, locators = entry[0], entry[1], entry[2], entry[3], entry[4]
        expected: list[dict] = []
        answers: list[str] = []
        pages: list[int] = []
        ok = True
        for spec in locators:
            try:
                loc, answer, page = resolve(spec, canon, indexes)
            except (LookupError, ValueError) as exc:
                errors.append(f"{iid}: {exc}")
                ok = False
                break
            expected.append(loc)
            answers.append(answer)
            if page is not None:
                pages.append(page)
        if not ok:
            continue
        if len(answers) == 1:
            reference = answers[0]
        else:
            reference = "\n\n".join(
                f"[{i + 1}] {a}" for i, a in enumerate(answers)
            )
        strata = {"xlsx": 0, "pdf": 0}
        for loc in expected:
            strata["xlsx" if "sheet" in loc else "pdf"] += 1
        if strata["xlsx"] and strata["pdf"]:
            stratum = "cross"
        elif strata["pdf"]:
            stratum = "prose"
        else:
            stratum = "tabular"

        # Альтернативный локатор добавляется ПОСЛЕ вычисления страты: страта
        # описывает, где лежит ответ, и не должна меняться от того, что тот же
        # факт продублирован в другом документе.
        any_of = False
        if len(expected) == 1 and expected[0]:
            alias = alias_locator(
                core_answer(reference),
                {loc["source"] for loc in expected},
                canon,
                indexes,
            )
            if alias is not None:
                expected.append(alias)
                any_of = True

        answerable = bool(expected)
        items.append(
            {
                "id": iid,
                "question": question,
                "lang": "en",
                "topic": topic,
                "register": register,
                "stratum": "out-of-scope" if not answerable else stratum,
                "answerable": answerable,
                "expected": expected,
                "any_of": any_of,
                "reference_answer": reference,
                "pages": sorted(set(pages)),
                "provenance": "index-extracted",
                "needs_review": True,
                "notes": "; ".join(
                    f"{i}: {loc.get('contains', '')[:60]}" for i, loc in enumerate(expected)
                )[:300],
            }
        )

    if errors:
        print(f"BUILD FAILED: {len(errors)} unresolved item(s)\n", file=sys.stderr)
        for e in errors:
            print(" -", e, file=sys.stderr)
        return 1

    Path(OUT).write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n",
        encoding="utf-8",
    )

    from collections import Counter

    by_reg = Counter(it["register"] for it in items)
    by_str = Counter(it["stratum"] for it in items)
    print(f"wrote {len(items)} items to {OUT}")
    print("by register:", dict(sorted(by_reg.items())))
    print("by stratum: ", dict(sorted(by_str.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
