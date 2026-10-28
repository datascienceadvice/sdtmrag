"""Контроль качества эталонного набора судьёй.

Проверяет ровно то, что обычно проверяет эксперт: содержит ли эталонный
фрагмент ответ на поставленный вопрос. Результат — не замена экспертизе, а её
количественная характеристика: доля вопросов, для которых эталон признан
достаточным, публикуется вместе с набором.

Второй режим (--out-of-scope) проверяет обратное: что вопросы, помеченные как
неотвечаемые, действительно не могут быть отвечены по корпусу CDISC.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error, request


def load_config(path: str = "config.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def chat(cfg: dict, system: str, user: str, retries: int = 3) -> dict:
    ds = cfg["deepseek"]
    key = ds.get("api_key")
    if not key:
        raise SystemExit("no deepseek.api_key in config.toml")
    url = ds.get("base_url", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": ds.get("model", "deepseek-chat"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    last = None
    for attempt in range(retries):
        req = request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            return json.loads(content)
        except (error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
            last = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"judge call failed: {last}")


SYSTEM = (
    "You audit a retrieval benchmark for laboratory data standards. "
    "You receive a QUESTION and a REFERENCE PASSAGE that was extracted verbatim "
    "from an official CDISC document (SDTM or SDTMIG). Decide only one thing: "
    "does the passage contain enough information to answer the question fully "
    "and correctly? Be strict: a passage that merely mentions the topic, or that "
    "answers a different question, is not sufficient. "
    'Reply with JSON: {"sufficient": true|false, "why": "<up to 20 words>"}'
)

SYSTEM_UNANS = (
    "You audit a retrieval benchmark for laboratory data standards. You receive a "
    "QUESTION and you must decide whether it can be answered from the CDISC SDTM "
    "and SDTMIG documents (Study Data Tabulation Model and its Implementation "
    "Guide), which describe how to structure and submit clinical trial data. "
    "Questions about clinical practice, other standards (CLSI, ISO, HL7, IVDR) or "
    "analysis datasets cannot be answered from those documents. "
    'Reply with JSON: {"answerable_from_cdisc": true|false, "why": "<up to 20 words>"}'
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold.en.jsonl")
    ap.add_argument("--out-of-scope", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="eval/reports/gold_qc")
    args = ap.parse_args()

    cfg = load_config()
    items = [json.loads(line) for line in open(args.gold, encoding="utf-8") if line.strip()]
    if args.out_of_scope:
        items = [it for it in items if not it["answerable"]]
    else:
        items = [it for it in items if it["answerable"]]
    if args.limit:
        items = items[: args.limit]

    def one(it: dict) -> dict:
        if args.out_of_scope:
            verdict = chat(cfg, SYSTEM_UNANS, f"QUESTION: {it['question']}")
            ok = not verdict.get("answerable_from_cdisc", True)
            why = verdict.get("why", "")
        else:
            user = (
                f"QUESTION: {it['question']}\n\n"
                f"REFERENCE PASSAGE:\n{it['reference_answer'][:4000]}"
            )
            verdict = chat(cfg, SYSTEM, user)
            ok = bool(verdict.get("sufficient", False))
            why = verdict.get("why", "")
        return {"id": it["id"], "ok": ok, "why": why}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, items))

    passed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    rate = len(passed) / len(results) if results else 0.0

    stamp = time.strftime("%Y-%m-%d-%H-%M-%SZ", time.gmtime())
    out_dir = Path(args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "mode": "out-of-scope" if args.out_of_scope else "sufficiency",
                "items": len(results),
                "passed": len(passed),
                "rate": rate,
                "seconds": round(time.time() - t0, 1),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    label = "unanswerable questions refused" if args.out_of_scope else "reference passage sufficient"
    print(f"{label}: {len(passed)}/{len(results)} = {rate:.3f}")
    print(f"elapsed {time.time() - t0:.0f}s, report: {out_dir}/report.json")
    if failed:
        print("\nflagged:")
        for r in failed:
            print(f"  {r['id']}: {r['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
