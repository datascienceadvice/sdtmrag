"""Baseline «без поиска»: та же генеративная модель, но без выдержек.

Отвечает на вопрос из собственных знаний и обязана сказать, если не знает.
Оцениваются только корректность и отказ: faithfulness и корректность ссылок
для ответа без выдержек не определены.

Запуск: python eval/closedbook.py [--workers 6]
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

GENERATE_SYSTEM = (
    "You are an expert on the CDISC SDTM standard and the SDTM Implementation Guide. "
    "Answer the question from your own knowledge. If you do not know the answer, say "
    "clearly that you do not know; do not guess variable names, domains or rules."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator of answers about the CDISC SDTM standard. "
    "You receive a QUESTION, a REFERENCE ANSWER (ground truth) and a SYSTEM ANSWER "
    "produced without access to any documentation. "
    "Score correctness in [0,1] as agreement with the reference answer. "
    "Set refused to true if the system answer states that the information is unknown "
    "or unavailable instead of answering. For questions that are NOT answerable from "
    "the CDISC documents, correctness = 1 when the system refuses and 0 when it invents "
    "an answer. Return JSON only."
)


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
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return json.loads(payload["choices"][0]["message"]["content"])
        except (error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
            last = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"api call failed: {last}")


def answer(cfg: dict, question: str) -> str:
    ds = cfg["deepseek"]
    key = ds.get("api_key")
    url = ds.get("base_url", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": ds.get("model", "deepseek-chat"),
            "messages": [
                {"role": "system", "content": GENERATE_SYSTEM},
                {"role": "user", "content": f"Question: {question}"},
            ],
            "temperature": 0.0,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    for attempt in range(3):
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        except (error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
            time.sleep(2 + attempt * 3)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold.en.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="eval/reports/closedbook")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    items = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[: args.limit]

    def one(it: dict) -> dict:
        text = answer(cfg, it["question"])
        if not it["answerable"]:
            user = (
                f"QUESTION:\n{it['question']}\n\n"
                "REFERENCE ANSWER:\n(none — this question cannot be answered from the CDISC documents)\n\n"
                f"SYSTEM ANSWER:\n{text}\n\nThis question is NOT answerable from the CDISC documents. "
                'Return {"correctness": 0.0, "refused": false, "rationale": "..."}'
            )
        else:
            user = (
                f"QUESTION:\n{it['question']}\n\n"
                f"REFERENCE ANSWER:\n{it['reference_answer'][:3000]}\n\n"
                f"SYSTEM ANSWER:\n{text}\n\n"
                'Return {"correctness": 0.0, "refused": false, "rationale": "..."}'
            )
        verdict = chat(cfg, JUDGE_SYSTEM, user)
        return {
            "id": it["id"],
            "register": it["register"],
            "answerable": it["answerable"],
            "answer": text,
            "correctness": float(verdict.get("correctness", 0.0)),
            "refused": bool(verdict.get("refused", False)),
            "rationale": verdict.get("rationale", ""),
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, items))

    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]
    summary = {
        "n": len(results),
        "correctness_answerable": sum(r["correctness"] for r in ans) / max(len(ans), 1),
        "refused_answerable": sum(1 for r in ans if r["refused"]) / max(len(ans), 1),
        "refused_unanswerable": sum(1 for r in unans if r["refused"]) / max(len(unans), 1),
        "correctness_unanswerable": sum(r["correctness"] for r in unans) / max(len(unans), 1),
        "seconds": round(time.time() - t0, 1),
    }

    stamp = time.strftime("%Y-%m-%d-%H-%M-%SZ", time.gmtime())
    out = Path(args.out) / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps({"summary": summary, "items": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"report: {out}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
