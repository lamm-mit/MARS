#!/usr/bin/env python3
"""Judge a MARS run from its three subsystem summaries with the detailed rubric.

Same 12 metrics, same 1-5 scale, same weighted-total math and the same output schema as
scripts/run_evaluation.py, so downstream tables and notebooks keep working. What changes
is the judge's INPUT: instead of mars.json (final export, no retrieval trail) the judge
reads the System 1/2/3 summary documents produced by build_subsystem_summaries.py, in
which every item carries an [EVIDENCE BASIS] line with per-database document counts and
a provenance trace of its numbers.

Usage (single run):
    python evaluation_refined/run_refined_evaluation.py \
        --summaries evaluation_refined/seed_101/summaries --label seed_101 \
        --out evaluation_refined/seed_101/evaluation --model gpt-5.6-sol

Several --summaries/--label pairs are judged in ONE blind call with A/B/... labels
(random.seed 42, as the original protocol). One pair gets label A.

The API key is read from OPENAI_API_KEY, else from --api-key-file (KEY=value lines).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import yaml  # noqa: E402
from openai import OpenAI  # noqa: E402

import run_evaluation as rev  # noqa: E402  (original protocol helpers: parsing, clamping, headers)

DEFAULT_RUBRIC = Path(__file__).resolve().parent / "rubric_detailed.yaml"
SUMMARY_FILES = ["system1_summary.md", "system2_summary.md", "system3_summary.md"]


def load_rubric(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        r = yaml.safe_load(f)
    if not isinstance(r, dict) or "dimensions" not in r:
        raise ValueError(f"invalid rubric: {path}")
    return r


def load_summaries(d: Path) -> Dict[str, str]:
    out = {}
    for fn in SUMMARY_FILES:
        p = d / fn
        if not p.exists():
            raise FileNotFoundError(p)
        out[fn] = p.read_text(encoding="utf-8")
    return out


def query_sentence_from_summary(s1: str) -> str:
    for ln in s1.splitlines():
        if ln.startswith("Sentence: "):
            return ln[len("Sentence: "):].strip()
    return ""


def build_prompt(query_sentence: str, systems: Dict[str, Dict[str, str]], blind: Dict[str, str], rubric: Dict[str, Any]) -> Tuple[str, str]:
    dims = rubric["dimensions"]
    labels = sorted(blind.keys())
    n = len(labels)
    conv = rubric.get("evidence_conventions") or []
    conv_block = "\n".join(f"- {c}" for c in conv)

    system_prompt = (
        f"You are an expert materials scientist acting as a blind evaluator. "
        f"You will receive outputs from {n} anonymized system{'s' if n > 1 else ''} (labeled {', '.join(labels)}) "
        "that attempted to solve the same material substitution task. Each system is presented as three "
        "summary documents (System 1: requirements, System 2: candidate discovery, System 3: manufacturability) "
        "built from the run's raw artifacts. "
        "You must evaluate each system on every subsystem criterion in the rubric "
        "(Systems 1–3 intermediate outputs), using integer scores 1–5 only.\n\n"
        "IMPORTANT:\n"
        "- Evaluate each system independently on its merits.\n"
        "- Do NOT try to guess which system is which.\n"
        "- Be critical and specific in your reasoning.\n"
        "- Use your materials science expertise to assess technical correctness.\n"
        "- Fabricated or hallucinated materials should receive low scores on System 2 criteria "
        "(especially Realism and Reasoning quality), even if the text sounds plausible.\n\n"
        "HOW TO READ THE SUMMARY DOCUMENTS:\n" + conv_block + "\n\n"
        "You MUST respond with a valid JSON object and nothing else."
    )

    scale_block = "\n".join(f"- {line}" for line in (rubric.get("ordinal_scale_lines") or []))
    rubric_text = f"### Shared ordinal scale (1–5)\n{scale_block}\n" if scale_block else ""
    for dk, dim in dims.items():
        rubric_text += f"\n### {dim['name']} (weight: {dim['weight']})\n{dim['rubric']}\n"

    blocks = ""
    for lab in labels:
        key = blind[lab]
        blocks += f"\n{'=' * 60}\nSYSTEM {lab}\n{'=' * 60}\n"
        for fn in SUMMARY_FILES:
            blocks += f"\n{'-' * 60}\nSYSTEM {lab} — {fn.replace('_summary.md', '').upper()} SUMMARY DOCUMENT\n{'-' * 60}\n"
            blocks += systems[key][fn] + "\n"

    dim_keys = list(dims.keys())
    ranking_placeholders = ", ".join(f'"<#{i + 1} label>"' for i in range(n))
    schema = "{\n"
    for lab in labels:
        schema += f'  "{lab}": {{\n'
        for dk in dim_keys:
            schema += f'    "{dk}": {{"score": <int 1-5>, "reasoning": "<2-5 sentences citing the counts/items relied on>"}},\n'
        schema += '    "overall_comment": "<1-2 sentences>"\n'
        schema += "  },\n"
    schema += f'  "ranking": [{ranking_placeholders}],\n'
    schema += '  "ranking_reasoning": "<1-3 sentences explaining the ranking>"\n'
    schema += "}"

    user_prompt = (
        f"## Material Substitution Query\n\n{query_sentence}\n\n"
        f"## Evaluation Rubric\n{rubric_text}\n\n"
        f"## System Outputs (Anonymized; three summary documents per system)\n{blocks}\n\n"
        f"## Required Output Format\n\nRespond with ONLY a JSON object in this exact structure:\n\n"
        f"```\n{schema}\n```\n\n"
        "Provide integer scores (1–5) for each subsystem criterion and reasoning for each that cites the "
        "specific counts or items from the summaries you relied on. Then provide an overall ranking from best to worst."
    )
    return system_prompt, user_prompt


def _extract_json(text: str) -> Dict[str, Any]:
    if "```json" in text:
        text = text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].rsplit("```", 1)[0].strip()
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        text = text[a : b + 1]
    return json.loads(text)


def call_judge(client: OpenAI, model: str, system_prompt: str, user_prompt: str, rubric: Dict[str, Any], max_retries: int = 3) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Returns (parsed_json, api_meta, protocol_deviations)."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    temperature = rubric.get("temperature", 0)
    max_tokens = int(rubric.get("max_tokens", 8000))
    deviations: List[str] = []
    kw: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if rubric.get("reasoning_effort"):
        kw["reasoning_effort"] = rubric["reasoning_effort"]
    last_err = ""
    raw_text = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kw)
        except Exception as e:  # parameter fallbacks, recorded as deviations
            msg = str(e)
            low = msg.lower()
            if "temperature" in low and "temperature" in kw:
                kw.pop("temperature")
                deviations.append(f"temperature omitted: model rejected temperature={temperature} ({msg[:120]})")
                continue
            if rev._should_retry_completion_with_max_completion_tokens(e) and "max_tokens" in kw:
                kw["max_completion_tokens"] = kw.pop("max_tokens")
                deviations.append("max_tokens renamed to max_completion_tokens (model requirement)")
                continue
            last_err = msg
            print(f"  attempt {attempt + 1}/{max_retries}: API error: {msg[:200]}")
            time.sleep(2 ** attempt)
            continue
        choice = resp.choices[0]
        raw_text = (choice.message.content or "").strip()
        meta: Dict[str, Any] = {"finish_reason": getattr(choice, "finish_reason", None)}
        if getattr(choice.message, "refusal", None):
            meta["refusal"] = choice.message.refusal
        u = getattr(resp, "usage", None)
        if u is not None:
            meta["usage"] = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens}
        meta["request_params"] = {k: v for k, v in kw.items() if k != "messages"}
        try:
            parsed = _extract_json(raw_text)
        except json.JSONDecodeError as e:
            last_err = f"JSON parse error: {e}"
            print(f"  attempt {attempt + 1}/{max_retries}: {last_err}")
            meta["raw_text"] = raw_text
            time.sleep(2 ** attempt)
            continue
        rev._clamp_parsed_scores(parsed, rubric)
        meta["raw_text"] = raw_text
        return parsed, meta, deviations
    return {"error": last_err or "judge failed", "raw": raw_text}, {}, deviations


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summaries", action="append", required=True, help="directory with system{1,2,3}_summary.md (repeatable)")
    ap.add_argument("--label", action="append", required=True, help="run label for each --summaries, in order (repeatable)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--rubric", default=str(DEFAULT_RUBRIC))
    ap.add_argument("--model", default=None, help="judge model (default: rubric judge_model)")
    ap.add_argument("--seed", type=int, default=42, help="blind-label shuffle seed (original protocol: 42)")
    ap.add_argument("--api-key-file", default=os.path.expanduser("~/Github/api_key.env"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--study", default="Refined evaluation cycle: subsystem summaries with evidence basis")
    args = ap.parse_args()
    if len(args.summaries) != len(args.label):
        sys.exit("--summaries and --label must be given the same number of times")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and os.path.exists(args.api_key_file):
        for line in open(args.api_key_file):
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set and no key in --api-key-file")

    rubric = load_rubric(Path(args.rubric))
    model = args.model or rubric.get("judge_model", "gpt-4.1")
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    systems: Dict[str, Dict[str, str]] = {}
    for d, lab in zip(args.summaries, args.label):
        p = Path(d)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        systems[lab] = load_summaries(p)
    keys = list(systems.keys())
    query_sentence = query_sentence_from_summary(systems[keys[0]]["system1_summary.md"])

    random.seed(args.seed)
    shuffled = keys[:]
    random.shuffle(shuffled)
    blind = {chr(ord("A") + i): k for i, k in enumerate(shuffled)}
    reverse = {v: k for k, v in blind.items()}
    print("Blind mapping (hidden from judge):", json.dumps(blind))

    system_prompt, user_prompt = build_prompt(query_sentence, systems, blind, rubric)
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M")
    (out / f"judge_prompt_{ts}.txt").write_text("### SYSTEM PROMPT\n" + system_prompt + "\n\n### USER PROMPT\n" + user_prompt, encoding="utf-8")
    print(f"Prompt size: {len(user_prompt):,} chars (system prompt {len(system_prompt):,}); judge model {model}")

    start = time.time()
    parsed, meta, deviations = call_judge(client, model, system_prompt, user_prompt, rubric)
    elapsed = time.time() - start
    print(f"Judge responded in {elapsed:.1f}s; finish_reason={meta.get('finish_reason')}; usage={meta.get('usage')}")
    if deviations:
        print("Protocol deviations:", "; ".join(deviations))
    if "error" in parsed:
        (out / f"judge_raw_{ts}.txt").write_text(str(parsed.get("raw", "")), encoding="utf-8")
        sys.exit(f"ERROR: {parsed['error']}")
    (out / f"judge_raw_{ts}.json").write_text(json.dumps({"response": meta.pop("raw_text", ""), "meta": meta}, indent=2, ensure_ascii=False), encoding="utf-8")

    dims = list(rubric["dimensions"].keys())
    wsum = sum(rubric["dimensions"][d]["weight"] for d in dims)
    scores: Dict[str, Any] = {}
    for k in keys:
        ld = parsed.get(reverse[k], {}) or {}
        s = {d: {"score": (ld.get(d) or {}).get("score", 0), "reasoning": (ld.get(d) or {}).get("reasoning", "")} for d in dims}
        wtot = sum(s[d]["score"] * rubric["dimensions"][d]["weight"] for d in dims)
        s["weighted_total"] = round(wtot / wsum, 2) if wsum else 0
        s["overall_comment"] = ld.get("overall_comment", "")
        scores[k] = s
    ranking = [blind.get(r, r) for r in parsed.get("ranking", [])]

    result = {
        "study": args.study,
        "query_name": "Query1",
        "query_sentence": query_sentence,
        "input_format": "subsystem summary documents with [EVIDENCE BASIS] lines (build_subsystem_summaries.py)",
        "summaries": {k: str(Path(d)) for k, d in zip(args.label, args.summaries)},
        "rubric_path": str(Path(args.rubric)),
        "blind_mapping": blind,
        "scores": scores,
        "ranking": ranking,
        "ranking_reasoning": parsed.get("ranking_reasoning", ""),
        "judge_model": model,
        "protocol_deviation": "; ".join(deviations) if deviations else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "judge_elapsed_seconds": round(elapsed, 1),
        "judge_completion": meta,
    }
    outfile = out / f"eval_{'_'.join(keys)}_{ts}.json"
    outfile.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("saved:", outfile)

    hdr = [rev.rubric_column_header(rubric["dimensions"][d]) for d in dims]
    print(f"\n{'run':<12} " + "".join(f"{h:<10}" for h in hdr) + f"{'Weighted':>9}")
    for k in keys:
        s = scores[k]
        print(f"{k:<12} " + "".join(f"{s[d]['score']:<10}" for d in dims) + f"{s['weighted_total']:>9.2f}")
    for k in keys:
        for pfx, name in (("system1", "S1"), ("system2", "S2"), ("system3", "S3")):
            xs = [scores[k][d]["score"] for d in dims if d.startswith(pfx)]
            print(f"{k} {name} mean: {sum(xs) / len(xs):.2f}", end="  ")
        print()
    if len(keys) > 1:
        print("Ranking:", " > ".join(ranking))


if __name__ == "__main__":
    main()
