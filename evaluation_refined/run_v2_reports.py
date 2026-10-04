#!/usr/bin/env python
"""Build System 1/2/3 reports for one MARS run and judge them with rubric_s{1,2,3}_final.yaml.

Reuses the report builders and evaluator system prompts defined in
evaluate_summaries.ipynb (read-only: the notebook cells are loaded and only
their imports, constants and function definitions are executed), so the
reports have exactly the format used for the seed 101 trial in
evaluation_trial/trial1. The judge prompt is assembled from the rubric YAML on
disk: every top-level text section (system role, principles, ...) is rendered
in file order, so a renamed or added section is never silently dropped.

Usage:
  conda activate MARS
  python evaluation_refined/run_v2_reports.py \
      --run-dir results_seedstudy_gemma4/seed_202/Query1 \
      --label gemma4_seed_202 --out-dir evaluation_trial/gemma4_seed_202
  Add --systems 1 to build/judge only System 1; --no-judge to skip the API.
"""
import argparse
import ast
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
# Default to the dummy material database shipped with the repository; export
# MARS_MATERIAL_DB to point the System 2 report at another database.
os.environ.setdefault("MARS_MATERIAL_DB", str(
    PROJECT_ROOT / "data/MARS_Data/MaterialDB_paper/internal_material_database.json"))
for p in (HERE, PROJECT_ROOT / "scripts", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import run_refined_evaluation as rre  # noqa: E402

NOTEBOOK = HERE / "evaluate_summaries.ipynb"
# cells that define the builders/renderers, and the evaluator-prompt cell
BUILDER_CELLS = {1: (2, 3), 2: (5, 6), 3: (8, 9)}
PROMPT_CELL = 11


def _defs_only(source):
    """Keep imports, function defs and UPPER_CASE/_private constants; drop the
    cell's trailing 'build and write' statements."""
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            keep.append(node)
        elif isinstance(node, ast.Assign) and all(
            isinstance(t, ast.Name) and (t.id.isupper() or t.id.startswith("_"))
            for t in node.targets
        ):
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    return compile(mod, str(NOTEBOOK), "exec")


def load_notebook_code():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    # One namespace per system: the builder cells define same-named private
    # helpers (regexes etc.), and a shared namespace lets a later cell
    # silently shadow an earlier one's helper.
    ns = {}
    for sysno, idxs in BUILDER_CELLS.items():
        sys_ns = {}
        for i in idxs:
            exec(_defs_only("".join(cells[i]["source"])), sys_ns)
        ns[sysno] = sys_ns
    prompt_ns = {}
    exec(_defs_only("".join(cells[PROMPT_CELL]["source"])), prompt_ns)
    return ns, prompt_ns


def _one(glob_pattern, base):
    """Return the artifact matching the pattern. A run in which System 3 sent
    feedback to System 2 has one file per feedback iteration (suffix _0, _1,
    ...); the last iteration is the one that produced the final outcome."""
    hits = sorted(base.glob(glob_pattern),
                  key=lambda p: int(p.stem.rsplit("_", 1)[-1]) if p.stem.rsplit("_", 1)[-1].isdigit() else 0)
    if not hits:
        raise FileNotFoundError(f"no {glob_pattern} under {base}")
    if len(hits) > 1:
        print(f"  {len(hits)} files match {glob_pattern}; using the last iteration: {hits[-1].name}")
    return hits[-1]


def build_reports(ns, run_dir, out_dir, systems):
    art = run_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    built = {}
    if 1 in systems:
        rep = ns[1]["build_system1_report"](
            _one("system1_*.json", art), _one("chats/system1_chat_log_*.json", art))
        md = ns[1]["render_system1_report_md"](rep)
        built[1] = (rep, md)
    if 2 in systems:
        rep = ns[2]["build_system2_report"](
            _one("system2_*.json", art), _one("chats/system2_chat_log_*.json", art))
        md = ns[2]["render_system2_report_md"](rep)
        built[2] = (rep, md)
    if 3 in systems:
        rep = ns[3]["build_system3_report"](_one("chats/system3_chat_log_*.json", art))
        md = ns[3]["render_system3_report_md"](rep)
        built[3] = (rep, md)
    for sysno, (rep, md) in built.items():
        (out_dir / f"system{sysno}_report.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"system{sysno}_report.md").write_text(md, encoding="utf-8")
        print(f"System {sysno} report: {len(md):,} chars -> {out_dir / f'system{sysno}_report.md'}")
    return built


def _title(key):
    return key.replace("_", " ").capitalize().replace("System1", "System 1") \
        .replace("System2", "System 2").replace("System3", "System 3")


def build_user_prompt(sysno, rubric, query_line, report_text):
    smin = float(rubric.get("score_min", 0))
    smax = float(rubric.get("score_max", 10))
    step = float(rubric.get("score_step", 0.5))
    dim_keys = list(rubric["dimensions"])
    schema_keys = set(json.loads(rubric.get("output_schema", "{}")))
    assert schema_keys == set(dim_keys), (
        f"rubric_s{sysno}_final.yaml: output_schema keys {sorted(schema_keys)} "
        f"!= dimensions {dim_keys}")
    for dk in dim_keys:
        d = rubric["dimensions"][dk]
        assert isinstance(d, dict) and "rubric" in d and "short_name" in d, (
            f"dimension {dk} malformed (check YAML indentation)")

    scale = "\n".join(f"- {l}" for l in (rubric.get("ordinal_scale_lines") or []))
    text = f"### Score scale ({smin:g}-{smax:g}, step {step:g})\n{scale}\n"
    # every top-level free-text section, in file order
    skip = {"ordinal_scale_lines", "dimensions", "output_instructions", "output_schema"}
    for key, val in rubric.items():
        if key in skip or not isinstance(val, str) or not val.strip():
            continue
        text += f"\n### {_title(key)}\n{val}\n"
    for dk in dim_keys:
        d = rubric["dimensions"][dk]
        text += f"\n### {d['name']} (weight: {d['weight']})\n"
        if d.get("primary_question"):
            text += f"Primary question: {d['primary_question'].strip()}\n\n"
        text += f"{d['rubric']}\n"

    out_block = ""
    if rubric.get("output_instructions"):
        out_block += rubric["output_instructions"].strip() + "\n\n"
    out_block += ("Respond with ONLY a JSON object in this exact structure:\n\n"
                  f"```\n{rubric.get('output_schema', '').strip()}\n```")
    user = (f"## Material Substitution Query\n\n{query_line}\n\n"
            f"## Evaluation Rubric (System {sysno} only)\n{text}\n\n"
            f"## System {sysno} Report\n\n{report_text}\n\n"
            f"## Required Output Format\n\n{out_block}")
    return user, dim_keys, (smin, smax, step)


def judge(sysno, prompt_ns, rubric_path, report_md, query_line, label, out_dir, model, api_key):
    rubric = rre.load_rubric(rubric_path)
    system_prompt = prompt_ns[f"S{sysno}_JUDGE_SYSTEM_PROMPT"]
    report_text = report_md.read_text(encoding="utf-8")
    user_prompt, dim_keys, (smin, smax, step) = build_user_prompt(
        sysno, rubric, query_line, report_text)
    print(f"\n[System {sysno}] judge model: {model} | rubric: {rubric_path.name} | "
          f"prompt: {len(user_prompt):,} chars (system {len(system_prompt):,})")

    eval_dir = out_dir / f"s{sysno}_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M")
    (eval_dir / f"judge_prompt_{ts}.txt").write_text(
        "### SYSTEM PROMPT\n" + system_prompt + "\n\n### USER PROMPT\n" + user_prompt,
        encoding="utf-8")

    from openai import OpenAI
    t0 = time.time()
    parsed, meta, deviations = rre.call_judge(
        OpenAI(api_key=api_key), model, system_prompt, user_prompt, rubric)
    elapsed = time.time() - t0
    print(f"judge responded in {elapsed:.1f}s | finish_reason={meta.get('finish_reason')} "
          f"| usage={meta.get('usage')}")
    if deviations:
        print("protocol deviations:", "; ".join(deviations))
    (eval_dir / f"judge_raw_{ts}.json").write_text(
        json.dumps({"response": meta.pop("raw_text", ""), "meta": meta},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    if "error" in parsed:
        raise RuntimeError(f"judge failed for System {sysno}: {parsed}")

    block = parsed.get("A") if isinstance(parsed.get("A"), dict) else parsed

    def snap(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return round(max(smin, min(smax, v)) / step) * step

    scores = {}
    for dk in dim_keys:
        e = block.get(dk) or {}
        scores[dk] = {"score": snap(e.get("score")), "strengths": e.get("strengths", []),
                      "weaknesses": e.get("weaknesses", []), "rationale": e.get("rationale", ""),
                      "confidence": e.get("confidence", "")}
    valid = [s["score"] for s in scores.values() if s["score"] is not None]
    mean = round(sum(valid) / len(valid), 2) if valid else None

    hdr = [rubric["dimensions"][dk]["short_name"] for dk in dim_keys]
    print(f"\n{'run':<20}" + "".join(f"{h:<10}" for h in hdr) + f"{'Mean':>7}")
    print(f"{label:<20}" + "".join(f"{str(scores[dk]['score']):<10}" for dk in dim_keys) + f"{mean:>7}")
    for dk in dim_keys:
        s = scores[dk]
        print(f"\n[{rubric['dimensions'][dk]['name']}]  score = {s['score']} (confidence: {s['confidence']})")
        for tag, items in (("+", s["strengths"]), ("-", s["weaknesses"])):
            for it in items:
                print(f"   {tag} {it}")
        print(f"   rationale: {s['rationale']}")

    result = {
        "study": f"System {sysno} only: LLM judge on system{sysno}_report.md (final rubric, 0-10 scale)",
        "query_name": "Query1", "query_sentence": query_line,
        "report_path": str(report_md), "rubric_path": str(rubric_path),
        "dimensions": dim_keys, "scores": {label: scores}, "mean_score": mean,
        "judge_model": model,
        "protocol_deviation": "; ".join(deviations) if deviations else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "judge_elapsed_seconds": round(elapsed, 1), "judge_completion": meta,
    }
    outfile = eval_dir / f"eval_{label}_s{sysno}_v2_{ts}.json"
    outfile.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("saved:", outfile)
    return scores, mean


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="e.g. results_seedstudy_gemma4/seed_202/Query1")
    ap.add_argument("--label", required=True, help="run label used in output file names, e.g. gemma4_seed_202")
    ap.add_argument("--out-dir", required=True, help="e.g. evaluation_trial/gemma4_seed_202")
    ap.add_argument("--systems", default="1,2,3")
    ap.add_argument("--judge-model", default="gpt-5.6-sol")
    ap.add_argument("--no-judge", action="store_true", help="only build the reports")
    ap.add_argument("--api-key-file", default=str(Path.home() / "Github/api_key.env"))
    args = ap.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    systems = [int(s) for s in args.systems.split(",") if s.strip()]

    mars_path = run_dir / "mars.json"
    if mars_path.exists():
        q = json.loads(mars_path.read_text(encoding="utf-8")).get("query", {})
        query_line = q.get("sentence", "") if isinstance(q, dict) else str(q)
    else:
        # a run that crashed before the end never writes mars.json; the System 1
        # artifact carries the same query sentence
        s1 = json.loads(_one("system1_*.json", run_dir / "artifacts").read_text(encoding="utf-8"))
        query_line = s1.get("sentence", "")
        print(f"note: {mars_path} missing (crashed run); query taken from the System 1 artifact")

    ns, prompt_ns = load_notebook_code()
    build_reports(ns, run_dir, out_dir, systems)
    if args.no_judge:
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and Path(args.api_key_file).exists():
        for line in open(args.api_key_file):
            if line.strip().startswith("OPENAI_API_KEY="):
                api_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    assert api_key, "no API key: set OPENAI_API_KEY or --api-key-file"

    summary = {}
    for sysno in systems:
        scores, mean = judge(
            sysno, prompt_ns, HERE / f"rubric_s{sysno}_final.yaml",
            out_dir / f"system{sysno}_report.md", query_line, args.label, out_dir,
            args.judge_model, api_key)
        summary[sysno] = (scores, mean)
    print("\n===== summary:", args.label, "=====")
    for sysno, (scores, mean) in summary.items():
        print(f"System {sysno}: " + ", ".join(f"{dk.split('_', 1)[1]}={s['score']}" for dk, s in scores.items())
              + f" | mean {mean}")


if __name__ == "__main__":
    main()
