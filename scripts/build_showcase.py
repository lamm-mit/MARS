"""Generator for notebooks/mars_showcase.ipynb — the "run any MARS query" showcase.

WHAT THIS FILE IS
    A *generator*, not the notebook itself. Running it emits a Jupyter notebook:

        python scripts/build_showcase.py notebooks/mars_showcase.ipynb

    The produced notebook is what you open and Run All. This script only writes
    that .ipynb file; it never loads the MARS knowledge graphs or runs the
    pipeline. All the logic that reads mars.json, picks the latest run, and draws
    the figures is Python *text* dropped into the notebook's code cells — it runs
    later, inside Jupyter.

WHY A GENERATOR (instead of hand-editing the .ipynb)
    A notebook file is bulky JSON where every source line is a quoted string in an
    array — painful and error-prone to edit by hand. Writing the cells as ordinary
    Python here and letting the script assemble the JSON means we can regenerate a
    clean, valid notebook deterministically after any tweak.

HOW IT WORKS (three moving parts)
    1. `md(...)` / `code(...)` helpers append cells to the `cells` list, in order.
       Everything between here and the "assemble" section at the bottom is just a
       sequence of those calls — the notebook's cells, top to bottom.
    2. Each `code(r'''...''')` block is the *verbatim source* of one notebook code
       cell. Raw triple-quoted strings so f-strings/braces/backslashes inside pass
       through untouched. (Rule: never put ''' inside a cell body — it would close
       the string here.)
    3. The "assemble" section converts `cells` into nbformat JSON and writes it.

    The generated notebook is self-contained: a CONFIG cell (QUERY / PREFER), a
    SETUP cell (resolve data + design system), then one section per pipeline stage.

stdlib-only — no third-party imports needed to build the notebook.
"""
import json, sys
from pathlib import Path

# `cells` accumulates (kind, source) tuples in document order; md()/code() append
# a markdown or code cell respectively. The whole script body below is just calls
# to these two helpers, followed by the assembly step at the very bottom.
cells = []
# NOTE: prose is rendered as HTML *output* (not native markdown cells). Native
# markdown cells follow the IDE theme (dark in VS Code) and can't be restyled from
# output CSS — so md() emits a code cell that calls prose(...), which converts the
# markdown to HTML on a forced-white, dark-text card. Result: white in ANY theme.
def md(t): cells.append(("code", "prose(" + json.dumps(t) + ")"))  # prose card
def code(t): cells.append(("code", t))  # add a code cell (t = verbatim cell source)

# The title is emitted LATER (after setup), because it renders via prose() which is
# defined in the setup cell. Cells run top-to-bottom, so prose() must exist first.

# ── config cell (first cell — the query knob) ─────────────────────────────────
code(r'''# ══ CONFIG — change this, then Run All ══════════════════════════════
QUERY  = 1          # ← query number: 1, 2, 3, …
PREFER = "auto"     # "auto"  → frozen paper result if present, else latest local run
                    # "frozen"→ force results_from_paper/  |  "local" → force results/
''')

# ── setup cell ───────────────────────────────────────────────────────────────
code(r'''# Setup — resolve data (latest run), load artifacts, design system
import json, html, re
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import HTML, Markdown, Image, display
%matplotlib inline
esc = html.escape

def _find_root():
    for cand in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (cand / "config").is_dir() and (cand / "src").is_dir():
            return cand
    return Path.cwd().resolve()
ROOT = _find_root()
def _load(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

QN = f"Query{QUERY}"
# Where to look for this query's data. "frozen" = curated paper results,
# "local" = your own pipeline runs. PREFER decides the search order; "auto" tries
# the frozen result first (the polished showcase) and falls back to a local run.
ROOTS = {"frozen": ROOT / "results_from_paper", "local": ROOT / "results"}
SEARCH = {"auto": ["frozen", "local"], "frozen": ["frozen"], "local": ["local"]}[PREFER]
DATA = SRC = None
for key in SEARCH:                       # take the first location that actually has a mars.json
    d = ROOTS[key] / QN
    if (d / "mars.json").exists():
        DATA, SRC = d, key; break
if DATA is None:
    raise FileNotFoundError(f"No mars.json for {QN} under {[str(ROOTS[k]/QN) for k in SEARCH]}")
IS_FROZEN = (SRC == "frozen")
ART = DATA / "artifacts"

mars = _load(DATA / "mars.json")

# ── pick the LATEST run on disk (run-ids are YYYYMMDDHH → lexical max = latest) ──
run_ids = sorted({p.stem.replace("pipeline_run_", "") for p in ART.glob("pipeline_run_*.json")})
RUN_ID = max(run_ids) if run_ids else str(mars["metadata"]["pipeline_run_id"])
N_RUNS = len(run_ids)
prun = ART / f"pipeline_run_{RUN_ID}.json"
pipe = _load(prun) if prun.exists() else _load(sorted(ART.glob("pipeline_run_*.json"))[-1])

# latest subgraph for this run (…{RUN_ID}_{iter}…); else any latest
sgs = sorted(ART.glob(f"subgraphs/{RUN_ID}_*_material_informed.json")) \
      or sorted(ART.glob("subgraphs/*_material_informed.json"))
subg = _load(sgs[-1])["data"] if sgs else {"nodes": [], "links": []}
CHATS = sorted(p for p in (ART / "chats").glob("*.json") if RUN_ID in p.name) \
        or sorted((ART / "chats").glob("*.json"))

# evaluation only if an aggregate covering THIS query exists under the chosen root
EVALF = ROOTS[SRC] / "evaluation" / "aggregate_results.json"
agg = _load(EVALF) if EVALF.exists() else None
HAS_EVAL = bool(agg) and any(pq.get("query_name") == QN for pq in agg.get("per_query", []))

# ── design tokens (validated colorblind-safe palette; forced white surface) ──
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, LINE = "#ffffff", "#e1e0d9", "#c3c2b7"
ACCENT, GOOD, CRIT, WARN = "#2a78d6", "#0ca30c", "#d03b3b", "#eda100"
CAT = ["#2a78d6","#1baf7a","#eda100","#008300","#4a3aa7","#e34948","#e87ba4","#eb6834"]
mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11, "axes.edgecolor": LINE, "axes.linewidth": 0.8,
    "axes.grid": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK, "svg.fonttype": "none",
})
display(HTML("""<style>
.jp-RenderedMarkdown,.text_cell_render{background:#ffffff!important;color:#141414!important;border-radius:8px;padding:8px 14px;}
.jp-RenderedMarkdown *,.text_cell_render *{color:#141414!important;}
.jp-RenderedMarkdown a,.text_cell_render a{color:#1c5cab!important;}
.jp-RenderedMarkdown code,.text_cell_render code{background:#f2f2ef!important;color:#0b0b0b!important;}
.jp-RenderedMarkdown th,.jp-RenderedMarkdown td,.text_cell_render th,.text_cell_render td{border-color:#e1e0d9!important;}
.jp-OutputArea-output,.output_html,.jp-RenderedHTMLCommon{background:#ffffff!important;}
.m-wrap{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#0b0b0b;line-height:1.5;background:#ffffff;padding:14px 16px;border-radius:12px;}
.m-tiles{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0;}
.m-tile{flex:1 1 130px;background:#fff;border:1px solid rgba(11,11,11,.08);border-radius:12px;padding:14px 16px;}
.m-tile .v{font-size:25px;font-weight:700;letter-spacing:-.02em;line-height:1.1;}
.m-tile .l{font-size:11px;color:#52514e;margin-top:4px;text-transform:uppercase;letter-spacing:.05em;}
.m-card{background:#fff;border:1px solid rgba(11,11,11,.08);border-radius:12px;padding:18px 20px;box-shadow:0 1px 2px rgba(11,11,11,.04);margin:10px 0;}
.m-h{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#52514e;margin:0 0 8px;}
.m-title{font-size:19px;font-weight:700;letter-spacing:-.02em;margin:2px 0 8px;}
.m-badge{display:inline-block;padding:3px 11px;border-radius:999px;font-size:12px;font-weight:700;vertical-align:middle;}
.m-good{background:rgba(12,163,12,.13);color:#0a7d0a;} .m-crit{background:rgba(208,59,59,.13);color:#b52e2e;}
.m-warn{background:rgba(237,161,0,.16);color:#8a6000;} .m-info{background:rgba(42,120,214,.13);color:#1c5cab;}
.m-list{margin:0;padding:0;list-style:none;} .m-list li{padding:7px 2px;border-bottom:1px solid #f0efec;font-size:13.5px;}
.m-term{font-weight:650;color:#0b0b0b;}
.m-2col{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
.m-tl{position:relative;margin:4px 0 4px 8px;padding-left:26px;border-left:2px solid #e1e0d9;}
.m-node{position:relative;margin:0 0 14px;padding:12px 16px;background:#fff;border:1px solid rgba(11,11,11,.09);border-radius:10px;border-left-width:4px;}
.m-node:before{content:"";position:absolute;left:-33px;top:18px;width:11px;height:11px;border-radius:50%;background:#c3c2b7;border:2px solid #fff;}
.m-node .st{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#898781;}
table.m-tbl{border-collapse:collapse;width:100%;font-size:13px;}
.m-tbl th{text-align:left;color:#52514e;font-weight:600;border-bottom:2px solid #e1e0d9;padding:7px 10px;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;}
.m-tbl td{border-bottom:1px solid #f0efec;padding:8px 10px;vertical-align:top;} .m-tbl tr.hi td{background:rgba(42,120,214,.06);font-weight:600;}
details.m-det{background:#fff;border:1px solid rgba(11,11,11,.08);border-radius:10px;padding:7px 13px;margin:7px 0;}
details.m-det>summary{cursor:pointer;font-weight:600;color:#0b0b0b;list-style:none;} details.m-det>summary::-webkit-details-marker{display:none;}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre-wrap;color:#333;}
.m-cap{font-size:12px;color:#898781;margin:6px 2px;} .pill{display:inline-block;font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:6px;margin-right:6px;}
</style>"""))

def tiles(items):
    out = []
    for it in items:
        v, l = it[0], it[1]; c = it[2] if len(it) > 2 else INK
        out.append(f'<div class="m-tile"><div class="v" style="color:{c}">{v}</div><div class="l">{l}</div></div>')
    return HTML('<div class="m-wrap"><div class="m-tiles">' + "".join(out) + "</div></div>")

# ── prose(): render a markdown string as an inline-white HTML card ──
# A tiny markdown→HTML converter for the subset we use (headings, bold, inline
# code, links, bullet/numbered lists, tables, blockquotes, hr). Text is HTML-escaped
# first (so model output like "<10 cm" and ">200 %" is safe), then patterns applied.
def _md2html(t):
    def il(s):
        s = esc(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`([^`]+)`", r'<code style="background:#f2f2ef;padding:1px 5px;border-radius:4px;font-size:.92em">\1</code>', s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" style="color:#1c5cab">\1</a>', s)
        return s
    L, out, i = t.split("\n"), [], 0
    while i < len(L):
        st = L[i].strip()
        if not st: i += 1; continue
        if st == "---":
            out.append('<hr style="border:none;border-top:1px solid #e1e0d9;margin:16px 0">'); i += 1; continue
        m = re.match(r"(#{1,6})\s+(.*)", st)          # heading
        if m:
            lv = len(m.group(1)); sz = {1: "22px", 2: "19px", 3: "15px"}.get(lv, "13px")
            out.append(f'<div style="font-weight:700;font-size:{sz};margin:{"18px" if lv<=2 else "12px"} 0 6px;color:#0b0b0b">{il(m.group(2))}</div>')
            i += 1; continue
        if st.startswith("> "):                        # blockquote
            buf = []
            while i < len(L) and L[i].strip().startswith(">"):
                buf.append(il(L[i].strip().lstrip(">").strip())); i += 1
            out.append('<div style="border-left:3px solid #c3c2b7;padding:6px 12px;margin:8px 0;color:#52514e;background:#faf9f7">' + "<br>".join(buf) + "</div>"); continue
        if st.startswith("|"):                          # table
            tb = []
            while i < len(L) and L[i].strip().startswith("|"): tb.append(L[i].strip()); i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in tb]
            rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]   # drop |---| separator
            if rows:
                th = "".join(f'<th style="text-align:left;border-bottom:2px solid #e1e0d9;padding:6px 10px;font-size:12px;color:#52514e">{il(c)}</th>' for c in rows[0])
                tr = "".join("<tr>" + "".join(f'<td style="border-bottom:1px solid #f0efec;padding:6px 10px;font-size:13px">{il(c)}</td>' for c in r) + "</tr>" for r in rows[1:])
                out.append(f'<table style="border-collapse:collapse;width:100%;margin:8px 0"><tr>{th}</tr>{tr}</table>')
            continue
        if re.match(r"(-|\*|\d+\.)\s+", st):            # bullet / numbered list
            od, items = bool(re.match(r"\d+\.\s+", st)), []
            while i < len(L) and re.match(r"(-|\*|\d+\.)\s+", L[i].strip()):
                items.append("<li style='margin:3px 0'>" + il(re.sub(r"^(-|\*|\d+\.)\s+", "", L[i].strip())) + "</li>"); i += 1
            tag = "ol" if od else "ul"
            out.append(f"<{tag} style='margin:6px 0 6px 4px;padding-left:22px'>" + "".join(items) + f"</{tag}>"); continue
        buf = []                                         # paragraph
        while i < len(L) and L[i].strip() and not re.match(r"(#{1,6}\s|>\s|\||-\s|\*\s|\d+\.\s|---)", L[i].strip()):
            buf.append(il(L[i].strip())); i += 1
        out.append(f'<p style="margin:8px 0;font-size:14px">{" ".join(buf)}</p>')
    return "".join(out)

def prose(t):
    style = ("background:#ffffff;color:#141414;font-family:system-ui,-apple-system,sans-serif;"
             "line-height:1.55;padding:16px 20px;border-radius:12px;border:1px solid rgba(11,11,11,.07);margin:6px 0")
    display(HTML(f'<div style="{style}">' + _md2html(t) + "</div>"))

print(f"{QN}: source={SRC}  run={RUN_ID}  ({N_RUNS} run(s) on disk)  eval={HAS_EVAL}")
print(f"  candidate: {mars['candidate_selection']['final_candidate']['material_name']}")
''')

# ── title (emitted here so prose() is already defined; renders white in any theme) ─
md(r"""# MARS — Material-Substitution Showcase

### Hierarchical Multi-Agent Reasoning for Knowledge-Grounded Material Substitution

**Set the query number in the config cell above, then Run All.** This notebook renders
any MARS run end-to-end — the requirements, the closed-loop search, the knowledge
subgraph, the manufacturing route, and (where available) the blind evaluation.

It auto-resolves where the data lives: it prefers the **frozen paper result**
(`results_from_paper/QueryN/`) when one exists, otherwise your **latest local run**
(`results/QueryN/`). When a query has several runs on disk, it always uses the
**latest** one.""")

# ── dynamic banner (which query / material / data source) ─────────────────────
code(r'''q = mars["query"]
if IS_FROZEN:
    note = ('<span class="m-badge m-info">Frozen paper result</span> Loaded from '
            f'<code>{esc(str(DATA.relative_to(ROOT)))}</code> — produced with the full proprietary '
            'corpora and the <code>gpt-oss-20b</code> backbone; the exact output reported in the paper.')
else:
    note = ('<span class="m-badge m-warn">Local run</span> Loaded from '
            f'<code>{esc(str(DATA.relative_to(ROOT)))}</code> — a run on this machine using placeholder '
            '(dummy) retrieval corpora. The pipeline &amp; reasoning are real; the specific materials '
            'reflect the dummy inventory, not the proprietary corpora. That retrieval layer is the '
            'plug-in point for the real proprietary data.')
runsnote = f" · latest of {N_RUNS} runs on disk" if N_RUNS > 1 else ""
display(HTML(f"""<div class="m-wrap"><div class="m-card" style="border-left:4px solid {ACCENT}">
  <div class="m-h">{esc(QN)} &nbsp;·&nbsp; run {esc(RUN_ID)}{esc(runsnote)}</div>
  <div class="m-title">Replace: {esc(q['material_X'])}</div>
  <div style="font-size:13px;color:#52514e;margin-bottom:8px">for the context of <b>{esc(q['application_Y'])}</b></div>
  <div style="font-size:12.5px;color:#333">{note}</div></div></div>"""))
''')

# ── Act 1: task ───────────────────────────────────────────────────────────────
md(r"""## 1 · The Task

The full natural-language query, split by System 1's front end into the target
material `X`, the application context `Y`, and the free-text intent.""")

code(r'''# Count ALL non-empty items so the tiles match the numbered list below (render_reqs)
props = [p for p in mars["required_material_properties"]["properties"] if p.strip()]
cons  = [c for c in mars["required_material_properties"]["constraints"] if c.strip()]
meta  = mars["metadata"]; nrej = meta["total_rejected_candidates"]
statustxt = mars["manufacturing_process"]["status"].capitalize()
display(HTML(f"""<div class="m-wrap"><div class="m-card">
  <div class="m-h">Evaluation query</div><div style="font-size:15px;">{esc(q['sentence'])}</div></div></div>"""))
tiles([
    (len(props), "Required properties"), (len(cons), "Hard constraints"),
    (pipe["system2_system3_loop"]["total_iterations"], "S2↔S3 iterations"),
    (nrej, "Candidates rejected", CRIT if nrej else MUTED),
    (f"{pipe['total_duration_seconds']/60:.0f} min", "Wall-clock"),
    (statustxt, "Final verdict", GOOD if statustxt.lower() == "manufacturable" else CRIT),
])
''')

# ── architecture ─────────────────────────────────────────────────────────────
md(r"""## 2 · The MARS Architecture

Three cooperating agentic systems in a closed loop, each built from the same four
reusable agent roles — **Research Manager** (reasoning), **Analyst** (RAG),
**Scientist** (knowledge-graph traversal), **Assistant** (extraction).

| System | Job | Output |
|---|---|---|
| **1 · Requirements** | infer what the substitute must do | properties `W`, hard constraints `H` |
| **2 · Candidate generation** | propose a grounded, lab-available substitute | candidate `S` + evidence `E` |
| **3 · Manufacturability** | can it actually be made at lab scale? | recipe `P` **or** blocking constraints `C` → back to System 2 |
""")

code(r'''img = ROOT / "fullSystem.png"
if img.exists():
    display(HTML('<div class="m-wrap"><div class="m-cap">Figure 1 — closed-loop hierarchical architecture</div></div>'))
    display(Image(str(img), width=980))
''')

# ── Act 2: System 1 ──────────────────────────────────────────────────────────
md(r"""## 3 · System 1 — Requirements Analysis

System 1 decomposes the query into focused research questions, answers each with
dual text + KG retrieval, and distills the answers into the required properties `W`
and non-negotiable hard constraints `H` (shown verbatim below).""")

code(r'''def render_reqs(items):
    # Number EVERY item (queries vary: some list terse phrases with no separator).
    # If an item has a term–detail separator, bold the term; otherwise show it whole.
    out = []
    for n, it in enumerate((x.strip() for x in items if x.strip()), 1):
        num = f'<span style="color:{ACCENT};font-weight:700;font-variant-numeric:tabular-nums;margin-right:8px">{n}.</span>'
        sep = next((s for s in ("–", "—", ":") if s in it), None)
        if sep:
            term, detail = it.split(sep, 1)
            body = (f'<span class="m-term">{esc(term.strip())}</span>'
                    f'<span style="color:#52514e"> — {esc(detail.strip())}</span>')
        else:
            body = f'<span class="m-term">{esc(it)}</span>'
        out.append(f'<li style="display:flex;gap:2px">{num}<span>{body}</span></li>')
    return '<ul class="m-list">' + "".join(out) + "</ul>"
P = mars["required_material_properties"]
display(HTML(f"""<div class="m-wrap"><div class="m-2col">
  <div class="m-card"><div class="m-title">Required properties&nbsp;<span class="m-badge m-warn">W</span></div>{render_reqs(P['properties'])}</div>
  <div class="m-card"><div class="m-title">Hard constraints&nbsp;<span class="m-badge m-crit">H</span></div>{render_reqs(P['constraints'])}</div>
</div></div>"""))
''')

# ── Act 3: closed-loop trace ─────────────────────────────────────────────────
md(r"""## 4 · System 2 ⇄ System 3 — The Closed-Loop Search

System 2 proposes a grounded candidate; System 3 stress-tests manufacturability.
When System 3 **blocks** a candidate, its reasons feed back as new constraints and
the search resumes — the system *self-corrects*. Trace reconstructed from the run's
`pipeline_run` record.""")

code(r'''loop = pipe["system2_system3_loop"]; rej = mars["candidate_selection"]["rejected_candidates"]
def tnode(stage, title, body, color, badge=None):
    b = f'<span class="m-badge {badge[0]}" style="margin-left:8px">{badge[1]}</span>' if badge else ""
    return (f'<div class="m-node" style="border-left-color:{color}"><div class="st">{esc(stage)}</div>'
            f'<div style="font-weight:650;font-size:14.5px;margin:2px 0 4px">{esc(title)}{b}</div>'
            f'<div style="font-size:13px;color:#52514e">{body}</div></div>')
parts = [tnode("System 1", "Requirements extracted",
              f"{len(props)} required properties, {len(cons)} hard constraints → handed to System 2", MUTED)]
for it in loop["iterations"]:
    s2, s3, n = it["system2"], it["system3"], it["iteration"]
    parts.append(tnode(f"Iteration {n} · System 2", f"Proposes: {s2['candidate']}",
                       "Grounded in the lab-material database + Material-Informed Subgraph", ACCENT))
    if s3["status"] == "manufacturable":
        parts.append(tnode(f"Iteration {n} · System 3", "Manufacturability assessment",
                           "Evidence-supported lab-scale synthesis route confirmed.", GOOD, ("m-good", "✓ MANUFACTURABLE")))
    else:
        bc = s3.get("blocking_constraints", [])
        reason = esc(bc[0].get("description", str(bc[0])) if isinstance(bc[0], dict) else str(bc[0]))[:240] if bc else ""
        parts.append(tnode(f"Iteration {n} · System 3", "Manufacturability assessment",
                           (reason or "Blocked.") + " &nbsp;→&nbsp; constraints fed back to System 2", CRIT, ("m-crit", "✕ BLOCKED")))
parts.append(tnode("Result", mars["candidate_selection"]["final_candidate"]["material_name"],
                   "Accepted substitute.", GOOD, ("m-good", "FINAL")))
if rej:
    rows = "".join(f'<tr><td style="width:60%">{esc(r["candidate"])}</td>'
                   f'<td><span class="pill" style="background:rgba(137,135,129,.15);color:#52514e">{esc(r["source"])}</span></td>'
                   f'<td style="font-size:12px;color:#52514e">{esc("; ".join(r["constraints_violated"])[:160]) or "—"}</td></tr>' for r in rej)
    extra = (f'<details class="m-det"><summary>All {len(rej)} rejected candidates</summary>'
             f'<table class="m-tbl" style="margin-top:8px"><tr><th>Candidate</th><th>Rejected by</th><th>Reason</th></tr>{rows}</table></details>')
else:
    extra = '<div class="m-cap">No candidates were rejected — MARS converged on its first proposal for this query.</div>'
display(HTML(f'<div class="m-wrap"><div class="m-tl">{"".join(parts)}</div>{extra}</div>'))
''')

md("### The accepted substitute\n\nThe verbatim System 2 justification for the final candidate:")

code(r'''cand = mars["candidate_selection"]["final_candidate"]; just = cand["justification"]
if "Justification:" in just: just = just.split("Justification:", 1)[1].strip()
display(HTML(f"""<div class="m-wrap"><div class="m-card" style="border-left:4px solid {GOOD}">
  <div class="m-h">Final candidate S</div><div class="m-title">{esc(cand['material_name'])}</div></div></div>"""))
prose(just)   # model justification is markdown → render as white prose card
''')

# ── Act 4: KG subgraph ───────────────────────────────────────────────────────
md(r"""## 5 · The Knowledge Substrate — Material-Informed Subgraph

Before proposing, System 2 grounds the required properties and every lab material
onto two knowledge graphs (Material Properties + Patents), links them by shortest
paths, and merges semantically-equivalent nodes. This is the relational evidence
the Research Manager reasons over.""")

code(r'''G = nx.DiGraph()
for nd in subg["nodes"]: G.add_node(nd["id"], **{k: v for k, v in nd.items() if k != "id"})
for e in subg.get("links", subg.get("edges", [])): G.add_edge(e["source"], e["target"], relation=e.get("relation", ""))
def kgc(pred): return sum(1 for _, d in G.nodes(data=True) if pred(d.get("source_kgs", [])))
tiles([
    (G.number_of_nodes(), "Nodes"), (G.number_of_edges(), "Edges (relations)"),
    (kgc(lambda s: "material_properties" in s and "patents" not in s), "Material-Props KG"),
    (kgc(lambda s: "patents" in s), "Patents KG"),
    (sum(1 for _, d in G.nodes(data=True) if "material" in str(d.get("type", "")).lower()), "Material nodes"),
])
''')

code(r'''import matplotlib.patches as mp
TOP_N, SEED = 70, 42
BUCKETS = [("material", CAT[0], ("material","component","sample","polymer","composite")),
           ("property", CAT[1], ("propert","parameter","strength","modulus")),
           ("chemical/compound", CAT[2], ("chemical","compound","substance","solvent")),
           ("process/method", CAT[4], ("process","method","technique","mechanism","treatment")),
           ("application", CAT[7], ("application","use"))]
def bucket(t):
    t = str(t).lower()
    for name, _c, keys in BUCKETS:
        if any(k in t for k in keys): return name
    return "other"
BCOL = {n: c for n, c, _ in BUCKETS}; BCOL["other"] = MUTED
def artifact(n, d):
    t = str(d.get("type", "")).lower()
    return t in ("image", "table") or str(n).lower().endswith((".png", ".jpg", ".jpeg"))
Gv = G.copy(); Gv.remove_nodes_from([n for n, d in G.nodes(data=True) if artifact(n, d)])
if Gv.number_of_nodes() >= 3:
    top = [n for n, _ in sorted(Gv.degree(), key=lambda x: x[1], reverse=True)[:TOP_N]]
    H = Gv.subgraph(top).copy()
    bkt = {n: bucket(H.nodes[n].get("type", "")) for n in H.nodes}
    deg = np.array([H.degree(n) for n in H.nodes], float)
    nsz = 120 + 1400 * (deg - deg.min()) / (np.ptp(deg) + 1e-9)
    pos = nx.spring_layout(H, seed=SEED, k=2.0/np.sqrt(len(H)), iterations=80)
    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.16, width=0.5, edge_color="#5b5a57", arrows=False)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=[BCOL[bkt[n]] for n in H.nodes], node_size=nsz,
                           edgecolors="#ffffff", linewidths=0.9)
    hubs = [n for n, _ in sorted(H.degree(), key=lambda x: x[1], reverse=True)[:12]]
    nx.draw_networkx_labels(H, pos, {n: n for n in hubs}, ax=ax, font_size=9, font_color=INK)
    present = [b for b in [x[0] for x in BUCKETS] + ["other"] if b in set(bkt.values())]
    ax.legend(handles=[mp.Patch(color=BCOL[b], label=b) for b in present], loc="upper left",
              frameon=False, fontsize=9.5, title="node type", title_fontsize=9.5)
    ax.set_title(f"Material-Informed Subgraph — top {min(TOP_N, len(H))} hubs of {G.number_of_nodes()} nodes (size ∝ degree)",
                 fontsize=12, color=INK, loc="left", pad=12)
    ax.axis("off"); plt.tight_layout(); plt.show()
    display(HTML('<div class="m-wrap"><div class="m-cap">Full graph in <code>graph_viz.ipynb</code>; the high-degree core is shown for legibility.</div></div>'))
else:
    display(HTML('<div class="m-wrap"><div class="m-cap">Subgraph too small to visualize for this run.</div></div>'))
''')

# ── Act 5: recipe ────────────────────────────────────────────────────────────
md(r"""## 6 · System 3 — Manufacturability & Process Design

System 3 decomposes the candidate into constituents, retrieves process knowledge
from textbooks + patents + spec sheets, and emits a lab-scale process recipe `P`
(or blocking constraints).""")

code(r'''mp_ = mars["manufacturing_process"]
badge = ('<span class="m-badge m-good">✓ Manufacturable</span>' if mp_["status"] == "manufacturable"
         else '<span class="m-badge m-crit">Blocked</span>')
def fmt_inputs(inp): return "—" if not inp else "<br>".join(f"• {esc(x)}" for x in inp)
if mp_.get("process_recipe"):
    rows = "".join(f'<tr><td style="text-align:center;font-weight:700;color:{ACCENT}">{s["step_index"]}</td>'
                   f'<td><b>{esc(s["description"])}</b></td>'
                   f'<td class="mono" style="font-size:11.5px">{esc(str(s.get("conditions") or "—"))}</td>'
                   f'<td style="font-size:12px;color:#52514e">{esc(s.get("equipment_class") or "—")}</td>'
                   f'<td style="font-size:12px">{fmt_inputs(s.get("inputs"))}</td></tr>' for s in mp_["process_recipe"])
    body = f'<table class="m-tbl"><tr><th>#</th><th>Step</th><th>Conditions</th><th>Equipment class</th><th>Inputs</th></tr>{rows}</table>'
else:
    bc = mp_.get("blocking_constraints", [])
    items = "".join(f"<li>{esc(b.get('description', str(b)) if isinstance(b, dict) else str(b))}</li>" for b in bc)
    body = f'<div class="m-h">Blocking constraints</div><ul>{items or "<li>—</li>"}</ul>'
display(HTML(f"""<div class="m-wrap"><div class="m-card"><div class="m-title">Lab-scale process recipe &nbsp;{badge}</div>{body}</div></div>"""))
''')

# ── Act 6: evaluation (runtime-conditional) ──────────────────────────────────
md(r"""## 7 · Evaluation — Blind LLM-as-Judge

Where available, MARS is compared against four reduced baselines: each produces the
same three standardized outputs, anonymized and shuffled to labels A–E, scored by an
independent judge (`gpt-4.1`) on 12 criteria (1–5), then unblinded.""")

code(r'''if not HAS_EVAL:
    display(HTML(f"""<div class="m-wrap"><div class="m-card">
      <div class="m-h">Not evaluated</div>
      <div style="font-size:13.5px;color:#333">The blind LLM-judge comparison was run for the paper's
      <b>Query1 (THV)</b> case study. It was not run for {esc(QN)}, so no comparative scores are shown.
      Set <code>QUERY = 1</code> above to see the full evaluation.</div></div></div>"""))
else:
    LBL = {"evaluation":"MARS (full)","ablation_3agent":"3 LLM Calls (no RAG/KG)","ablation_1agent_rag":"1 LLM Call (w/ RAG/KG)",
           "ablation_1agent_no_rag":"1 LLM Call (no RAG/KG)","ablation_1agent_no_rag_openai":"1 LLM Call (GPT‑5.4)"}
    DIMS = {"System 1":["system1_completeness","system1_relevance","system1_scientific","system1_clarity"],
            "System 2":["system2_alignment","system2_novelty","system2_realism","system2_reasoning"],
            "System 3":["system3_plausibility","system3_practicality","system3_compatibility","system3_industrial"]}
    ALL = [d for dl in DIMS.values() for d in dl]
    pq = next(p for p in agg["per_query"] if p["query_name"] == QN)
    sc = {}
    for cond, dd in pq["scores"].items():
        sc[cond] = {d: (dd[d]["score"] if isinstance(dd[d], dict) else dd[d]) for d in ALL}
        sc[cond]["weighted_avg"] = dd.get("weighted_total", float(np.mean([sc[cond][d] for d in ALL])))
    order = sorted(sc, key=lambda c: sc[c]["weighted_avg"], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    vals = [sc[c]["weighted_avg"] for c in order]
    cols = [ACCENT if c == "evaluation" else MUTED for c in order]
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, color=cols, height=0.62, zorder=3)
    for yi, v in zip(y, vals): ax.text(v + 0.06, yi, f"{v:.2f}", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([LBL.get(c, c) for c in order], fontsize=10.5, color=INK)
    ax.set_xlim(0, 5); ax.set_xticks(range(6)); ax.set_xlabel("Overall score (mean of 12 criteria, 1–5)")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0); ax.set_axisbelow(True); ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_title("Blind LLM-judge — overall ranking", fontsize=12, color=INK, loc="left", pad=10, fontweight="bold")
    plt.tight_layout(); plt.show()
    def tint(v): return f"background:rgba(42,120,214,{max(0.0,min(1.0,(v-2.5)/2.5))*0.34:.2f})"
    head = "".join(f"<th style='text-align:center'>{h}</th>" for h in ["Configuration","System 1","System 2","System 3","Total"])
    rows = []
    for c in order:
        subs = {k: float(np.mean([sc[c][d] for d in dl])) for k, dl in DIMS.items()}
        tot = sc[c]["weighted_avg"]; hi = " class='hi'" if c == "evaluation" else ""
        tc = "".join(f"<td style='text-align:center;{tint(subs[k])}'>{subs[k]:.2f}</td>" for k in DIMS)
        rows.append(f"<tr{hi}><td>{esc(LBL.get(c,c))}</td>{tc}<td style='text-align:center;font-weight:700;{tint(tot)}'>{tot:.2f}</td></tr>")
    display(HTML(f"""<div class="m-wrap"><div class="m-card">
      <div class="m-h">Subsystem-level scores (judge: {esc(agg['judge_model'])}, seed {agg['seed']})</div>
      <table class="m-tbl"><tr>{head}</tr>{''.join(rows)}</table></div></div>"""))
''')

# ── Act 7: traces ────────────────────────────────────────────────────────────
md(r"""## 8 · Under the Hood — Agent Reasoning Traces

Every agent turn is logged. Below: a summary of all logs, then a drill-down of each
system's trace (LLM prompts/responses, RAG queries, KG traversals) — this is what
makes MARS interpretable. Click any row to expand it.""")

code(r'''def summ(p):
    d = json.load(open(p, encoding="utf-8")); return d, d.get("summary", {})
rows = []
for p in CHATS:
    _, s = summ(p); nm = p.name.replace("_chat_log", "").replace(".json", "")
    rows.append(f"<tr><td class='mono' style='font-size:11px'>{esc(nm)}</td>"
                f"<td style='text-align:center'>{s.get('llm_calls',0)}</td>"
                f"<td style='text-align:center'>{s.get('rag_queries',0)}</td>"
                f"<td style='text-align:center'>{s.get('kg_queries',0)}</td></tr>")
display(HTML(f"""<div class="m-wrap"><div class="m-card"><div class="m-h">Reasoning logs for run {esc(RUN_ID)}</div>
  <table class="m-tbl"><tr><th>Log</th><th>LLM calls</th><th>RAG queries</th><th>KG queries</th></tr>{''.join(rows)}</table></div></div>"""))
''')

code(r'''def render_trace(logpath, sysname):
    d, s = summ(logpath)
    def clip(t, n=520):
        t = (t or "").strip(); return esc(t[:n] + ("…" if len(t) > n else ""))
    PILL = {"llm":("rgba(42,120,214,.14)","#1c5cab"),"rag":("rgba(27,175,122,.16)","#127a55"),"kg":("rgba(74,58,167,.14)","#4a3aa7")}
    blocks = []
    for ix, itr in enumerate(d["interactions"], 1):
        t = itr["type"]; bg, fg = PILL.get(t, ("#eee", "#333")); dta = itr.get("data", {})
        head = (f'<span class="pill" style="background:{bg};color:{fg}">{t.upper()}</span>'
                f'<b>{esc(itr.get("agent",""))}</b> · <span style="color:#52514e">{esc(itr.get("method",""))}</span>')
        if t == "llm":
            inner = (f'<div class="m-h" style="margin-top:8px">User prompt</div><div class="mono">{clip(dta.get("user_prompt",""))}</div>'
                     f'<div class="m-h" style="margin-top:8px">Response</div><div class="mono">{clip(dta.get("response",""))}</div>')
        elif t == "rag":
            res = dta.get("results", []); first = clip(res[0].get("content", "") if res else "", 300)
            inner = (f'<div style="margin-top:6px;font-size:12.5px"><b>Query:</b> {esc(dta.get("query",""))} · {dta.get("num_results",0)} results</div>'
                     f'<div class="mono" style="margin-top:6px">{first}</div>')
        else:
            inner = (f'<div style="margin-top:6px;font-size:12.5px"><b>Keywords:</b> {esc(", ".join(dta.get("keywords",[])))} · '
                     f'{dta.get("total_paths",0)} paths found</div>')
        blocks.append(f'<details class="m-det"><summary>{ix:>2}. {head}</summary>{inner}</details>')
    display(HTML(f"""<div class="m-wrap"><div class="m-card">
      <div class="m-title">{sysname} trace <span style="font-size:13px;color:#898781">{s.get('total_interactions',0)} interactions</span></div>
      <div style="font-size:13px;color:#52514e;margin-bottom:6px">{s.get('llm_calls',0)} LLM calls · {s.get('rag_queries',0)} RAG queries · {s.get('kg_queries',0)} KG traversals</div>
      {''.join(blocks)}</div></div>"""))

# One drill-down card per system. When a system ran over several iterations there are
# multiple logs — show the latest (the one behind the accepted candidate / final call).
for _label, _key in [("System 1", "system1"), ("System 2", "system2"), ("System 3", "system3")]:
    _m = [p for p in CHATS if _key in p.name]
    if _m:
        render_trace(_m[-1], _label)
''')

code(r'''prose(
    "---\n\n### Reproduce / explore further\n\n"
    f"- **This run's artifacts:** `{DATA.relative_to(ROOT)}/`\n"
    "- **Full subgraph viz:** `notebooks/graph_viz.ipynb`\n"
    "- **Interactive runs:** the MARS MCP server (`mcp_server/`)\n"
    f"- **Re-run:** `python scripts/run_mars.py --queries {QN} --override config/overrides/local_LLM_downloaded_KGs.yaml`\n")
''')

# Belt-and-suspenders: force white bg + dark text inline on every rendered card, so
# outputs stay readable even if an injected <style> block doesn't reach a renderer.
cells = [(k, s.replace('<div class="m-wrap">',
                       '<div class="m-wrap" style="background:#fff;color:#0b0b0b">'))
         if k == "code" else (k, s) for k, s in cells]

# ── assemble ──────────────────────────────────────────────────────────────────
# Turn the `cells` list into the nbformat-4 JSON structure Jupyter expects.
# Each tuple becomes a cell dict; code cells carry the extra fields Jupyter needs
# (empty execution_count / outputs, filled in when the user runs the notebook).
nb = {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": c} if k == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c}
        for k, c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
# nbformat stores `source` as a list of lines, each ending in "\n" except the last.
# Split our single-string sources into that shape so the JSON is canonical.
for cell in nb["cells"]:
    s = cell["source"]; cell["source"] = [l + "\n" for l in s.split("\n")]
    if cell["source"]: cell["source"][-1] = cell["source"][-1].rstrip("\n")
# Output path is the first CLI arg, e.g. `notebooks/mars_showcase.ipynb`.
Path(sys.argv[1]).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", sys.argv[1], "·", len(nb["cells"]), "cells")
