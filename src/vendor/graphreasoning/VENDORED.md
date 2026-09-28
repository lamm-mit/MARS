# Vendored: GraphReasoning

Upstream: <https://github.com/lamm-mit/GraphReasoning> (Markus J. Buehler, MIT)
Pinned commit: `f1d6d44a7d44123b7896029158ada9442c9dcc1a`
Package version at that commit: `0.2.0`

## Why this is vendored rather than pip-installed

`environment.yml` previously installed this straight from `git+https://…git`, i.e.
**unpinned master**. Two people installing on different days got different code,
which is a poor foundation for a reproducibility artifact. Upstream had also
already broken MARS once: `find_best_fitting_node_list` changed signature between
the version MARS was developed against and the public release.

Before vendoring, MARS compensated with three separate layers of import-time
monkeypatching (a `sys.meta_path` hook and an `agents` stub in `src/__init__.py`,
plus langchain/guidance shims and a `colors2Community` patch in
`data/KG_Generation/build_kg.py`). Those patches only applied to code paths that
imported `src/`, so `build_kg.py` — which does not — failed outright. A pinned
local copy replaces all of it with edits that are visible where they take effect.

## Licence note — unresolved upstream

The distributed `LICENSE` file (copied here verbatim) is the **Apache License
2.0** boilerplate with an unfilled `Copyright [yyyy] [name of copyright owner]`
placeholder, while the package metadata declares
`Classifier: License :: OSI Approved :: MIT License`. The two disagree.

Both licences permit redistribution with modification, so vendoring is fine under
either reading, and Apache-2.0 (the stricter of the two here) requires that
modifications be stated — which is what this file does. Worth confirming the
intended licence with the authors before any public release of MARS.

## Modifications

Every change is marked in-source with `MARS MODIFICATION`.

### 1. `__init__.py` — dropped the `agents` import

Upstream did `from GraphReasoning.agents import *`. `agents.py` imports
`llama_index.core`, `llama_index.embeddings.huggingface` and `guidance`, none of
which MARS uses, and none of which are in `environment.yml`. That single line made
the whole package unimportable, which is what broke `build_kg.py`. The file is
retained for reference but is no longer imported.

### 2. `graph_generation.py` — rewrote `graphPrompt()`

Upstream asked the LLM for a bare JSON array of `{node_1, node_2, edge}` and
treated `generate()`'s return value as a raw string
(`response.replace(...)`, `extract(...)`, `json.loads(...)`). MARS's `generate()`
returns a validated pydantic `KnowledgeGraph`, so the first `.replace()` raised
`AttributeError: 'KnowledgeGraph' object has no attribute 'replace'` before any
validation ran. Upstream also passed its own `system_prompt=` on every call,
which made the prompt defined in `build_kg.py` unreachable dead code.

Additionally, `build_kg.py` sets `response_format={"type": "json_object"}` while
upstream's prompt demanded a top-level JSON **array** — mutually unsatisfiable,
and the cause of the Azure structured-output failures.

The rewritten `graphPrompt()`:

- asks for `{"nodes": [{id, type}], "edges": [{source, target, relation}]}`,
  matching `build_kg.py`'s pydantic schema and OpenAI/Azure JSON-object mode
- accepts a pydantic object, a dict, or a JSON string (fenced or not), so either
  contract works
- returns upstream's shape — a list of `{node_1, node_2, edge}` dicts merged with
  `metadata` — because `graph2Df()` drops rows lacking `node_1`/`node_2`
- makes **one** LLM call per chunk instead of upstream's four sequential
  self-refinement calls; the extra passes now happen only when `repeat_refine > 0`

### 3. `graph_tools.py` — folded in `find_best_fitting_node_list`

Upstream assumes a HuggingFace tokenizer+model pair and has no
`similarity_threshold`. MARS embeds with SentenceTransformer (passing
`tokenizer=""`) and supplies a threshold from config. This was previously patched
by a `sys.meta_path` hook in `src/__init__.py`; it now lives in the function.
Both call styles are supported.

### 4. `graph_tools.py`, `graph_generation.py`, `graph_analysis.py` — langchain paths

`langchain.document_loaders` → `langchain_community.document_loaders` and
`langchain.text_splitter` → `langchain_text_splitters` (moved in langchain 0.1).
This removes the need for the `sys.modules` shims in `build_kg.py`.

### 5. Internal imports made relative

`from GraphReasoning.x import *` → `from .x import *`, so the vendored copy is
self-contained and cannot accidentally resolve against a pip-installed
`GraphReasoning` that may still be in the environment.

## Behavioural caveat

The `.graphml` files shipped in `data/MARS_Data/KGs/` were produced by the
**unmodified** upstream package. Modification 2 changes the extraction prompt, so
graphs built after this change are not guaranteed comparable to the shipped ones.
Re-run a small regression set before regenerating anything you depend on.

## Updating

Do not `pip install --upgrade`. Re-vendor deliberately: copy the new upstream
tree, re-apply the five modifications above, bump the commit hash here, and
re-run the KG regression.
