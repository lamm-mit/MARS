#!/usr/bin/env python3
"""
Build a knowledge graph from a corpus of markdown files.

For each document, either input_dir/*.md (flat corpus) or input_dir/{doc_id}/{doc_id}.md
(legacy). Then:
  1. Extract a per-document KG via LLM → save as .graphml in graph_dir/
  2. Merge into a growing global graph (checkpointed in merged_dir/)
  3. Embed all nodes with nomic-embed-text-v1.5

The final graph and embeddings are written to output_dir/.

Resumable: documents with an existing .graphml in graph_dir/ are skipped.
Merge resumes from the last valid checkpoint in merged_dir/.

NOTE: This script is intentionally single-threaded to keep the pipeline simple
and fully reproducible. The original notebook (make_KG_PFAS.ipynb) supports
parallel extraction across multiple workers — see that file if parallelism is needed.

Usage:
    python build_kg.py
    python build_kg.py path/to/config.yaml
"""

import glob
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import networkx as nx
import torch
import yaml
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# GraphReasoning compatibility shims
# GraphReasoning uses old LangChain APIs moved in LangChain >= 0.1:
#   langchain.document_loaders → langchain_community.document_loaders
#   langchain.text_splitter    → langchain_text_splitters
# guidance.models.Chat was removed in guidance >= 0.2.
# Patch all three before importing so GraphReasoning loads without a downgrade.
# ---------------------------------------------------------------------------
import sys as _sys
try:
    import langchain_community.document_loaders as _dl
    _sys.modules.setdefault('langchain.document_loaders', _dl)
except ImportError:
    pass
try:
    import langchain_text_splitters as _ts
    _sys.modules.setdefault('langchain.text_splitter', _ts)
except ImportError:
    pass
try:
    import guidance.models as _gm
    if not hasattr(_gm, 'Chat'):
        class _Chat:
            pass
        _gm.Chat = _Chat
except ImportError:
    pass

from GraphReasoning import (
    make_graph_from_text,
    add_new_subgraph_from_text,
    generate_node_embeddings,
    load_embeddings,
    save_embeddings,
)


def _patch_graphreasoning_colors2community() -> None:
    """graph_tools.graph_Louvain() calls colors2Community(); upstream has the body commented out."""

    try:
        import random

        import pandas as pd
        import seaborn as sns

        import GraphReasoning.graph_tools as gr_gt
    except ImportError:
        return

    if getattr(gr_gt, "colors2Community", None) is not None:
        return

    def colors2Community(communities):  # noqa: N802 — match upstream GraphReasoning name
        plat = getattr(gr_gt, "palette", "hls")
        n = max(1, len(communities))
        palette_colors = sns.color_palette(plat, n).as_hex()
        colors_pool = list(palette_colors)
        random.shuffle(colors_pool)
        rows = []
        group = 0
        for community in communities:
            color = colors_pool.pop()
            group += 1
            for node in community:
                rows.append({"node": node, "color": color, "group": group})
        return pd.DataFrame(rows)

    gr_gt.colors2Community = colors2Community


_patch_graphreasoning_colors2community()


_HERE = Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------

class Node(BaseModel):
    id: str
    type: str

class Edge(BaseModel):
    source: str
    target: str
    relation: str

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a scientific assistant extracting knowledge graphs from text.\n"
    "Return a JSON object with exactly two fields: \"nodes\" and \"edges\".\n"
    "Each node must have \"id\" (string) and \"type\" (string).\n"
    "Each edge must have \"source\" (string), \"target\" (string), and \"relation\" (string).\n"
    "Example: {\"nodes\": [{\"id\": \"PEEK\", \"type\": \"material\"}], "
    "\"edges\": [{\"source\": \"PEEK\", \"target\": \"thermoplastic\", \"relation\": \"is_a\"}]}"
)


def make_generate_fn(client: OpenAI, model_name: str, max_tokens: int, temperature: float):
    """Return a generate() function compatible with GraphReasoning.

    Uses OpenAI JSON mode and validates the response against the KnowledgeGraph
    pydantic schema — no instructor required.
    """

    def generate(
        system_prompt: str = SYSTEM_PROMPT,
        prompt: str = "",
        temperature: float = temperature,
        max_tokens: int = max_tokens,
        response_model=KnowledgeGraph,
    ):
        messages = (
            [{"role": "user", "content": prompt}]
            if system_prompt is None
            else [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            return response_model(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Could not parse LLM response as KnowledgeGraph: {e}")
            return response_model(nodes=[], edges=[])

    return generate


def make_generate_figure_fn():
    """Stub — local LLM is text-only; image nodes are not used."""
    def generate_figure(image, system_prompt=None, prompt="", temperature=0):
        return KnowledgeGraph(nodes=[], edges=[])
    return generate_figure


# ---------------------------------------------------------------------------
# Checkpoint helpers (adapted from make_KG_PFAS.ipynb)
# ---------------------------------------------------------------------------

def find_last_merged_index(merged_dir: Path) -> int:
    """Return the index of the last valid merged checkpoint, or -1 if none."""
    pattern = str(merged_dir / "*_integrated.graphml")
    files = glob.glob(pattern)
    if not files:
        return -1

    def extract_index(path: str) -> Optional[int]:
        name = Path(path).name
        m = re.match(r'^(\d+)_', name)
        return int(m.group(1)) if m else None

    indexed = [(extract_index(f), f) for f in files]
    indexed = [(idx, f) for idx, f in indexed if idx is not None]
    if not indexed:
        return -1

    indexed.sort(key=lambda x: x[0], reverse=True)

    for idx, path in indexed:
        try:
            nx.read_graphml(path)
            return idx
        except Exception:
            logger.warning(f"Corrupt checkpoint {path}, removing and rolling back.")
            os.remove(path)

    return -1


def collect_documents(input_dir: Path) -> List[str]:
    """Markdown paths under input_dir: all *.md files at the root, plus legacy
    one-folder-per-doc layouts where basename matches the parent folder name."""
    flat = [
        str(p.resolve())
        for p in sorted(input_dir.glob("*.md"))
        if p.is_file()
    ]
    nested = [
        str((sub / f"{sub.name}.md").resolve())
        for sub in sorted(input_dir.iterdir(), key=lambda p: p.name)
        if sub.is_dir() and (sub / f"{sub.name}.md").is_file()
    ]
    return sorted(set(flat + nested))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    paths = cfg['paths']
    input_dir  = (_HERE / paths['input_dir']).resolve()
    graph_dir  = (_HERE / paths['graph_dir']).resolve()
    merged_dir = (_HERE / paths['merged_dir']).resolve()
    output_dir = (_HERE / paths['output_dir']).resolve()

    graph_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = cfg['llm']
    emb_cfg = cfg['embeddings']
    kg_cfg  = cfg['kg']

    # --- LLM client ---
    provider = llm_cfg.get('provider', 'local')
    if provider == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("provider is 'openai' but OPENAI_API_KEY env var is not set")
        client = OpenAI(api_key=api_key)
        logger.info(f"Using OpenAI API (model: {llm_cfg['model_name']})")
    else:
        client = OpenAI(
            base_url=llm_cfg['base_url'],
            api_key=llm_cfg['api_key'],
        )
        logger.info(f"Using local LLM at {llm_cfg['base_url']} (model: {llm_cfg['model_name']})")

    generate        = make_generate_fn(client, llm_cfg['model_name'], llm_cfg['max_tokens'], llm_cfg['temperature'])
    generate_figure = make_generate_figure_fn()

    # --- Embedding model ---
    device = emb_cfg.get('device', 'auto')
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda:0' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    logger.info(f"Loading embedding model: {emb_cfg['model_name']} on {device}")
    embedding_model = SentenceTransformer(
        emb_cfg['model_name'], trust_remote_code=True, device=device
    )
    embedding_tokenizer = ''

    # --- Load or initialise global graph + embeddings ---
    embedding_file    = graph_dir / paths['embedding_file']
    current_merged_i  = find_last_merged_index(merged_dir)

    G = nx.DiGraph()
    if current_merged_i >= 0:
        checkpoint = sorted(glob.glob(str(merged_dir / f"{current_merged_i}_*_integrated.graphml")))
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint[0]}")
            G = nx.read_graphml(checkpoint[0])

    with torch.no_grad():
        if embedding_file.exists() and current_merged_i >= 0:
            logger.info(f"Loading existing node embeddings: {embedding_file}")
            node_embeddings = load_embeddings(str(embedding_file))
        else:
            logger.info("Initialising empty node embeddings")
            node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model)

    # --- Collect document list ---
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    doc_list = collect_documents(input_dir)
    logger.info(f"Found {len(doc_list)} documents in {input_dir}")

    # --- Main loop ---
    with torch.no_grad():
        for i, doc in enumerate(doc_list):
            if i <= current_merged_i:
                continue

            title      = Path(doc).stem
            graph_root = f"{i}_{title[:min(100, len(title))]}"
            doc_graph  = graph_dir / f"{graph_root}.graphml"

            # Phase 1: extract per-document KG if not already done
            txt = Path(doc).read_text(encoding='utf-8')
            while not doc_graph.exists():
                logger.info(f"Extracting KG [{i}]: {title}")
                try:
                    now = datetime.now()
                    _, doc_graph_path, _, _, _ = make_graph_from_text(
                        txt,
                        generate,
                        generate_figure,
                        image_list='',
                        graph_root=graph_root,
                        do_distill=False,
                        chunk_size=kg_cfg['chunk_size'],
                        chunk_overlap=kg_cfg['chunk_overlap'],
                        repeat_refine=0,
                        verbatim=False,
                        data_dir=str(graph_dir),
                        save_PDF=False,
                    )
                    doc_graph = Path(doc_graph_path)
                    logger.info(f"  Done in {datetime.now() - now}")
                except Exception as e:
                    logger.warning(f"  Error: {e}. Retrying in 60 s.")
                    time.sleep(60)

            # Phase 2: merge into global graph
            do_simplify = (i % kg_cfg['simplify_every_n'] == 0)
            merged_path = merged_dir / f"{graph_root}_integrated.graphml"

            if merged_path.exists():
                G = nx.read_graphml(str(merged_path))
                logger.info(f"Loaded existing merged checkpoint: {merged_path}")
                continue

            logger.info(f"Merging [{i}]: {title}")
            now = datetime.now()
            try:
                current_graph = nx.read_graphml(str(doc_graph))
                nx.set_edge_attributes(current_graph, title, "DOI")
            except Exception as e:
                logger.warning(f"  Cannot read {doc_graph}: {e}. Skipping.")
                continue

            _, G, _, node_embeddings, _ = add_new_subgraph_from_text(
                txt='',
                node_embeddings=node_embeddings,
                tokenizer=embedding_tokenizer,
                model=embedding_model,
                original_graph=G,
                data_dir_output=str(merged_dir),
                graph_root=graph_root,
                do_simplify_graph=do_simplify,
                size_threshold=kg_cfg['simplify_size_threshold'] if do_simplify else 0,
                do_update_node_embeddings=do_simplify,
                repeat_refine=0,
                similarity_threshold=kg_cfg['similarity_threshold'],
                do_Louvain_on_new_graph=do_simplify,
                return_only_giant_component=False,
                save_common_graph=False,
                G_to_add=current_graph,
                graph_GraphML_to_add=None,
                verbatim=False,
            )
            save_embeddings(node_embeddings, str(embedding_file))
            logger.info(f"  Merged in {datetime.now() - now}  |  Graph: {G}")

    # --- Save final outputs ---
    final_graph_path = output_dir / paths['graph_output_file']
    final_emb_path   = output_dir / paths['embedding_file']

    logger.info(f"Saving final graph → {final_graph_path}")
    nx.write_graphml(G, str(final_graph_path))

    logger.info(f"Saving final embeddings → {final_emb_path}")
    save_embeddings(node_embeddings, str(final_emb_path))

    print("\n" + "=" * 60)
    print("KG Build Summary")
    print("=" * 60)
    print(f"  Documents processed : {len(doc_list)}")
    print(f"  Nodes               : {G.number_of_nodes()}")
    print(f"  Edges               : {G.number_of_edges()}")
    print(f"  Graph output        : {final_graph_path}")
    print(f"  Embeddings output   : {final_emb_path}")
    print("=" * 60)


if __name__ == '__main__':
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / 'config.yaml'
    main(config_path)
