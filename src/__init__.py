"""Material Discovery Package - Two-System Pipeline for Material Substitution"""

# Conda envs can have a conda-installed transformers whose dependency_versions_check
# rejects huggingface-hub>=1.0, even though the installed hub is fully API-compatible.
# Pre-loading a no-op module skips the check without touching the conda environment.
import sys as _sys
import types as _types
if "transformers.dependency_versions_check" not in _sys.modules:
    _dvc = _types.ModuleType("transformers.dependency_versions_check")
    _dvc.dep_version_check = lambda *a, **kw: None  # no-op; deepspeed.py imports this
    _sys.modules["transformers.dependency_versions_check"] = _dvc

# huggingface_hub >= 1.0 raises RemoteEntryNotFoundError (404) when a model repo
# has no additional_chat_templates folder. Transformers 4.x expects an empty list.
# tokenization_utils_base.py binds list_repo_templates at module load time via a
# `from .utils.hub import` statement, so we must patch its namespace at the exact
# moment the module is imported. A sys.meta_path hook achieves this reliably.
class _PatchListRepoTemplates:
    def find_module(self, name, path=None):
        return self if name == "transformers.tokenization_utils_base" else None

    def load_module(self, name):
        _sys.meta_path.remove(self)
        import importlib as _il
        mod = _il.import_module(name)
        _orig = getattr(mod, "list_repo_templates", None)
        if _orig is not None:
            def _safe(*a, **kw):
                try:
                    return _orig(*a, **kw)
                except Exception:
                    return []
            mod.list_repo_templates = _safe
        return mod

_sys.meta_path.insert(0, _PatchListRepoTemplates())

# GraphReasoning.__init__ does `from GraphReasoning.agents import *`, but agents.py
# pulls in many optional packages (guidance, llama_index, etc.) with brittle version
# requirements that MARS does not need. Pre-stubbing the module prevents agents.py
# from loading entirely. MARS only uses GraphReasoning's graph utilities
# (find_best_fitting_node_list, load_embeddings) which are unaffected.
if "GraphReasoning.agents" not in _sys.modules:
    _sys.modules["GraphReasoning.agents"] = _types.ModuleType("GraphReasoning.agents")

# GraphReasoning's find_best_fitting_node_list changed API between the version used
# when MARS was developed and the current public release. Specifically:
#   - current version: HuggingFace tokenizer+model style, no similarity_threshold
#   - MARS usage: SentenceTransformer (tokenizer=""), with similarity_threshold kwarg
# Intercept the import of GraphReasoning.graph_tools and replace the function.
class _PatchGraphReasoning:
    def find_module(self, name, path=None):
        return self if name == "GraphReasoning.graph_tools" else None

    def load_module(self, name):
        _sys.meta_path.remove(self)
        import importlib as _il
        mod = _il.import_module(name)

        import heapq as _hq
        import numpy as _np

        def _find_best_fitting_node_list(
            keyword, embeddings, tokenizer, model,
            N_samples=5, similarity_threshold=0.0, **kwargs
        ):
            from scipy.spatial.distance import cosine as _cos
            # Embed the keyword — SentenceTransformer when tokenizer is falsy
            if not tokenizer:
                kw = model.encode(keyword, show_progress_bar=False)
            else:
                inputs = tokenizer(keyword, return_tensors="pt")
                outputs = model(**inputs)
                kw = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            kw = _np.array(kw).flatten()

            heap = []
            _hq.heapify(heap)
            for node, emb in embeddings.items():
                emb = _np.array(emb).flatten()
                try:
                    sim = float(1 - _cos(kw, emb))
                except Exception:
                    continue
                if sim < similarity_threshold:
                    continue
                if len(heap) < N_samples:
                    _hq.heappush(heap, (sim, node))
                elif sim > heap[0][0]:
                    _hq.heapreplace(heap, (sim, node))

            return [(node, sim) for sim, node in sorted(heap, key=lambda x: -x[0])]

        mod.find_best_fitting_node_list = _find_best_fitting_node_list
        return mod

_sys.meta_path.insert(0, _PatchGraphReasoning())

__version__ = "0.1.0"

