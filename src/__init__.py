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

# GraphReasoning is vendored at src/vendor/graphreasoning (see its VENDORED.md).
# The former agents-stub and the find_best_fitting_node_list meta_path hook that
# lived here are no longer needed: the vendored __init__ does not import agents,
# and the MARS signature is now folded into the vendored graph_tools.py.

__version__ = "0.1.0"

