#!/usr/bin/env python3
"""
Build the material database from InternalMaterials markdown files.

Reads:  MARS_Data/MDs_paper/InternalMaterials/
Writes: MARS_Data/MaterialDB_paper/internal_material_database.json

Usage:
    python build_materialdb.py
    python build_materialdb.py path/to/config.yaml   # custom config
"""

import sys
from pathlib import Path

# Resolve config relative to this script, not the caller's CWD
_HERE = Path(__file__).parent

sys.path.insert(0, str(_HERE))

from material_db.main import main

if __name__ == "__main__":
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml"
    main(config_path)
