"""Database persistence for material records."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .schemas import MaterialDatabase, MaterialRecord


class Database:
    """Manages material database persistence."""

    def __init__(self, db_file: Path):
        self.db_file = db_file
        self.db = self._load_database()

    def _load_database(self) -> MaterialDatabase:
        if self.db_file.exists():
            try:
                with open(self.db_file, "r") as f:
                    data = json.load(f)
                return MaterialDatabase.from_dict(data)
            except Exception:
                return MaterialDatabase()
        return MaterialDatabase()

    def save_database(self) -> None:
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_file, "w") as f:
            json.dump(self.db.to_dict(), f, indent=2)

    def get_material(self, material_id: str) -> Optional[MaterialRecord]:
        return self.db.materials.get(material_id)

    def upsert_material(self, material: MaterialRecord) -> None:
        existing = self.db.materials.get(material.id)
        if existing:
            material.created_at = existing.created_at
            material.updated_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        self.db.materials[material.id] = material

    def get_all_materials(self) -> Dict[str, MaterialRecord]:
        return self.db.materials.copy()
