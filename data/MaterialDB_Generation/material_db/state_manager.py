"""State management for resumable processing."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from .schemas import MaterialState, ProcessingState, ProcessingStatus
from .utils import compute_file_fingerprint, find_all_markdown_files, compute_combined_fingerprint


class StateManager:
    """Manages processing state persistence."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> ProcessingState:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                return ProcessingState.from_dict(data)
            except Exception:
                return ProcessingState()
        return ProcessingState()

    def save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def get_material_state(self, folder_path: str) -> Optional[MaterialState]:
        return self.state.materials.get(folder_path)

    def update_material_state(
        self,
        folder_path: str,
        material_id: Optional[str] = None,
        file_fingerprints: Optional[Dict[str, str]] = None,
        combined_fingerprint: Optional[str] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
    ) -> MaterialState:
        if folder_path not in self.state.materials:
            self.state.materials[folder_path] = MaterialState(folder_path=folder_path)

        state = self.state.materials[folder_path]

        if material_id is not None:
            state.material_id = material_id
            self.state.folder_to_id[folder_path] = material_id
        if file_fingerprints is not None:
            state.file_fingerprints = file_fingerprints
        if combined_fingerprint is not None:
            state.combined_fingerprint = combined_fingerprint
        if status is not None:
            state.status = status
        if error is not None:
            state.last_error = error

        state.last_processed = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        return state

    def get_material_id(self, folder_path: str) -> Optional[str]:
        return self.state.folder_to_id.get(folder_path)

    def compute_current_fingerprints(
        self, folder_path: Path, base_path: Path
    ) -> Tuple[Dict[str, str], str]:
        md_files, mmd_files = find_all_markdown_files(folder_path)
        files_to_use = mmd_files if mmd_files else md_files

        file_fingerprints = {}
        for md_file in files_to_use:
            rel_path = str(md_file.relative_to(base_path))
            file_fingerprints[rel_path] = compute_file_fingerprint(md_file)

        fingerprints_list = list(file_fingerprints.values())
        combined = compute_combined_fingerprint(fingerprints_list) if fingerprints_list else ""
        return file_fingerprints, combined

    def should_process(self, folder_path: str, current_combined_fingerprint: str) -> bool:
        state = self.get_material_state(folder_path)
        if state is None:
            return True
        if state.status == ProcessingStatus.complete:
            if state.combined_fingerprint == current_combined_fingerprint:
                return False
        return True
