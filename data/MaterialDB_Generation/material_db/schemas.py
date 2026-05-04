"""Pydantic schemas for material database."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class LLMResponse(BaseModel):
    """Schema for LLM extraction response."""

    supplier: str = Field(..., description="Material supplier name")
    name: str = Field(..., description="Material name")
    extracted: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted material properties and data"
    )

    @field_validator("supplier", "name")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure supplier and name are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class MaterialRecord(BaseModel):
    """Schema for a material database record."""

    id: str = Field(..., description="Stable material ID slug")
    name: str = Field(..., description="Material name")
    supplier: str = Field(..., description="Supplier name")
    raw_markdown_text: str = Field(..., description="Concatenated markdown content")
    extracted: Dict[str, Any] = Field(
        default_factory=dict,
        description="LLM-extracted material data"
    )
    llm_prompt: str = Field(..., description="Exact prompt sent to LLM")
    llm_response_raw: str = Field(..., description="Raw LLM response text")
    llm_response_json: Dict[str, Any] = Field(..., description="Parsed JSON response")
    created_at: str = Field(..., description="ISO-8601 creation timestamp")
    updated_at: str = Field(..., description="ISO-8601 update timestamp")
    source_files: List[str] = Field(..., description="Relative paths of markdown files")

    @classmethod
    def create(
        cls,
        material_id: str,
        supplier: str,
        name: str,
        raw_markdown_text: str,
        extracted: Dict[str, Any],
        llm_prompt: str,
        llm_response_raw: str,
        llm_response_json: Dict[str, Any],
        source_files: List[str],
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> "MaterialRecord":
        """Create a material record with timestamps."""
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        return cls(
            id=material_id,
            name=name,
            supplier=supplier,
            raw_markdown_text=raw_markdown_text,
            extracted=extracted,
            llm_prompt=llm_prompt,
            llm_response_raw=llm_response_raw,
            llm_response_json=llm_response_json,
            created_at=created_at or now,
            updated_at=updated_at or now,
            source_files=source_files,
        )


class MaterialDatabase(BaseModel):
    """Schema for the entire material database."""

    materials: Dict[str, MaterialRecord] = Field(
        default_factory=dict,
        description="Material records keyed by material ID"
    )

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to plain dict for JSON serialization."""
        return {k: v.model_dump() for k, v in self.materials.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> "MaterialDatabase":
        """Create from plain dict."""
        materials = {
            k: MaterialRecord(**v) for k, v in data.items()
        }
        return cls(materials=materials)


class ProcessingStatus:
    """Status constants for material processing."""

    pending = "pending"
    complete = "complete"
    failed = "failed"


class MaterialState(BaseModel):
    """State tracking for a single material folder."""

    folder_path: str = Field(..., description="Relative path to material folder")
    material_id: Optional[str] = Field(None, description="Assigned material ID")
    file_fingerprints: Dict[str, str] = Field(
        default_factory=dict,
        description="SHA256 fingerprints per markdown file"
    )
    combined_fingerprint: Optional[str] = Field(
        None,
        description="Combined fingerprint for all files"
    )
    status: str = Field(
        default=ProcessingStatus.pending,
        description="Processing status"
    )
    last_error: Optional[str] = Field(None, description="Last error message if failed")
    last_processed: Optional[str] = Field(
        None,
        description="ISO-8601 timestamp of last processing"
    )


class ProcessingState(BaseModel):
    """Overall processing state tracking."""

    materials: Dict[str, MaterialState] = Field(
        default_factory=dict,
        description="State per material folder path"
    )
    folder_to_id: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from folder path to material ID"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for JSON serialization."""
        return {
            "materials": {k: v.model_dump() for k, v in self.materials.items()},
            "folder_to_id": self.folder_to_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingState":
        """Create from plain dict."""
        materials = {
            k: MaterialState(**v) for k, v in data.get("materials", {}).items()
        }
        folder_to_id = data.get("folder_to_id", {})
        return cls(materials=materials, folder_to_id=folder_to_id)
