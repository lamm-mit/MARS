"""Configuration management via YAML."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file: Optional[str] = Field(None, description="Optional log file path")


class LLMConfig(BaseModel):
    """LLM configuration for local gpt-oss server."""

    model: str = Field(..., description="Model name (e.g., 'gpt-oss-20b')")
    base_url: str = Field(..., description="Base URL for local server (e.g., 'http://localhost:8081/v1')")
    max_tokens: int = Field(default=4000, description="Maximum tokens in response (leave room for input context)")
    max_input_tokens: int = Field(default=35000, description="Maximum tokens for input (system + user prompt)")
    temperature: float = Field(default=0.0, description="Sampling temperature (0.0 for deterministic)")
    timeout: int = Field(default=1200, description="Request timeout in seconds")


class ValidationConfig(BaseModel):
    """Validation and repair configuration."""

    max_repair_attempts: int = Field(default=3, description="Max repair attempts")
    re_prompt_on_failure: bool = Field(
        default=False,
        description="Re-prompt LLM on validation failure"
    )


class Config(BaseModel):
    """Main configuration."""

    input_root: str = Field(
        default="supplier_datasheet",
        description="Root directory for material folders"
    )
    database_file: str = Field(
        default="material_database.json",
        description="Output database JSON filename"
    )
    state_file: str = Field(
        default="processing_state.json",
        description="Processing state JSON filename"
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
