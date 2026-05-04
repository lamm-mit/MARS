"""Material processing logic with LLM extraction."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging

from .llm_client import LLMClient
from .schemas import LLMResponse, MaterialRecord, ProcessingStatus
from .database import Database
from .utils import (
    find_all_markdown_files,
    concatenate_markdown_files,
    slugify,
    estimate_tokens,
    split_text_into_chunks,
    deep_merge_dicts,
)
from .validation import validate_and_repair_llm_response

logger = logging.getLogger(__name__)


class MaterialProcessor:
    """Processes material folders and extracts data using LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        base_path: Path,
        max_repair_attempts: int = 3,
        re_prompt_on_failure: bool = False,
        max_input_tokens: int = 35000,
        database: Optional[Database] = None,
    ):
        self.llm_client = llm_client
        self.base_path = base_path
        self.max_repair_attempts = max_repair_attempts
        self.re_prompt_on_failure = re_prompt_on_failure
        self.max_input_tokens = max_input_tokens
        self.database = database

    def build_system_prompt(self) -> str:
        return """You are an expert in material science and technical documentation. Your task is to extract structured information from material datasheets and technical documents.

You must extract:
1. supplier: The name of the material supplier/manufacturer
2. name: The name/product identifier of the material
3. extracted: A JSON object containing relevant material properties, specifications, and data

Important:
- Extract values verbatim (no unit conversion)
- Store all relevant properties in the extracted object
- Output ONLY valid JSON, no commentary or markdown formatting
- Be thorough and extract all available technical information"""

    def build_user_prompt(self, markdown_text: str) -> str:
        return """Extract material information from the following markdown content.

Use this exact schema:

{{
  "supplier": "...",
  "name": "...",
  "extracted": {{ ... }}
}}

Markdown content:
```markdown
{markdown_text}
```

Output the JSON now:""".format(markdown_text=markdown_text)

    def extract_with_llm(self, markdown_text: str) -> Tuple[str, str, Dict[str, Any]]:
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(markdown_text)
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

        for attempt in range(self.max_repair_attempts + 1):
            try:
                raw_response = self.llm_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                )
                logger.debug(f"LLM raw response (attempt {attempt + 1}): {raw_response[:500]}")
                parsed_json = validate_and_repair_llm_response(raw_response)
                llm_response = LLMResponse(**parsed_json)
                return full_prompt, raw_response, llm_response.model_dump()
            except Exception as e:
                logger.warning(f"LLM extraction attempt {attempt + 1} failed: {e}")
                if attempt < self.max_repair_attempts:
                    if self.re_prompt_on_failure:
                        user_prompt += f"\n\nPrevious attempt failed: {e}\nPlease fix the JSON output."
                        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
                    continue
                raise ValueError(
                    f"Failed to extract valid data after {self.max_repair_attempts + 1} attempts: {e}"
                )

        raise ValueError("Unexpected error in extraction loop")

    def process_material_folder(
        self,
        folder_path: Path,
        material_id: Optional[str] = None,
    ) -> MaterialRecord:
        logger.info(f"Processing material folder: {folder_path}")

        md_files, mmd_files = find_all_markdown_files(folder_path)
        if not md_files and not mmd_files:
            raise ValueError(f"No markdown files (.md or .mmd) found in {folder_path}")

        file_sets = [(mmd_files, ".mmd"), (md_files, ".md")]

        last_error = None
        llm_prompt = llm_response_raw = llm_response_json = raw_markdown_text = source_files = None

        for files, file_type in file_sets:
            if not files:
                continue

            logger.debug(f"Trying {len(files)} {file_type} file(s)")

            try:
                system_prompt = self.build_system_prompt()
                wrapper_tokens = estimate_tokens(system_prompt + self.build_user_prompt(""))
                available_tokens = int((self.max_input_tokens - wrapper_tokens) * 0.8)

                raw_markdown_text, source_files = concatenate_markdown_files(
                    files, self.base_path, max_tokens=None
                )

                estimated_tokens = estimate_tokens(raw_markdown_text)

                if estimated_tokens > available_tokens:
                    logger.info(
                        f"Content ({estimated_tokens} est. tokens) exceeds limit ({available_tokens}). "
                        f"Splitting into chunks..."
                    )
                    chunks = split_text_into_chunks(raw_markdown_text, available_tokens)
                    logger.info(f"Split into {len(chunks)} chunks")

                    llm_prompt, llm_response_raw, llm_response_json = self.extract_with_llm(chunks[0])
                    supplier = llm_response_json["supplier"]
                    name = llm_response_json["name"]
                    extracted = llm_response_json["extracted"]

                    if material_id is None:
                        material_id = slugify(f"{supplier}-{name}")

                    material = MaterialRecord.create(
                        material_id=material_id,
                        supplier=supplier,
                        name=name,
                        raw_markdown_text=chunks[0],
                        extracted=extracted,
                        llm_prompt=llm_prompt,
                        llm_response_raw=llm_response_raw,
                        llm_response_json=llm_response_json,
                        source_files=source_files,
                    )

                    if self.database:
                        self.database.upsert_material(material)
                        self.database.save_database()

                    if len(chunks) > 1 and self.database:
                        for chunk_idx, chunk in enumerate(chunks[1:], start=2):
                            logger.info(f"Processing chunk {chunk_idx}/{len(chunks)}...")
                            self._append_chunk_to_material(material, chunk, chunk_idx)

                    material.raw_markdown_text = raw_markdown_text
                    logger.info(f"Successfully processed {len(chunks)} chunks using {file_type} files")
                    return material
                else:
                    llm_prompt, llm_response_raw, llm_response_json = self.extract_with_llm(raw_markdown_text)
                    logger.info(f"Successfully processed using {file_type} files")
                    break

            except Exception as e:
                last_error = e
                logger.warning(f"Processing with {file_type} files failed: {e}")
                if file_type == ".mmd" and md_files:
                    logger.info("Falling back to .md files...")
                    continue
                else:
                    raise

        if llm_response_json is None:
            if last_error:
                raise ValueError(
                    f"Failed to process material folder with both .mmd and .md files: {last_error}"
                ) from last_error
            raise ValueError("No markdown files could be processed")

        supplier = llm_response_json["supplier"]
        name = llm_response_json["name"]
        extracted = llm_response_json["extracted"]

        if material_id is None:
            material_id = slugify(f"{supplier}-{name}")

        logger.info(f"Extracted: supplier={supplier}, name={name}, id={material_id}")

        return MaterialRecord.create(
            material_id=material_id,
            supplier=supplier,
            name=name,
            raw_markdown_text=raw_markdown_text,
            extracted=extracted,
            llm_prompt=llm_prompt,
            llm_response_raw=llm_response_raw,
            llm_response_json=llm_response_json,
            source_files=source_files,
        )

    def _append_chunk_to_material(self, material: MaterialRecord, chunk_text: str, chunk_number: int) -> None:
        if not self.database:
            raise ValueError("Database instance required for chunking support")

        existing = self.database.get_material(material.id)
        if existing:
            material = existing

        llm_prompt, llm_response_raw, llm_response_json = self.extract_with_llm(chunk_text)

        chunk_supplier = llm_response_json.get("supplier", "")
        chunk_name = llm_response_json.get("name", "")

        if chunk_supplier and chunk_supplier != material.supplier:
            logger.warning(f"Chunk {chunk_number}: Supplier mismatch ('{chunk_supplier}'). Using original.")
        if chunk_name and chunk_name != material.name:
            logger.warning(f"Chunk {chunk_number}: Name mismatch ('{chunk_name}'). Using original.")

        chunk_extracted = llm_response_json.get("extracted", {})
        material.extracted = deep_merge_dicts(material.extracted, chunk_extracted)

        material.llm_prompt += f"\n\n--- Chunk {chunk_number} ---\n\n{llm_prompt}"
        material.llm_response_raw += f"\n\n--- Chunk {chunk_number} ---\n\n{llm_response_raw}"
        material.llm_response_json = {
            "supplier": material.supplier,
            "name": material.name,
            "extracted": material.extracted,
        }

        from datetime import datetime, timezone
        material.updated_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        self.database.upsert_material(material)
        self.database.save_database()

        logger.info(f"Chunk {chunk_number} appended to material {material.id}")
