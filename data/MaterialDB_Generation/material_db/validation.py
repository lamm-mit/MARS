"""JSON validation and repair utilities."""

import json
import re
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract first complete JSON object from text."""
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx >= 0:
                return text[start_idx:i + 1]
    return None


def repair_json_string(json_str: str) -> Optional[str]:
    """Attempt to repair common JSON issues (code fences, trailing commas, single quotes)."""
    json_str = re.sub(r"^```(?:json)?\s*\n", "", json_str, flags=re.MULTILINE)
    json_str = re.sub(r"\n```\s*$", "", json_str, flags=re.MULTILINE)
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
    return json_str.strip()


def validate_and_repair_llm_response(raw_response: str) -> Dict[str, Any]:
    """
    Validate and repair LLM JSON response.

    Handles standard format {"supplier": ..., "name": ..., "extracted": {...}},
    wrapped formats, and list formats.

    Raises:
        ValueError: If JSON cannot be extracted or repaired.
    """
    parsed = None

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    if parsed is None:
        json_str = extract_json_from_text(raw_response)
        if json_str:
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                pass
            if parsed is None:
                repaired = repair_json_string(json_str)
                if repaired:
                    try:
                        parsed = json.loads(repaired)
                    except json.JSONDecodeError:
                        pass

    if parsed is None:
        repaired = repair_json_string(raw_response)
        if repaired:
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                pass

    if parsed is None:
        first_brace = raw_response.find("{")
        if first_brace >= 0:
            try:
                parsed = json.loads(raw_response[first_brace:])
            except json.JSONDecodeError:
                pass

    if parsed is None:
        raise ValueError(
            f"Could not extract valid JSON from LLM response. "
            f"Response preview: {raw_response[:200]}"
        )

    if isinstance(parsed, dict):
        if "supplier" in parsed and "name" in parsed:
            if "extracted" not in parsed:
                parsed["extracted"] = {}
            return parsed
        if "materials" in parsed and isinstance(parsed["materials"], list):
            if len(parsed["materials"]) > 0:
                first = parsed["materials"][0]
                if isinstance(first, dict) and "supplier" in first and "name" in first:
                    if "extracted" not in first:
                        first["extracted"] = {}
                    return first
        for key, value in parsed.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and "supplier" in value[0] and "name" in value[0]:
                    result = value[0].copy()
                    if "extracted" not in result:
                        result["extracted"] = {}
                    return result

    if isinstance(parsed, list) and len(parsed) > 0:
        if isinstance(parsed[0], dict) and "supplier" in parsed[0] and "name" in parsed[0]:
            result = parsed[0].copy()
            if "extracted" not in result:
                result["extracted"] = {}
            return result

    raise ValueError(
        f"Unexpected JSON format. Expected {{supplier, name, extracted}}, "
        f"got: {type(parsed).__name__}. Preview: {raw_response[:200]}"
    )
