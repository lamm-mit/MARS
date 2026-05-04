"""Utility functions for material database processing."""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unicodedata


def slugify(text: str) -> str:
    """
    Create a stable slug from text (lowercase, dashes, alphanumeric only).

    Example: "Victrex-PEEK 650G" -> "victrex-peek-650g"
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text if unicodedata.category(c) != "Mn"
    ).lower()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^\w\-]", "", text)
    text = text.replace("®", "").replace("™", "").replace("©", "")
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def compute_file_fingerprint(file_path: Path) -> str:
    """Compute SHA256 fingerprint of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_combined_fingerprint(fingerprints: List[str]) -> str:
    """Compute combined fingerprint from multiple file fingerprints."""
    combined = "|".join(sorted(fingerprints))
    return hashlib.sha256(combined.encode()).hexdigest()


def find_markdown_files(folder_path: Path, prefer_md: bool = False) -> List[Path]:
    """Find markdown files in a folder, preferring .mmd over .md by default."""
    md_files = []
    mmd_files = []
    for path in folder_path.rglob("*"):
        if path.is_file():
            if path.suffix.lower() == '.md':
                md_files.append(path)
            elif path.suffix.lower() == '.mmd':
                mmd_files.append(path)
    if prefer_md:
        return sorted(md_files) if md_files else sorted(mmd_files)
    else:
        return sorted(mmd_files) if mmd_files else sorted(md_files)


def find_all_markdown_files(folder_path: Path) -> Tuple[List[Path], List[Path]]:
    """Find all markdown files, returned separately as (.md files, .mmd files)."""
    md_files = []
    mmd_files = []
    for path in folder_path.rglob("*"):
        if path.is_file():
            if path.suffix.lower() == '.md':
                md_files.append(path)
            elif path.suffix.lower() == '.mmd':
                mmd_files.append(path)
    return sorted(md_files), sorted(mmd_files)


def read_markdown_file(file_path: Path) -> str:
    """Read markdown file content (.md or .mmd)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


def estimate_tokens(text: str) -> int:
    """Estimate token count (conservative: ~2 chars per token)."""
    return len(text) // 2


def truncate_text_to_tokens(text: str, max_tokens: int, truncate_at: str = "\n\n") -> str:
    """Truncate text to fit within token limit, breaking at natural boundaries."""
    if estimate_tokens(text) <= max_tokens:
        return text
    target_chars = max_tokens * 2
    if len(text) <= target_chars:
        return text
    truncated = text[:target_chars]
    last_break = truncated.rfind(truncate_at)
    if last_break > target_chars * 0.8:
        return truncated[:last_break] + "\n\n[Content truncated due to length...]"
    return truncated + "\n\n[Content truncated due to length...]"


def split_text_into_chunks(text: str, max_tokens_per_chunk: int, truncate_at: str = "\n\n") -> List[str]:
    """Split text into chunks that fit within token limit."""
    if estimate_tokens(text) <= max_tokens_per_chunk:
        return [text]
    chunks = []
    remaining_text = text
    target_chars = int(max_tokens_per_chunk * 0.8) * 2
    while remaining_text:
        if len(remaining_text) <= target_chars:
            chunks.append(remaining_text)
            break
        chunk = remaining_text[:target_chars]
        last_break = chunk.rfind(truncate_at)
        if last_break > target_chars * 0.8:
            chunks.append(chunk[:last_break])
            remaining_text = remaining_text[last_break + len(truncate_at):].lstrip()
        else:
            chunks.append(chunk)
            remaining_text = remaining_text[target_chars:]
    return chunks


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dicts; dict2 values override dict1. Lists are concatenated."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_dicts(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def concatenate_markdown_files(
    file_paths: List[Path],
    base_path: Path,
    max_tokens: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Concatenate multiple markdown files into a single string."""
    texts = []
    relative_paths = []
    for file_path in file_paths:
        relative_path = str(file_path.relative_to(base_path))
        relative_paths.append(relative_path)
        content = read_markdown_file(file_path)
        texts.append(f"<!-- File: {relative_path} -->\n\n{content}\n\n")
    combined = "\n".join(texts)
    if max_tokens is not None:
        combined = truncate_text_to_tokens(combined, max_tokens)
    return combined, relative_paths
