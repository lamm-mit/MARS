#!/usr/bin/env python3
"""
Build a ChromaDB collection from a directory of markdown files.

Each subdirectory of input_root is treated as a source group. All .md and .mmd
files within it are chunked and embedded, then stored in a ChromaDB collection.

Usage:
    python build_chromadb.py                        # uses config.yaml
    python build_chromadb.py path/to/config.yaml    # custom config
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Union

import chromadb
import numpy as np
import numpy.typing as npt
import tiktoken
import yaml
from chromadb.api.types import EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

_HERE = Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class TransformerEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding function using sentence-transformers.

    Adapted from textbooks2chromaDb/ingest_to_chromadb.py.
    """

    def __init__(self, embedding_model, batch_size: int = 32):
        try:
            import torch
            self._torch = torch
        except ImportError:
            raise ValueError("torch is not installed. Run: pip install torch")
        self._model = embedding_model
        self._batch_size = batch_size

    @staticmethod
    def _normalize(vector: npt.NDArray) -> npt.NDArray:
        """L2-normalize to unit length for cosine similarity."""
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def __call__(self, input: Union[str, List[str]]) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        all_embeddings = []
        for i in range(0, len(input), self._batch_size):
            batch = input[i:i + self._batch_size]
            all_embeddings.append(self._model.encode(batch, convert_to_numpy=True))
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        embeddings = (
            np.concatenate(all_embeddings, axis=0)
            if len(all_embeddings) > 1
            else all_embeddings[0]
        )
        return [self._normalize(e).tolist() for e in embeddings]


class ChromaDBBuilder:
    """Builds a ChromaDB collection from a directory of markdown files.

    Walks input_root/{source_group}/ and chunks every .md/.mmd file found.
    Metadata stored per chunk: source_group, source_file, chunk_index.
    """

    def __init__(
        self,
        output_dir: str,
        collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: str = "auto",
        batch_size: int = 16,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        import torch
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "cuda:0" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        if "cuda" in device:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        logger.info(f"Loading embedding model: {embedding_model_name} on {device}")
        self.embedding_model = SentenceTransformer(
            embedding_model_name, trust_remote_code=True, device=device
        )
        self.embedding_function = TransformerEmbeddingFunction(
            embedding_model=self.embedding_model,
            batch_size=batch_size,
        )

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning("tiktoken unavailable, using approximate token counting")
            self.tokenizer = None

        logger.info(f"Initializing ChromaDB at: {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=output_dir)

        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Created new collection: {collection_name}")

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of ~chunk_size tokens with overlap at sentence boundaries."""
        if not text.strip():
            return []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_chunk: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self._count_tokens(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                current_chunk, current_tokens = overlap_chunk, overlap_tokens
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def process_group(self, source_group: str, files: List[Path]) -> int:
        """Chunk and ingest all files for one source group. Returns chunks added."""
        all_ids: List[str] = []
        all_chunks: List[str] = []
        all_metadatas: List[dict] = []

        for file_path in files:
            try:
                text = file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"Cannot read {file_path}: {e}")
                continue
            chunks = self._chunk_text(text)
            logger.info(f"  {file_path.name}: {len(chunks)} chunks")
            for chunk_idx, chunk in enumerate(chunks):
                all_ids.append(f"{source_group}_{file_path.stem}_c{chunk_idx}")
                all_chunks.append(chunk)
                all_metadatas.append({
                    'source_group': source_group,
                    'source_file': file_path.name,
                    'chunk_index': chunk_idx,
                })

        if not all_chunks:
            return 0

        import torch
        add_batch = self.batch_size * 2
        total_added = 0
        for i in range(0, len(all_chunks), add_batch):
            self.collection.add(
                ids=all_ids[i:i + add_batch],
                documents=all_chunks[i:i + add_batch],
                metadatas=all_metadatas[i:i + add_batch],
            )
            total_added += len(all_ids[i:i + add_batch])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"  Added {total_added} chunks for {source_group}")
        return total_added

    def build(self, input_root: Path) -> Dict[str, int]:
        """Walk input_root and ingest every subdirectory as a source group."""
        input_root = Path(input_root)
        if not input_root.exists():
            raise ValueError(f"Input directory does not exist: {input_root}")

        group_dirs = sorted(d for d in input_root.iterdir() if d.is_dir())
        logger.info(f"Found {len(group_dirs)} source group(s)")

        results: Dict[str, int] = {}
        for group_dir in group_dirs:
            source_group = group_dir.name
            files = sorted(
                f for f in group_dir.iterdir()
                if f.is_file() and f.suffix in {'.md', '.mmd'}
            )
            if not files:
                logger.warning(f"Skipping {source_group}: no .md/.mmd files found")
                continue
            logger.info(f"Processing group: {source_group} ({len(files)} file(s))")
            try:
                results[source_group] = self.process_group(source_group, files)
            except Exception as e:
                logger.error(f"Error processing {source_group}: {e}", exc_info=True)
                results[source_group] = 0

        return results


def main(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    input_root = (_HERE / cfg['input_root']).resolve()
    output_dir = (_HERE / cfg['output_dir']).resolve()

    builder = ChromaDBBuilder(
        output_dir=str(output_dir),
        collection_name=cfg['collection_name'],
        chunk_size=cfg.get('chunk_size', 500),
        chunk_overlap=cfg.get('chunk_overlap', 50),
        embedding_model_name=cfg['embeddings']['model_name'],
        device=cfg['embeddings'].get('device', 'auto'),
        batch_size=cfg['embeddings'].get('batch_size', 16),
    )

    results = builder.build(input_root)

    print("\n" + "=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    total = 0
    for group, n_chunks in results.items():
        print(f"  {group}: {n_chunks} chunks")
        total += n_chunks
    print(f"\nTotal chunks: {total}")
    print(f"Output:       {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / 'config.yaml'
    main(config_path)
