"""Main entry point for material database builder."""

import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config
from .database import Database
from .llm_client import GPTOSSClient, LLMClient
from .processor import MaterialProcessor
from .schemas import ProcessingStatus
from .state_manager import StateManager
from .utils import find_all_markdown_files


def setup_logging(config: Config) -> None:
    log_config = config.logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_config.file:
        handlers.append(logging.FileHandler(log_config.file))
    logging.basicConfig(
        level=getattr(logging, log_config.level.upper()),
        format=log_config.format,
        handlers=handlers,
    )


def discover_material_folders(input_root: Path) -> List[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root directory does not exist: {input_root}")
    material_folders = []
    for item in input_root.iterdir():
        if item.is_dir():
            md_files, mmd_files = find_all_markdown_files(item)
            if md_files or mmd_files:
                material_folders.append(item)
    return sorted(material_folders)


def process_materials(config: Config, base_path: Path) -> None:
    logger = logging.getLogger(__name__)

    input_root = Path(config.input_root)
    if not input_root.is_absolute():
        input_root = base_path / input_root

    db_file = Path(config.database_file)
    if not db_file.is_absolute():
        db_file = base_path / db_file

    state_file = Path(config.state_file)
    if not state_file.is_absolute():
        state_file = base_path / state_file

    logger.info(f"Input root:     {input_root}")
    logger.info(f"Database file:  {db_file}")
    logger.info(f"State file:     {state_file}")

    database = Database(db_file)
    state_manager = StateManager(state_file)

    llm_config = config.llm
    llm_client = GPTOSSClient(
        base_url=llm_config.base_url,
        model=llm_config.model,
        max_tokens=llm_config.max_tokens,
        timeout=llm_config.timeout,
    )

    processor = MaterialProcessor(
        llm_client=llm_client,
        base_path=base_path,
        max_repair_attempts=config.validation.max_repair_attempts,
        re_prompt_on_failure=config.validation.re_prompt_on_failure,
        max_input_tokens=config.llm.max_input_tokens,
        database=database,
    )

    logger.info("Discovering material folders...")
    material_folders = discover_material_folders(input_root)
    logger.info(f"Found {len(material_folders)} material folders")

    processed_count = skipped_count = failed_count = 0

    for folder_path in material_folders:
        folder_rel_path = str(folder_path.relative_to(base_path))
        try:
            file_fps, combined_fp = state_manager.compute_current_fingerprints(folder_path, base_path)
            if not state_manager.should_process(folder_rel_path, combined_fp):
                logger.info(f"Skipping {folder_rel_path} (unchanged)")
                skipped_count += 1
                continue

            material_id = state_manager.get_material_id(folder_rel_path)
            state_manager.update_material_state(folder_rel_path, status=ProcessingStatus.pending)
            state_manager.save_state()

            logger.info(f"Processing {folder_rel_path}...")
            material = processor.process_material_folder(folder_path, material_id)

            if material_id is None:
                material_id = material.id

            database.upsert_material(material)
            database.save_database()

            state_manager.update_material_state(
                folder_rel_path,
                material_id=material_id,
                file_fingerprints=file_fps,
                combined_fingerprint=combined_fp,
                status=ProcessingStatus.complete,
            )
            state_manager.save_state()

            logger.info(f"Completed {folder_rel_path} -> {material_id}")
            processed_count += 1

        except Exception as e:
            logger.error(f"Failed to process {folder_rel_path}: {e}", exc_info=True)
            failed_count += 1
            state_manager.update_material_state(
                folder_rel_path, status=ProcessingStatus.failed, error=str(e)
            )
            state_manager.save_state()

    logger.info("=" * 60)
    logger.info(f"Processed: {processed_count}  Skipped: {skipped_count}  Failed: {failed_count}")
    logger.info("=" * 60)

    if failed_count > 0:
        logger.warning(f"{failed_count} materials failed. Check logs for details.")
        sys.exit(1)


def main(config_path: Optional[Path] = None) -> None:
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Please create a config.yaml file."
        )

    config = Config.from_yaml(config_path)
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Starting material database builder")

    base_path = config_path.parent.absolute()

    try:
        process_materials(config, base_path)
        logger.info("Material database build completed successfully")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
