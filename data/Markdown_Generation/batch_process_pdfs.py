#!/usr/bin/env python3
"""
Batch processing script to convert all PDF files in MARS_Data/PDFs_paper/InternalMaterials to markdown.
Processes PDFs sequentially and skips already processed files.
Run from the repo root: python Markdown_Generation/batch_process_pdfs.py
"""

import os
import sys
import glob
import traceback
from pathlib import Path
from typing import List
from datetime import datetime
from tqdm import tqdm

# Script lives in Markdown_Generation/; repo root is one level up
_REPO_ROOT = Path(__file__).parent.parent

# Add the vllm directory to path so run_dpsk_ocr_pdf can be imported
_VLLM_DIR = _REPO_ROOT / "MARS_Data" / "Models" / "DeepSeek-OCR" / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"
sys.path.insert(0, str(_VLLM_DIR))

from run_dpsk_ocr_pdf import process_pdf, Colors

PDF_INPUT_DIR = _REPO_ROOT / "MARS_Data" / "PDFs_paper" / "InternalMaterials"


def find_all_pdfs(root_dir: str) -> List[str]:
    root_path = Path(root_dir).resolve()
    pattern = str(root_path / "**" / "*.pdf")
    pdf_files = glob.glob(pattern, recursive=True)
    pdf_files = [f for f in pdf_files if not any(part.endswith('.md') for part in Path(f).parts)]
    return sorted(pdf_files)


def is_already_processed(pdf_path: str) -> bool:
    pdf_path_obj = Path(pdf_path)
    output_dir = pdf_path_obj.parent / f"{pdf_path_obj.stem}.md"
    if not output_dir.exists() or not output_dir.is_dir():
        return False
    mmd_file = output_dir / f"{pdf_path_obj.stem}.mmd"
    return mmd_file.exists()


def generate_output_path(pdf_path: str) -> str:
    pdf_path_obj = Path(pdf_path)
    output_dir = pdf_path_obj.parent / f"{pdf_path_obj.stem}.md"
    return str(output_dir.resolve())


def main():
    if not PDF_INPUT_DIR.exists():
        print(f"{Colors.RED}Error: input directory not found at {PDF_INPUT_DIR}{Colors.RESET}")
        return

    print(f"{Colors.BLUE}Scanning for PDF files in: {PDF_INPUT_DIR}{Colors.RESET}")

    pdf_files = find_all_pdfs(str(PDF_INPUT_DIR))

    if not pdf_files:
        print(f"{Colors.YELLOW}No PDF files found in {PDF_INPUT_DIR}{Colors.RESET}")
        return

    print(f"{Colors.GREEN}Found {len(pdf_files)} PDF file(s){Colors.RESET}")

    unprocessed_pdfs = []
    skipped_count = 0

    for pdf_path in pdf_files:
        if is_already_processed(pdf_path):
            skipped_count += 1
            print(f"{Colors.YELLOW}Skipping (already processed): {pdf_path}{Colors.RESET}")
        else:
            unprocessed_pdfs.append(pdf_path)

    if skipped_count > 0:
        print(f"{Colors.YELLOW}Skipped {skipped_count} already processed PDF(s){Colors.RESET}")

    if not unprocessed_pdfs:
        print(f"{Colors.GREEN}All PDFs have already been processed!{Colors.RESET}")
        return

    print(f"{Colors.BLUE}Processing {len(unprocessed_pdfs)} PDF file(s)...{Colors.RESET}\n")

    log_dir = Path(__file__).parent / "batch_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_process_{timestamp}.log"

    success_count = 0
    failure_count = 0
    failed_files = []

    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"Batch Processing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Total PDFs to process: {len(unprocessed_pdfs)}\n")
        log.write("=" * 80 + "\n\n")

        for pdf_path in tqdm(unprocessed_pdfs, desc="Processing PDFs", unit="file"):
            pdf_name = Path(pdf_path).name
            output_path = generate_output_path(pdf_path)

            log.write(f"Processing: {pdf_path}\n")
            log.write(f"Output directory: {output_path}\n")

            try:
                success = process_pdf(pdf_path, output_path)

                if success:
                    success_count += 1
                    log.write(f"Status: SUCCESS\n\n")
                else:
                    failure_count += 1
                    failed_files.append(pdf_path)
                    log.write(f"Status: FAILED\n\n")

            except Exception as e:
                failure_count += 1
                failed_files.append(pdf_path)
                error_msg = f"Exception: {str(e)}"
                print(f"{Colors.RED}Error processing {pdf_name}: {error_msg}{Colors.RESET}")
                log.write(f"Status: FAILED - {error_msg}\n\n")
                log.write(traceback.format_exc() + "\n\n")

        log.write("=" * 80 + "\n")
        log.write("SUMMARY\n")
        log.write("=" * 80 + "\n")
        log.write(f"Total processed: {len(unprocessed_pdfs)}\n")
        log.write(f"Successful: {success_count}\n")
        log.write(f"Failed: {failure_count}\n")

        if failed_files:
            log.write("\nFailed files:\n")
            for failed_file in failed_files:
                log.write(f"  - {failed_file}\n")

    print(f"\n{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.GREEN}Batch processing complete!{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"Total processed: {len(unprocessed_pdfs)}")
    print(f"{Colors.GREEN}Successful: {success_count}{Colors.RESET}")
    if failure_count > 0:
        print(f"{Colors.RED}Failed: {failure_count}{Colors.RESET}")
    print(f"\nLog file: {log_file}")

    if failed_files:
        print(f"\n{Colors.RED}Failed files:{Colors.RESET}")
        for failed_file in failed_files:
            print(f"  - {failed_file}")


if __name__ == "__main__":
    main()
