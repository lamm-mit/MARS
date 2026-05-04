# Markdown Generation — Supplier Spec Sheet PDFs to Markdown

This directory contains `batch_process_pdfs.py`, which converts supplier specification sheet PDFs into markdown using [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR).

The script reads PDFs from `data/MARS_Data/PDFs_paper/InternalMaterials/`, skips already-processed files, and writes a `.md/` output folder alongside each PDF containing the markdown and any extracted images. Run it again at any time to process newly added PDFs. Logs are written to `data/Markdown_Generation/batch_logs/`.

---

## Requirements

- Linux machine with a CUDA-capable GPU
- CUDA 11.8
- Python 3.12

---

## 1. Set up the environment (one-time)

```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install data/MARS_Data/Models/DeepSeek-OCR/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r data/MARS_Data/Models/DeepSeek-OCR/requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

---

## 2. Configure the model path

Set `MODEL_PATH` in `data/MARS_Data/Models/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py` to one of:

```python
# Option 1 — official DeepSeek HuggingFace (downloads weights at runtime):
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'

# Option 2 — project HuggingFace repo (exact weights used in the paper):
MODEL_PATH = 'tphage/DeepSeek-OCR'

# Option 3 — local weights (if downloaded):
MODEL_PATH = '/path/to/data/MARS_Data/Models/DeepSeek-OCR'
```

---

## 3. Place your PDFs

Add PDFs to `data/MARS_Data/PDFs_paper/InternalMaterials/`, one subfolder per supplier:

```
data/MARS_Data/PDFs_paper/InternalMaterials/
└── {Supplier-Name}/
    └── {file}.pdf
```

A dummy example is included to illustrate the expected structure.

---

## 4. Run the script

Run from the **repo root**:

```bash
conda activate deepseek-ocr
python data/Markdown_Generation/batch_process_pdfs.py
```

Output for each PDF at `data/MARS_Data/PDFs_paper/InternalMaterials/{Supplier}/{file}.pdf`:

```
data/MARS_Data/PDFs_paper/InternalMaterials/{Supplier}/{file}.md/
├── {file}.mmd          # clean markdown
├── {file}_det.mmd      # markdown with layout detection annotations
├── {file}_layouts.pdf  # original pages with bounding-box overlays
└── images/             # figures extracted from the document
```

