"""
Download large KG data files from HuggingFace that are too large for git.
Run from the repo root: python data/download_data.py
"""
from pathlib import Path
from huggingface_hub import hf_hub_download

KG_REPO = "tphage/MARS-KGs"
KG_DIR = Path(__file__).parent / "MARS_Data" / "KGs"

KG_FILES = [
    "MatProp_62999.graphml",
    "MatProp_62999_embeddings_nomic_v1_5.pkl",
    "PFAS_4612.graphml",
    "PFAS_embeddings_nomic_v1_5.pkl",
    "Patent_13654_US.graphml",
    "Patent_13654_embeddings_nomic_v1_5.pkl",
]

def download_kgs():
    KG_DIR.mkdir(parents=True, exist_ok=True)
    for filename in KG_FILES:
        dest = KG_DIR / filename
        if dest.exists():
            print(f"  [skip] {filename} already exists")
            continue
        print(f"  [download] {filename} ...")
        hf_hub_download(
            repo_id=KG_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=KG_DIR,
        )
        print(f"  [done] {filename}")

if __name__ == "__main__":
    print("Downloading knowledge graphs from HuggingFace...")
    download_kgs()
    print("All done.")
