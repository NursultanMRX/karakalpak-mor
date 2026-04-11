"""
Upload Karakalpak POS+Morph model to Hugging Face Hub (private repo).

Usage:
  python upload_to_hf.py --token hf_xxxx
  python upload_to_hf.py          # will prompt for token

Files uploaded:
  final_model7/model.safetensors     (~1.06 GB, via LFS)
  final_model7/config.json
  final_model7/tokenizer.json
  final_model7/tokenizer_config.json
  final_model7/sentencepiece.bpe.model
  final_model7/special_tokens_map.json
  label_mappings.pkl
  lemma_dict.pkl
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID   = "nickoo004/karakalpak-pos-morph-model"
BASE_DIR  = Path(__file__).parent

FILES = [
    # (local_path, path_in_repo)
    (BASE_DIR / "final_model7" / "model.safetensors",        "model.safetensors"),
    (BASE_DIR / "final_model7" / "config.json",              "config.json"),
    (BASE_DIR / "final_model7" / "tokenizer.json",           "tokenizer.json"),
    (BASE_DIR / "final_model7" / "tokenizer_config.json",    "tokenizer_config.json"),
    (BASE_DIR / "final_model7" / "sentencepiece.bpe.model",  "sentencepiece.bpe.model"),
    (BASE_DIR / "final_model7" / "special_tokens_map.json",  "special_tokens_map.json"),
    (BASE_DIR / "label_mappings.pkl",                         "label_mappings.pkl"),
    (BASE_DIR / "lemma_dict.pkl",                             "lemma_dict.pkl"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="Hugging Face write token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face WRITE token (from https://hf.co/settings/tokens): ").strip()
    if not token:
        print("ERROR: No token provided.")
        sys.exit(1)

    api = HfApi(token=token)

    # 1. Create repo (private, idempotent)
    print(f"\nCreating private repo: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        private=True,
        exist_ok=True,
        token=token,
        repo_type="model",
    )
    print("  Repo ready.")

    # 2. Upload files one by one (gives progress per file)
    total = len(FILES)
    for idx, (local_path, repo_path) in enumerate(FILES, 1):
        if not local_path.exists():
            print(f"  [{idx}/{total}] SKIP (not found): {local_path}")
            continue
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"  [{idx}/{total}] Uploading {repo_path} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
            token=token,
        )
        print(f"         Done.")

    print(f"\nAll files uploaded.")
    print(f"Repo: https://huggingface.co/{REPO_ID}")
    print(f"\nDownload on VPS with:")
    print(f"  python3 download_model.py --token YOUR_TOKEN --dest /opt/karakalpak-model")

if __name__ == "__main__":
    main()
