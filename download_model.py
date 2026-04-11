"""
Download Karakalpak POS+Morph model from Hugging Face Hub to a local directory.

Usage (VPS):
  python3 download_model.py --token hf_xxxx
  python3 download_model.py --token hf_xxxx --dest /opt/karakalpak-model

Files downloaded:
  model.safetensors       → <dest>/final_model7/
  config.json             → <dest>/final_model7/
  tokenizer.json          → <dest>/final_model7/
  tokenizer_config.json   → <dest>/final_model7/
  sentencepiece.bpe.model → <dest>/final_model7/
  special_tokens_map.json → <dest>/final_model7/
  label_mappings.pkl      → <dest>/
  lemma_dict.pkl          → <dest>/
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "nickoo004/karakalpak-pos-morph-model"

# (path_in_repo, relative_local_path)
FILES = [
    ("model.safetensors",        "final_model7/model.safetensors"),
    ("config.json",              "final_model7/config.json"),
    ("tokenizer.json",           "final_model7/tokenizer.json"),
    ("tokenizer_config.json",    "final_model7/tokenizer_config.json"),
    ("sentencepiece.bpe.model",  "final_model7/sentencepiece.bpe.model"),
    ("special_tokens_map.json",  "final_model7/special_tokens_map.json"),
    ("label_mappings.pkl",       "label_mappings.pkl"),
    ("lemma_dict.pkl",           "lemma_dict.pkl"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="Hugging Face READ token (or set HF_TOKEN env var)")
    parser.add_argument(
        "--dest",
        default="/opt/karakalpak-model",
        help="Destination directory on VPS (default: /opt/karakalpak-model)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face READ token: ").strip()
    if not token:
        print("ERROR: No token provided.")
        sys.exit(1)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "final_model7").mkdir(exist_ok=True)

    total = len(FILES)
    for idx, (repo_path, local_rel) in enumerate(FILES, 1):
        local_path = dest / local_rel
        if local_path.exists():
            size_mb = local_path.stat().st_size / 1024 / 1024
            print(f"  [{idx}/{total}] SKIP (already exists, {size_mb:.1f} MB): {local_rel}")
            continue

        print(f"  [{idx}/{total}] Downloading {repo_path}...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_path,
            token=token,
            repo_type="model",
            local_dir=str(dest / local_rel.rsplit("/", 1)[0]) if "/" in local_rel else str(dest),
        )
        # hf_hub_download may place the file under a cache subdir — move it to the exact path
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != local_path.resolve():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.replace(local_path)

        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"         Done ({size_mb:.1f} MB).")

    print(f"\nAll files ready in: {dest}")
    print(f"\nDirectory layout expected by the API:")
    print(f"  {dest}/")
    print(f"  ├── final_model7/")
    print(f"  │   ├── model.safetensors")
    print(f"  │   ├── config.json")
    print(f"  │   ├── tokenizer.json")
    print(f"  │   ├── tokenizer_config.json")
    print(f"  │   ├── sentencepiece.bpe.model")
    print(f"  │   └── special_tokens_map.json")
    print(f"  ├── label_mappings.pkl")
    print(f"  └── lemma_dict.pkl")


if __name__ == "__main__":
    main()
