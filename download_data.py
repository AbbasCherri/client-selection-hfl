import os
import getpass
from huggingface_hub import snapshot_download

def download_dataset():
    local_dir = os.path.abspath("./data")
    print(f"Downloading AbbasABC/HFL-Dataset from Hugging Face to {local_dir}...")
    dataset_revision = os.getenv("HF_DATASET_REVISION", "6cf97c900445e080e61cb45e1aa72515d3ff1de8")
    
    # Xet is the current fast path for Hub downloads. Enable high-performance mode
    # and keep the cache local so repeated attempts can reuse already-fetched chunks.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    # Keep repository metadata fan-out modest to avoid Hub rate limits.
    # Xet handles the heavy data transfer path separately.
    default_workers = int(os.getenv("HF_MAX_WORKERS", "4"))
    max_workers = int(os.getenv("HF_MAX_WORKERS", str(default_workers)))
    
    # Point HF cache inside the data dir to avoid doubling disk usage
    cache_dir = os.path.join(local_dir, ".hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ["HF_XET_CACHE"] = os.path.join(cache_dir, "xet")
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "64")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_UPDATE_CHECK", "1")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = getpass.getpass("Enter your Hugging Face access token: ").strip()
    if not hf_token:
        raise RuntimeError("A Hugging Face access token is required to download this dataset.")
    
    # Download files using snapshot_download
    snapshot_download(
        repo_id="AbbasABC/HFL-Dataset",
        repo_type="dataset",
        revision=dataset_revision,
        local_dir=local_dir,
        ignore_patterns=[".git*", "README.md"],
        max_workers=max_workers,
        resume_download=True,
        token=hf_token
    )
    print("Hugging Face dataset download complete.")

if __name__ == "__main__":
    download_dataset()
