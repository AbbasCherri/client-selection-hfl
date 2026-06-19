import os
from huggingface_hub import snapshot_download

def download_dataset():
    local_dir = os.path.abspath("./data")
    print(f"Downloading AbbasABC/HFL-Dataset from Hugging Face to {local_dir}...")
    
    # Xet is the current fast path for Hub downloads. Enable high-performance mode
    # and keep the cache local so repeated attempts can reuse already-fetched chunks.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    # Use an aggressive worker count by default; network-bound dataset downloads
    # on GCP usually benefit from more concurrent requests.
    default_workers = min(32, max(8, (os.cpu_count() or 8) * 2))
    max_workers = int(os.getenv("HF_MAX_WORKERS", str(default_workers)))
    
    # Point HF cache inside the data dir to avoid doubling disk usage
    cache_dir = os.path.join(local_dir, ".hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ["HF_XET_CACHE"] = os.path.join(cache_dir, "xet")
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "32")
    
    # Download files using snapshot_download
    snapshot_download(
        repo_id="AbbasABC/HFL-Dataset",
        repo_type="dataset",
        local_dir=local_dir,
        ignore_patterns=[".git*", "README.md"],
        max_workers=max_workers,
        resume_download=True
    )
    print("Hugging Face dataset download complete.")

if __name__ == "__main__":
    download_dataset()
