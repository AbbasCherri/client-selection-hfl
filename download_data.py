import os
from huggingface_hub import snapshot_download

def download_dataset():
    local_dir = os.path.abspath("./data")
    print(f"Downloading AbbasABC/HFL-Dataset from Hugging Face to {local_dir}...")
    
    # Avoid over-concurrency socket hangs by defaulting to 4 workers, configurable via env var
    max_workers = int(os.getenv("HF_MAX_WORKERS", "4"))
    
    # Download files using snapshot_download
    snapshot_download(
        repo_id="AbbasABC/HFL-Dataset",
        repo_type="dataset",
        local_dir=local_dir,
        ignore_patterns=[".git*", "README.md"],
        max_workers=max_workers
    )
    print("Hugging Face dataset download complete.")

if __name__ == "__main__":
    download_dataset()
