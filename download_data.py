import os
from huggingface_hub import snapshot_download

def download_dataset():
    local_dir = os.path.abspath("./data")
    print(f"Downloading AbbasABC/HFL-Dataset from Hugging Face to {local_dir}...")
    
    # Download files using snapshot_download
    snapshot_download(
        repo_id="AbbasABC/HFL-Dataset",
        repo_type="dataset",
        local_dir=local_dir,
        ignore_patterns=[".git*", "README.md"]
    )
    print("Hugging Face dataset download complete.")

if __name__ == "__main__":
    download_dataset()
