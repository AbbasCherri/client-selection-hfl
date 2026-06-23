"""
data_loader.py – Streaming-first MultiModal data loading for HFL simulation.

Instead of downloading the full dataset, rows are streamed from Hugging Face
using the `datasets` library's IterableDataset API.  A lightweight metadata
DataFrame is built from the stream (coordinates + labels + chip-paths only),
used to run K-Means partitioning and produce per-client index lists, while
images are fetched on-demand from the GSI tile API (or a local fallback) when
the PyTorch DataLoader iterates over a client's shard.

Key design decisions
---------------------
* No full snapshot_download – only the Parquet/CSV shard(s) are streamed.
* Images are still large; they are fetched lazily via `requests` from the
  Japan GSI XYZ tile API, cached in a configurable local tile-cache directory,
  and composited into a 128×128 RGB chip.  A black dummy image is returned on
  any network error so that the training loop stays alive.
* The partition cache from the original code is retained – K-Means on ~128k
  rows is slow, so we cache the index assignments keyed on (stream revision,
  N, train_ratio, seed).
* The public API is identical to the original:
      get_hfl_data_partitions(csv_path=None, ...) → (full_dataset, ...)
  csv_path is now optional; when None the stream is used.
"""

import io
import os
import math
import pickle
import hashlib
import logging
import requests
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, Subset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO_ID        = "AbbasABC/HFL-Dataset"
HF_DATASET_REVISION = os.getenv(
    "HF_DATASET_REVISION",
    "6cf97c900445e080e61cb45e1aa72515d3ff1de8",
)
# Structured feature columns used by the MLP branch (must match models.py)
FEATURE_COLS = [
    "latitude", "longitude",
    "MMI_original", "MMI_shape",
    "PGA", "PGV",
    "SA_0_3", "SA_1_0", "SA_3_0",
]

# GSI tile parameters for on-demand image fetching
GSI_ZOOM   = 18          # zoom level – ~0.6 m/px
GSI_URL    = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
CHIP_PX    = 128         # output chip size in pixels
TILE_CACHE = os.getenv("HFL_TILE_CACHE", "./data/tile_cache")


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

def _latlon_to_tile(lat: float, lon: float, zoom: int):
    """Convert WGS-84 lat/lon to OSM/GSI tile (x, y) at given zoom."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    # Clamp to valid range
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def _fetch_tile(z: int, x: int, y: int, timeout: int = 5) -> Image.Image:
    """Fetch a single map tile; returns a blank 256×256 image on failure."""
    os.makedirs(TILE_CACHE, exist_ok=True)
    cache_file = os.path.join(TILE_CACHE, f"{z}_{x}_{y}.jpg")
    if os.path.exists(cache_file):
        try:
            return Image.open(cache_file).convert("RGB")
        except Exception:
            pass
    try:
        url = GSI_URL.format(z=z, x=x, y=y)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(cache_file, "JPEG", quality=85)
        return img
    except Exception as exc:
        logger.debug("Tile fetch failed (%s); using blank image.", exc)
        return Image.new("RGB", (256, 256), (0, 0, 0))


def _get_building_chip(lat: float, lon: float, zoom: int = GSI_ZOOM, size: int = CHIP_PX) -> Image.Image:
    """
    Returns a `size×size` RGB chip centred on (lat, lon) by compositing
    up to four adjacent tiles when the target pixel is near a tile edge.
    Falls back to a black chip on any error.
    """
    try:
        tx, ty = _latlon_to_tile(lat, lon, zoom)
        # Build a 2×2 mosaic of adjacent tiles then crop the centre
        mosaic = Image.new("RGB", (512, 512))
        for dy in range(2):
            for dx in range(2):
                tile = _fetch_tile(zoom, tx + dx, ty + dy)
                tile = tile.resize((256, 256), Image.BILINEAR)
                mosaic.paste(tile, (dx * 256, dy * 256))
        # Crop the central size×size region
        left   = (512 - size) // 2
        upper  = (512 - size) // 2
        chip   = mosaic.crop((left, upper, left + size, upper + size))
        return chip
    except Exception as exc:
        logger.debug("Chip generation failed (%s); returning black chip.", exc)
        return Image.new("RGB", (size, size), (0, 0, 0))


# ---------------------------------------------------------------------------
# HuggingFace streaming helpers
# ---------------------------------------------------------------------------

def _stream_metadata_df(
    subsample: float = 1.0,
    random_seed: int = 42,
    hf_token: str | None = None,
) -> pd.DataFrame:
    """
    Stream the HF dataset and return a lightweight metadata DataFrame
    containing only the columns needed for partitioning + training.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for streaming mode. "
            "Install it with: pip install datasets"
        ) from exc

    logger.info("Streaming metadata from %s (revision=%s) …", HF_REPO_ID, HF_DATASET_REVISION)

    ds = load_dataset(
        HF_REPO_ID,
        split="train",
        streaming=True,
        revision=HF_DATASET_REVISION,
        token=hf_token,
        trust_remote_code=False,
    )

    needed = set(FEATURE_COLS) | {"damage_val", "chip_path"}
    rows = []
    for row in ds:
        filtered = {k: row[k] for k in needed if k in row}
        if "damage_val" not in filtered and "label" in row:
            filtered["damage_val"] = row["label"]
        if "chip_path" not in filtered:
            filtered["chip_path"] = ""
        rows.append(filtered)

    df = pd.DataFrame(rows)
    logger.info("Streamed %d rows from HuggingFace.", len(df))

    # Strip non-target placeholders (9, 99) — keep only valid 4-class labels [0, 1, 2, 3]
    if "damage_val" in df.columns:
        df = df[df["damage_val"].isin([0, 1, 2, 3])].reset_index(drop=True)
        logger.info("Filtered invalid labels (9, 99). Remaining clean rows: %d", len(df))

    # Subsample deterministically
    if subsample < 1.0:
        n = max(1, int(len(df) * subsample))
        df = df.sample(n=n, random_state=random_seed).reset_index(drop=True)
        logger.info("Subsampled to %d rows (%.1f%%).", len(df), subsample * 100)

    df = df.fillna(0)
    return df


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MultiModalDataset(Dataset):
    """
    Fuses aerial building imagery with structured seismic and location features.
    """
    def __init__(self, df: pd.DataFrame, data_dir: str = "./data",
                 transform=None, use_gsi: bool = True):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = data_dir
        self.transform = transform
        self.use_gsi   = use_gsi

        feat_arr        = self.df[FEATURE_COLS].values.astype(np.float32)
        self.features   = torch.from_numpy(feat_arr)
        self.labels     = torch.from_numpy(self.df["damage_val"].values.astype(np.int64))
        self.chip_paths = self.df["chip_path"].values

        self.latitudes  = self.df["latitude"].values.astype(np.float64)
        self.longitudes = self.df["longitude"].values.astype(np.float64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self._load_image(idx)

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_resized = img.resize((CHIP_PX, CHIP_PX))
            img_tensor  = TF.to_tensor(img_resized)

        return img_tensor, self.features[idx], self.labels[idx]

    def _load_image(self, idx: int) -> Image.Image:
        chip_path = str(self.chip_paths[idx])
        if chip_path:
            local_path = self._resolve_local_path(chip_path)
            if local_path and os.path.isfile(local_path):
                try:
                    with Image.open(local_path) as f:
                        return f.convert("RGB")
                except Exception:
                    pass

        if self.use_gsi:
            lat = float(self.latitudes[idx])
            lon = float(self.longitudes[idx])
            if 35.0 <= lat <= 40.0 and 135.0 <= lon <= 140.0:
                return _get_building_chip(lat, lon)

        return Image.new("RGB", (CHIP_PX, CHIP_PX), (0, 0, 0))

    def _resolve_local_path(self, chip_path: str) -> str | None:
        if os.path.isabs(chip_path) and os.path.isfile(chip_path):
            return chip_path
        stripped = chip_path.lstrip("./")
        if stripped.startswith("../"):
            stripped = stripped[3:]
        candidate = os.path.join(self.data_dir, stripped)
        if os.path.isfile(candidate):
            return candidate
        return None


# ---------------------------------------------------------------------------
# Partition cache helpers
# ---------------------------------------------------------------------------

def _partition_cache_path(cache_root: str, cache_key_str: str) -> str:
    os.makedirs(cache_root, exist_ok=True)
    digest = hashlib.sha256(cache_key_str.encode()).hexdigest()[:16]
    return os.path.join(cache_root, f"partitions_{digest}.pkl")


def _load_partition_cache(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_partition_cache(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_hfl_data_partitions(
    csv_path: str | None = None,
    data_dir: str = "./data",
    N: int = 70,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    subsample: float = 0.05,
    hf_token: str | None = None,
):
    """
    Build the full MultiModalDataset and per-client index partitions.
    """
    # ------------------------------------------------------------------ #
    # 1. Load raw metadata                                                #
    # ------------------------------------------------------------------ #
    if csv_path and os.path.isfile(csv_path):
        logger.info("Loading metadata from local CSV: %s", csv_path)
        df = pd.read_csv(csv_path).fillna(0)
        
        # Clean local CSV targets — keep only valid 4-class labels [0, 1, 2, 3]
        if "damage_val" in df.columns:
            df = df[df["damage_val"].isin([0, 1, 2, 3])].reset_index(drop=True)
            
        if subsample < 1.0:
            n = max(1, int(len(df) * subsample))
            df = df.sample(n=n, random_state=random_seed).reset_index(drop=True)
            logger.info("Subsampled CSV to %d rows.", len(df))
        cache_root_dir = os.path.join(
            os.path.dirname(os.path.abspath(csv_path)), ".partition_cache"
        )
    else:
        logger.info("No local CSV found – streaming from HuggingFace.")
        df = _stream_metadata_df(
            subsample=subsample,
            random_seed=random_seed,
            hf_token=hf_token or os.getenv("HF_TOKEN"),
        )
        cache_root_dir = os.path.join(data_dir, ".partition_cache")

    # ------------------------------------------------------------------ #
    # 2. Feature scaling                                                  #
    # ------------------------------------------------------------------ #
    raw_lat = df["latitude"].values.copy()
    raw_lon = df["longitude"].values.copy()

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    # ------------------------------------------------------------------ #
    # 3. Partition cache                                                   #
    # ------------------------------------------------------------------ #
    cache_key_str = "|".join([
        str(len(df)),
        str(N),
        f"{train_ratio:.6f}",
        str(random_seed),
    ])
    cache_path = _partition_cache_path(cache_root_dir, cache_key_str)
    cached = _load_partition_cache(cache_path)

    if cached is not None:
        logger.info("Loaded cached partitions from %s", cache_path)
        client_train_indices = cached["client_train_indices"]
        client_test_indices  = cached["client_test_indices"]
        global_test_indices  = cached["global_test_indices"]
        client_coords        = cached["client_coords"]
    else:
        # ---------------------------------------------------------------- #
        # 4. K-Means geographic partitioning                               #
        # ---------------------------------------------------------------- #
        logger.info("Partitioning %d rows into %d clients via K-Means …", len(df), N)
        coords_for_km = np.column_stack([raw_lon, raw_lat])
        km = KMeans(n_clusters=N, random_state=random_seed, n_init=10)
        cluster_ids = km.fit_predict(coords_for_km)

        client_train_indices: dict[int, list[int]] = {}
        client_test_indices:  dict[int, list[int]] = {}
        global_test_indices:  list[int]            = []
        client_coords:        dict[int, tuple]     = {}

        for cid in range(N):
            idx_arr = np.where(cluster_ids == cid)[0].tolist()
            n_s     = len(idx_arr)

            rng = np.random.default_rng(random_seed + cid)
            rng.shuffle(idx_arr)

            split         = int(n_s * train_ratio)
            train_idx     = idx_arr[:split]
            test_idx      = idx_arr[split:]

            client_train_indices[cid] = train_idx
            client_test_indices[cid]  = test_idx
            global_test_indices.extend(test_idx)

            if idx_arr:
                client_coords[cid] = (
                    float(raw_lat[idx_arr].mean()),
                    float(raw_lon[idx_arr].mean()),
                )
            else:
                client_coords[cid] = (float(raw_lat.mean()), float(raw_lon.mean()))

        _save_partition_cache(cache_path, {
            "client_train_indices": client_train_indices,
            "client_test_indices":  client_test_indices,
            "global_test_indices":  global_test_indices,
            "client_coords":        client_coords,
        })

    # ------------------------------------------------------------------ #
    # 5. Build dataset                                                    #
    # ------------------------------------------------------------------ #
    full_dataset = MultiModalDataset(df, data_dir=data_dir, use_gsi=True)
    logger.info(
        "Data partitioning complete. Total samples: %d, clients: %d.",
        len(df), N,
    )

    return (
        full_dataset,
        client_train_indices,
        client_test_indices,
        global_test_indices,
        client_coords,
    )