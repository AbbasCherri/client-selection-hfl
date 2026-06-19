import os
import pickle
import hashlib
import pandas as pd
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, Subset
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class MultiModalDataset(Dataset):
    """
    Fuses aerial building imagery with structured seismic and location features.
    """
    def __init__(self, df, data_dir="./data", transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        
        # 9 structured features:
        # latitude, longitude, MMI_original, MMI_shape, PGA, PGV, SA_0_3, SA_1_0, SA_3_0
        self.feature_cols = [
            'latitude', 'longitude', 'MMI_original', 'MMI_shape', 
            'PGA', 'PGV', 'SA_0_3', 'SA_1_0', 'SA_3_0'
        ]
        self.features = torch.from_numpy(self.df[self.feature_cols].values.astype(np.float32))
        
        # Target label: damage_val
        self.labels = torch.from_numpy(self.df['damage_val'].values.astype(np.int64))
        self.chip_paths = self.df['chip_path'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.chip_paths[idx]
        
        # Resolve path relative to data_dir.
        # Original paths in CSV are e.g. ../Images/石川県...
        if img_path.startswith('../'):
            img_path = os.path.join(self.data_dir, img_path[3:])
        else:
            img_path = os.path.join(self.data_dir, img_path)

        # Load image
        try:
            with Image.open(img_path) as img_file:
                img = img_file.convert('RGB')
        except Exception as e:
            # Fallback if image load fails (e.g. missing file)
            img = Image.new('RGB', (128, 128), (0, 0, 0))

        # Convert to tensor: TF.to_tensor handles /255, HWC->CHW in a single
        # zero-copy operation, avoiding 3 intermediate numpy array allocations.
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_resized = img.resize((128, 128))
            img_tensor = TF.to_tensor(img_resized)

        features_tensor = self.features[idx]
        label_tensor = self.labels[idx]

        return img_tensor, features_tensor, label_tensor


def _build_partition_cache_path(csv_path, data_dir, N, train_ratio, random_seed):
    cache_root = os.path.join(os.path.dirname(os.path.abspath(csv_path)), ".partition_cache")
    os.makedirs(cache_root, exist_ok=True)

    stat = os.stat(csv_path)
    cache_key = "|".join([
        os.path.abspath(csv_path),
        str(stat.st_mtime_ns),
        str(stat.st_size),
        os.path.abspath(data_dir),
        str(N),
        f"{train_ratio:.6f}",
        str(random_seed),
    ])
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_root, f"partitions_{digest}.pkl")


def _load_partition_cache(cache_path):
    try:
        with open(cache_path, "rb") as cache_file:
            return pickle.load(cache_file)
    except Exception:
        return None


def _save_partition_cache(cache_path, payload):
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "wb") as cache_file:
        pickle.dump(payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)

def get_hfl_data_partitions(csv_path, data_dir="./data", N=70, train_ratio=0.8, random_seed=42):
    """
    Loads dataset, scales structured features, partitions buildings into N clients using K-Means,
    and splits each client's local data into train and test subsets.
    """
    df = pd.read_csv(csv_path)
    cache_path = _build_partition_cache_path(csv_path, data_dir, N, train_ratio, random_seed)
    
    # Fill missing values if any
    df = df.fillna(0)

    raw_coords = df[['latitude', 'longitude']].copy()
    
    # Scale structured features for MLP training consistency
    feature_cols = [
        'latitude', 'longitude', 'MMI_original', 'MMI_shape', 
        'PGA', 'PGV', 'SA_0_3', 'SA_1_0', 'SA_3_0'
    ]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    cache_payload = _load_partition_cache(cache_path)
    if cache_payload is not None:
        print(f"Loaded cached client partitions from {cache_path}")
        client_train_indices = cache_payload['client_train_indices']
        client_test_indices = cache_payload['client_test_indices']
        global_test_indices = cache_payload['global_test_indices']
        client_coords = cache_payload['client_coords']
    else:
        # K-Means clustering on longitude and latitude to form geographic client groups
        print(f"Partitioning data into {N} client nodes via K-Means...")
        coords = raw_coords[['longitude', 'latitude']].values
        kmeans = KMeans(n_clusters=N, random_state=random_seed, n_init=10)
        df['client_id'] = kmeans.fit_predict(coords)

        client_train_indices = {}
        client_test_indices = {}
        global_test_indices = []
        client_coords = {}

        # Generate splits for each client
        for client_id in range(N):
            indices = df[df['client_id'] == client_id].index.tolist()
            n_samples = len(indices)

            # Shuffle indices
            rng = np.random.default_rng(random_seed + client_id)
            rng.shuffle(indices)

            split_idx = int(n_samples * train_ratio)
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]

            client_train_indices[client_id] = train_idx
            client_test_indices[client_id] = test_idx
            global_test_indices.extend(test_idx)

            if indices:
                coord_frame = raw_coords.iloc[indices]
                client_coords[client_id] = (
                    float(coord_frame['latitude'].mean()),
                    float(coord_frame['longitude'].mean())
                )
            else:
                client_coords[client_id] = (
                    float(raw_coords['latitude'].mean()),
                    float(raw_coords['longitude'].mean())
                )

        _save_partition_cache(
            cache_path,
            {
                'client_train_indices': client_train_indices,
                'client_test_indices': client_test_indices,
                'global_test_indices': global_test_indices,
                'client_coords': client_coords,
            },
        )

    # Initialize full dataset
    full_dataset = MultiModalDataset(df, data_dir=data_dir)

    print(f"Data partitioning complete. Total samples: {len(df)}.")
    
    return full_dataset, client_train_indices, client_test_indices, global_test_indices, client_coords
