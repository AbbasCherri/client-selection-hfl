import os
import pandas as pd
import numpy as np
import torch
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
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        
        # Target label: damage_val
        self.labels = self.df['damage_val'].values.astype(np.int64)
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
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback if image load fails (e.g. missing file)
            img = Image.new('RGB', (128, 128), (0, 0, 0))

        # Basic transform: convert to PyTorch tensor and normalize to [0, 1]
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_resized = img.resize((128, 128))
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            img_np = img_np.transpose(2, 0, 1)  # HWC to CHW
            img_tensor = torch.tensor(img_np)

        features_tensor = torch.tensor(self.features[idx])
        label_tensor = torch.tensor(self.labels[idx])

        return img_tensor, features_tensor, label_tensor

def get_hfl_data_partitions(csv_path, data_dir="./data", N=70, train_ratio=0.8, random_seed=42):
    """
    Loads dataset, scales structured features, partitions buildings into N clients using K-Means,
    and splits each client's local data into train and test subsets.
    """
    df = pd.read_csv(csv_path)
    
    # Fill missing values if any
    df = df.fillna(0)
    
    # Scale structured features for MLP training consistency
    feature_cols = [
        'latitude', 'longitude', 'MMI_original', 'MMI_shape', 
        'PGA', 'PGV', 'SA_0_3', 'SA_1_0', 'SA_3_0'
    ]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # K-Means clustering on longitude and latitude to form geographic client groups
    print(f"Partitioning data into {N} client nodes via K-Means...")
    coords = df[['longitude', 'latitude']].values
    kmeans = KMeans(n_clusters=N, random_state=random_seed, n_init=10)
    df['client_id'] = kmeans.fit_predict(coords)

    # Initialize full dataset
    full_dataset = MultiModalDataset(df, data_dir=data_dir)
    
    client_train_indices = {}
    client_test_indices = {}
    
    global_test_indices = []
    
    # Generate splits for each client
    for client_id in range(N):
        indices = df[df['client_id'] == client_id].index.tolist()
        n_samples = len(indices)
        
        # Shuffle indices
        np.random.seed(random_seed + client_id)
        np.random.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        client_train_indices[client_id] = train_idx
        client_test_indices[client_id] = test_idx
        global_test_indices.extend(test_idx)
        
    print(f"Data partitioning complete. Total samples: {len(df)}.")
    
    return full_dataset, client_train_indices, client_test_indices, global_test_indices
