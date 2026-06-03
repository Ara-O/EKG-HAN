import os
import h5py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from joblib import Parallel, delayed
import multiprocessing as mp

import sys

sys.path.append("../")
from data.utils.denoise import denoise

class ECGDataset(Dataset):
    def __init__(self, data_list, uses_beats=False):
        self.data = data_list
        self.uses_beats = uses_beats
        
        if uses_beats:
            self.num_beats = [len(x[0]) for x in data_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        beats_list, label, rr = self.data[idx]
        beats_tensor = [torch.tensor(b) for b in beats_list]
        rr_tensor = torch.as_tensor(rr, dtype=torch.float32)
        
        if self.uses_beats:
            return {"beats": beats_tensor, "label": int(label), "num_beats": int(len(beats_tensor)), "rr": rr_tensor}
        else:
            return {"signal": beats_tensor, "label": int(label)}


def load_sph_dataset(DATA_PATH):
    metadata = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))
    codes = pd.read_csv(os.path.join(DATA_PATH, "code.csv"))
    
    code_mappings = codes[['Code', 'Category']]
    code_mappings = {str(key): value for (_, (key, value)) in code_mappings.iterrows()}
    
    # Keep only single label instances
    metadata = metadata[~metadata['AHA_Code'].str.contains(";")]
    
    # Ignore the modifiers
    metadata['AHA_Code'] = [code.split("+")[0] for code in metadata['AHA_Code']]
    metadata['AHA_Code_Mapped'] = metadata['AHA_Code'].map(code_mappings)

    # create label -> index mapping
    unique_labels = sorted(metadata['AHA_Code_Mapped'].dropna().unique())
    label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    
    num_classes = len(unique_labels)
    print("num_classes:", num_classes, "labels:", unique_labels)

    data_list = [] 

    def process_row(row):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        ecg_data_path = os.path.join(DATA_PATH, "records", row.ECG_ID + ".h5")
        
        if not os.path.exists(ecg_data_path):
            return None
        try:
            with h5py.File(ecg_data_path, 'r') as f:
                ecg = np.array(f['ecg']) 
        except Exception as e:
            print("error reading", ecg_data_path, e)
            return None

        label = row.AHA_Code_Mapped

        ecg = np.array([denoise(lead) for lead in ecg])
        
        if pd.isna(label) or (label not in label2idx):
            return None
        
        return ((ecg, label2idx[label]))
     
    # Use all physical cores by default
    n_jobs = max(1, mp.cpu_count() - 1)

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(
        delayed(process_row)(row)
        for row in metadata.itertuples(index=False)
    )
    
    data_list = [r for r in results if r is not None]

    return data_list, num_classes, label2idx


# Collate functions
def default_collate(batch): 
    signals = torch.stack([item["signal"] for item in batch], dim=0)  
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"signal": signals, "label": labels}


# HAN-Fixed
def collate_by_num_beats(batch):
    beats = torch.stack([torch.stack(item["beats"]) for item in batch], dim=0)     
    rr_ints = torch.stack([item["rr"] for item in batch], dim=0) 
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    
    return {
        "beats": beats, 
        "label": labels, 
        "rr": rr_ints,
    }

# HAN_RR
def collate_variable_len(batch):
    all_beats = []
    labels = []
    beat_counts = []
    all_rr = []

    for item in batch:
        all_beats.extend(item["beats"])
        labels.append(item["label"])
        beat_counts.append(item["num_beats"])
        all_rr.append(item["rr"])

    labels = torch.tensor(labels, dtype=torch.long)
    beat_counts = torch.tensor(beat_counts, dtype=torch.long)
    rr = torch.cat(all_rr, dim=0)
    beats = torch.nn.utils.rnn.pad_sequence(all_beats, batch_first=True)  # (total_beats, T_max, 12)

    return {"beats": beats, "label": labels, "beat_counts": beat_counts, "rr": rr}