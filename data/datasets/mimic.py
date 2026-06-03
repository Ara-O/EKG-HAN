import pandas as pd
from torch.utils.data import IterableDataset
import torch 
import wfdb 
import os 
import random
import numpy as np
from collections import defaultdict

import sys
sys.path.append("../")

from data.utils.r_peak_processing import extract_beats_multi_lead, extract_rr_beats_multi_lead


def load_mimic_metadata(csv_path, classes):
    df = pd.read_csv(csv_path)

    df['file_name'] = df['file_name'].str.replace(
        'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/',
        ''
    )

    # Keep only records that have all required label columns
    df = df.dropna(subset=classes)

    return df


class OnTheFlyBucketedBatches(IterableDataset):
    def __init__(self, df, indices, mimic_path, pre_samples, post_samples,
                 classes, denoise_fn, batch_size=16, shuffle=True, drop_last=True,
                 sampling_rate=500, seed=10, cache_dir=None, mode='fixed'):
        super().__init__()
        self.df = df
        self.indices = list(indices)
        self.mimic_path = mimic_path
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.classes = classes
        self.denoise_fn = denoise_fn
        self.batch_size = int(batch_size)
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampling_rate = int(sampling_rate)
        self.epoch = 0
        self.cache_dir = cache_dir
        self.mode = mode  # 'fixed' or 'rr'

    def _cache_key(self, df_idx):
        row = self.df.iloc[df_idx]
        return int(row["orig_idx"]) if "orig_idx" in row.index else int(df_idx)

    def _cache_path_fixed(self, df_idx):
        return os.path.join(self.cache_dir, f"{self._cache_key(df_idx)}.npz")

    def _cache_path_rr(self, df_idx):
        return os.path.join(self.cache_dir, f"{self._cache_key(df_idx)}_rr.npz")


    def _load_fixed_from_cache(self, df_idx):
        cache_path = self._cache_path_fixed(df_idx)
        try:
            data = np.load(cache_path)
            beats = data["beats"]   # (S, 300, 12)
            rr    = data["rr"]      # (S,)
        except (FileNotFoundError, KeyError, ValueError, OSError):
            return None

        if beats is None or len(beats) == 0:
            return None

        labels = self.df.iloc[df_idx][self.classes].values.astype(np.float32)
        return {
            "signal":    torch.as_tensor(beats,  dtype=torch.float32),  # (S, 300, 12)
            "label":     torch.as_tensor(labels, dtype=torch.float32),  # (C,)
            "rr":        torch.as_tensor(rr,     dtype=torch.float32),  # (S,)
            "num_beats": beats.shape[0],
        }

    def _load_fixed_from_wfdb(self, df_idx):
        row = self.df.iloc[df_idx]
        record_path = os.path.join(self.mimic_path, row['file_name'])
        record = wfdb.rdrecord(record_path)
        ecg_signals = record.p_signal.T
        ecg_signals = [self.denoise_fn(signal) for signal in ecg_signals]

        beats, rr = extract_beats_multi_lead(
            ecg_signals,
            fs=self.sampling_rate,
            pre_samples=self.pre_samples,
            post_samples=self.post_samples,
        )

        if beats is None or len(beats) == 0:
            return None

        labels = row[self.classes].values.astype(np.float32)
        return {
            "signal":    torch.as_tensor(beats,  dtype=torch.float32),
            "label":     torch.as_tensor(labels, dtype=torch.float32),
            "rr":        torch.as_tensor(rr,     dtype=torch.float32),
            "num_beats": beats.shape[0],
        }

    def _load_rr_from_cache(self, df_idx):
        cache_path = self._cache_path_rr(df_idx)
        try:
            data  = np.load(cache_path, allow_pickle=True)
            beats = data["beats_arr"]   # numpy object array, S elements
            rr    = data["rr"]          # (S,)
        except (FileNotFoundError, KeyError, ValueError, OSError):
            return None

        if beats is None or len(beats) == 0:
            return None

        # Convert each beat (T_i, 12) -> torch tensor
        beat_tensors = [torch.as_tensor(b, dtype=torch.float32) for b in beats]
        labels = self.df.iloc[df_idx][self.classes].values.astype(np.float32)
        return {
            "signal":    beat_tensors,                                   # list of (T_i, 12) tensors
            "label":     torch.as_tensor(labels, dtype=torch.float32),  # (C,)
            "rr":        torch.as_tensor(rr,     dtype=torch.float32),  # (S,)
            "num_beats": len(beat_tensors),
        }

    def _load_rr_from_wfdb(self, df_idx):
        row = self.df.iloc[df_idx]
        record_path = os.path.join(self.mimic_path, row['file_name'])
        record = wfdb.rdrecord(record_path)
        ecg_signals = record.p_signal.T  # (12, T)

        if self.denoise_fn is not None:
            ecg_signals = np.stack(
                [self.denoise_fn(lead) for lead in ecg_signals], axis=0
            )

        beats, rr = extract_rr_beats_multi_lead(ecg_signals, fs=self.sampling_rate)

        if beats is None or len(beats) == 0:
            return None

        beat_tensors = [torch.as_tensor(b, dtype=torch.float32) for b in beats]
        labels = row[self.classes].values.astype(np.float32)
        return {
            "signal":    beat_tensors,
            "label":     torch.as_tensor(labels, dtype=torch.float32),
            "rr":        torch.as_tensor(rr,     dtype=torch.float32),
            "num_beats": len(beat_tensors),
        }


    def _load_record(self, df_idx):
        if self.mode == 'rr':
            if self.cache_dir is not None:
                result = self._load_rr_from_cache(df_idx)
                if result is not None:
                    return result
            return self._load_rr_from_wfdb(df_idx)
        else:  # 'fixed'
            if self.cache_dir is not None:
                return self._load_fixed_from_cache(df_idx)
            return self._load_fixed_from_wfdb(df_idx)

    def set_epoch(self, epoch):
        self.epoch = epoch


    def _stack_batch_fixed(self, items):
        signals = torch.stack([it["signal"] for it in items], dim=0)
        labels  = torch.stack([it["label"]  for it in items], dim=0)
        rr      = torch.stack([it["rr"]     for it in items], dim=0)
        return {"signal": signals, "label": labels, "rr": rr}

    def _stack_batch_rr(self, items):
        labels      = torch.stack([it["label"] for it in items], dim=0)  # (B, C)
        beat_counts = torch.tensor([it["num_beats"] for it in items], dtype=torch.long)  # (B,)
        rr          = torch.cat([it["rr"] for it in items], dim=0)     # (B, S)

        # Flat list of every beat across the whole batch
        all_beats = []
        for it in items:
            all_beats.extend(it["signal"])   # each element: (T_i, 12)

        beats_padded = torch.nn.utils.rnn.pad_sequence(all_beats, batch_first=True)  # (N, T_max, 12)
        return {
            "beats":       beats_padded,    # list[(T_i, 12)]
            "beat_counts": beat_counts,  # (B,)
            "rr":          rr,           # (B, S)
            "label":       labels,       # (B, C)
        }

    def _stack_batch(self, items):
        if self.mode == 'rr':
            return self._stack_batch_rr(items)
        return self._stack_batch_fixed(items)


    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        if wi is None:
            my_indices = list(self.indices)
        else:
            my_indices = self.indices[wi.id::wi.num_workers]

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch + (wi.id if wi else 0))
            rng.shuffle(my_indices)

        buckets = defaultdict(list)

        for df_idx in my_indices:
            try:
                item = self._load_record(df_idx)
            except Exception as e:
                print(f"Skip df_idx={df_idx} due to {e}")
                continue

            if item is None:
                continue

            S = item["num_beats"]
            buf = buckets[S]
            buf.append(item)

            if len(buf) >= self.batch_size:
                batch_items = buf[:self.batch_size]
                del buf[:self.batch_size]
                yield self._stack_batch(batch_items)

        if not self.drop_last:
            for S, buf in list(buckets.items()):
                while len(buf) > 0:
                    take = min(self.batch_size, len(buf))
                    batch_items = buf[:take]
                    del buf[:take]
                    yield self._stack_batch(batch_items)