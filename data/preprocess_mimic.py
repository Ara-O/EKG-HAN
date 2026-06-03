# preprocess_mimic.py
import os
import contextlib
import numpy as np
import pandas as pd
import wfdb
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
sys.path.append("../")
from data.utils.denoise import denoise
from data.utils.r_peak_processing import extract_beats_multi_lead
from data.datasets.mimic import load_mimic_metadata

RECORDS_PATH = r"c:\Users\eog5z\Documents\records.csv"
MIMIC_PATH = r"C:\Users\eog5z\Downloads\mimic\mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
CACHE_DIR = "mimic_cache"
PRE_SAMPLES = 99
POST_SAMPLES = 201
FS = 500

CLASSES = [
    "Chronic Ischemic Heart Disease (I25)",
    "Atrial Fibrillation (I48)",
    "Heart Failure (I50)",
    "Hypertensive Heart Diseases (I11-I16)",
    "Acute Myocardial Infarction (I21)",
    "Valve Disorders (I07,I08,I34,I35)",
    "Other",
]

os.makedirs(CACHE_DIR, exist_ok=True)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into a tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def process_one(df_idx, row):
    out_path = os.path.join(CACHE_DIR, f"{df_idx}.npz")
    if os.path.exists(out_path):
        return df_idx, True, None

    try:
        record_path = os.path.join(MIMIC_PATH, row["file_name"])
        record = wfdb.rdrecord(record_path)
        ecg = record.p_signal.T  # (12, T)

        ecg = np.stack([denoise(lead) for lead in ecg], axis=0).astype(np.float32)

        beats, rr = extract_beats_multi_lead(
            ecg, fs=FS, pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES
        )
        if beats is None or len(beats) == 0:
            return df_idx, False, "no beats"

        np.savez_compressed(
            out_path,
            beats=np.asarray(beats, dtype=np.float32),
            rr=np.asarray(rr, dtype=np.float32),
        )
        return df_idx, True, None
    except Exception as e:
        return df_idx, False, str(e)


def main():
    df = load_mimic_metadata(RECORDS_PATH, CLASSES).reset_index(drop=True)
    print(f"Preprocessing {len(df)} records to {CACHE_DIR}")

    already_done = sum(
        1 for i in range(len(df))
        if os.path.exists(os.path.join(CACHE_DIR, f"{i}.npz"))
    )
    if already_done > 0:
        print(f"  {already_done} already cached, will skip those")

    rows = list(df.iterrows())

    with tqdm_joblib(tqdm(desc="Preprocessing", total=len(rows), smoothing=0.05)):
        results = Parallel(n_jobs=32)(
            delayed(process_one)(idx, row) for idx, row in rows
        )

    ok = sum(1 for _, success, _ in results if success)
    fail = len(results) - ok
    print(f"\nDone. {ok} cached, {fail} failed.")

    failures = [(i, reason) for i, s, reason in results if not s]
    if failures:
        print(f"\nFirst few failure reasons:")
        for i, reason in failures[:20]:
            print(f"  idx {i}: {reason}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    good_idx = np.array([i for i, s, _ in results if s], dtype=np.int64)
    np.save(os.path.join(CACHE_DIR, "good_indices.npy"), good_idx)
    print(f"Saved good_indices.npy ({len(good_idx)} records)")


if __name__ == "__main__":
    main()