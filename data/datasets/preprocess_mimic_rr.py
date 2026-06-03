import argparse
import os
import sys
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import wfdb
from tqdm import tqdm

sys.path.append("../")
sys.path.append("../../")
from data.datasets.mimic import load_mimic_metadata
from data.utils.denoise import denoise
from data.utils.r_peak_processing import extract_rr_beats_multi_lead

warnings.filterwarnings("ignore", module="pywt")


CLASSES = [
    "Chronic Ischemic Heart Disease (I25)",
    "Atrial Fibrillation (I48)",
    "Heart Failure (I50)",
    "Hypertensive Heart Diseases (I11-I16)",
    "Acute Myocardial Infarction (I21)",
    "Valve Disorders (I07,I08,I34,I35)",
    "Other",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records_path", type=str, required=True)
    p.add_argument("--mimic_path",   type=str, required=True)
    p.add_argument("--cache_dir",    type=str, required=True)
    p.add_argument("--workers",      type=int, default=min(8, cpu_count()))
    p.add_argument("--sampling_rate", type=int, default=500)
    p.add_argument("--overwrite",    action="store_true",
                   help="Re-compute even if the .npz already exists")
    return p.parse_args()


def _process_one(task):
    orig_idx, file_name, mimic_path, cache_dir, fs, overwrite = task

    out_path = os.path.join(cache_dir, f"{orig_idx}_rr.npz")
    if not overwrite and os.path.exists(out_path):
        return orig_idx          # already done

    record_path = os.path.join(mimic_path, file_name)
    try:
        record = wfdb.rdrecord(record_path)
        ecg    = record.p_signal.T                                  # (12, T)
        ecg    = np.stack([denoise(lead) for lead in ecg], axis=0)  # (12, T)

        beats, rr = extract_rr_beats_multi_lead(ecg, fs=fs)

        if beats is None or len(beats) == 0:
            return None

        # Store as object array so np.savez can handle variable-length rows
        beats_arr = np.empty(len(beats), dtype=object)
        for i, b in enumerate(beats):
            beats_arr[i] = b.astype(np.float32)   # (T_i, 12)

        rr = np.asarray(rr, dtype=np.float32)     # (S,)

        np.savez(out_path, beats_arr=beats_arr, rr=rr)
        return orig_idx

    except Exception as e:
        print(f"[SKIP] orig_idx={orig_idx}  {file_name}  — {e}")
        return None


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    df = load_mimic_metadata(args.records_path, CLASSES).reset_index(drop=True)
    df["orig_idx"] = df.index
    print(f"Total records in metadata: {len(df)}")

    tasks = [
        (
            int(row["orig_idx"]),
            row["file_name"],
            args.mimic_path,
            args.cache_dir,
            args.sampling_rate,
            args.overwrite,
        )
        for _, row in df.iterrows()
    ]

    good_indices = []

    with Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=32),
            total=len(tasks),
            desc="Precomputing RR cache",
        ):
            if result is not None:
                good_indices.append(result)

    good_indices = np.array(sorted(good_indices), dtype=np.int64)
    out_idx_path = os.path.join(args.cache_dir, "good_indices_rr.npy")
    np.save(out_idx_path, good_indices)

    n_failed = len(df) - len(good_indices)
    print(f"\nDone.  {len(good_indices)} succeeded, {n_failed} failed/skipped.")
    print(f"Good indices saved to: {out_idx_path}")


if __name__ == "__main__":
    main()