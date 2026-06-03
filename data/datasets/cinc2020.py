import os
import pandas as pd
import wfdb 
import numpy as np
from data.utils.r_peak_processing import extract_beats_multi_lead
from joblib import Parallel, delayed
import multiprocessing as mp
import sys

sys.path.append("../")
from data.utils.denoise import denoise

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pywt")

def load_cinc_dataset(path="CINC/RECORDS"):
    RECORDS_PATH = os.path.join(path, 'RECORDS')
    available_labels = ['10370003', '164889003', '164909002', '164934002', '270492004', '284470004', '426177001', '426783006', '427084000', '427393009', '59118001']

    with open(RECORDS_PATH) as f:
        FOLDERS = f.read().strip().splitlines()

    # Collect all file paths and skip excluded folders
    all_files = []
    for folder in FOLDERS:
        if folder in ["training/ptb/g1/", "training/st_petersburg_incart/g1/"]:
            continue
        with open(os.path.join(path, folder, "RECORDS")) as r:
            files = r.read().strip().splitlines()[:-1]
            all_files.extend(os.path.join(path, folder, fn) for fn in files)

    label2idx = {lbl: i for i, lbl in enumerate(sorted(available_labels))}
    num_classes = len(available_labels)
    print("num_classes:", num_classes, "labels:", sorted(available_labels))

    def process_file(final_data_path):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        try:
            hdr = wfdb.rdheader(final_data_path)
        except Exception as e:
            print("error reading header", final_data_path, e)
            return None

        label = None
        for comment in hdr.comments:
            if comment.startswith("Dx:") and "," not in comment:
                label = comment.replace("Dx: ", "")
                break

        if label is None or label not in label2idx:
            return None

        try:
            record = wfdb.rdrecord(final_data_path)
            signal = record.p_signal
        except Exception as e:
            print("error reading record", final_data_path, e)
            return None

        if np.isnan(signal).any():
            signal = np.nan_to_num(signal, nan=0.0)

        signal = signal.T
        signal = np.nan_to_num(np.array([denoise(lead) for lead in signal]), nan=0.0)

        return (signal, label2idx[label])

    n_jobs = max(1, mp.cpu_count() - 1)

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(
        delayed(process_file)(fp) for fp in all_files
    )

    data_list = [r for r in results if r is not None]

    return data_list, num_classes, label2idx