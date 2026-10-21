# EKG-HAN

EKG arrhythmia classification using Hierarchical Attention Networks (HAN).

## Datasets

| Dataset      | Flag    | Notes                                                |
| ------------ | ------- | ---------------------------------------------------- |
| SPH          | `sph`   | Sinus rhythm / arrhythmia classification             |
| CinC 2020    | `cinc`  | PhysioNet/CinC Challenge 2020                        |
| MIMIC-IV ECG | `mimic` | Matched subset; requires separate preprocessing step |

## Setup

```bash
pip install torch numpy scikit-learn tqdm captum wfdb joblib matplotlib
```

## Usage

### Training (SPH / CinC 2020)

```bash
python trainer/train.py \
  --dataset sph \          # or cinc
  --model han_fixed \      # or han_rr
  --data_path /path/to/data \
  --device cuda \
  --output_dir results
```

Training runs over 10 random seeds and writes per-seed metrics plus a mean ± std summary to `results/results_<dataset>_<model>_<timestamp>.txt`.

### Training (MIMIC-IV ECG)

Preprocess the dataset first:

```bash
python data/preprocess_mimic.py
```

Then train:

```bash
python trainer/train_mimic.py \
  --model han_fixed \
  --mimic_path /path/to/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0 \
  --records_path /path/to/records.csv
```
