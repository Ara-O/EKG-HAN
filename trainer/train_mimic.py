import argparse
import sys
import os
import warnings

sys.path.append("../")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, hamming_loss, recall_score, roc_auc_score, f1_score
from tqdm import tqdm
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from datetime import datetime

from utils.set_seed import set_seed
from data.utils.denoise import denoise
from models.han_fixed import HANFixed
from models.han_rr import HAN_RR

from data.datasets.mimic import load_mimic_metadata, OnTheFlyBucketedBatches

warnings.filterwarnings("ignore", module="pywt")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

torch.set_num_threads(2)

CACHE_DIR = r"C:\Users\oladipea\Downloads\ddd\mimic_cache"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["han_fixed", "han_rr", "catnet"])
    p.add_argument("--records_path", type=str, required=True)
    p.add_argument("--mimic_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                   help="Which seeds to run in this process (e.g. --seeds 2 3)")
    p.add_argument("--cache_dir", type=str, default=CACHE_DIR,
                   help="Directory containing preprocessed .npz files")
    return p.parse_args()


def build_model(args, num_classes):
    if args.model == "han_fixed":
        return HANFixed(num_classes=num_classes, conv_channels=128, fc_hidden=2048)
    elif args.model == "han_rr":
        return HAN_RR(num_classes=num_classes)
    elif args.model == "catnet":
        from models.catnet import CatNet
        return CatNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")



def make_bucketed_loader(args, classes, df, indices, batch_size, shuffle, drop_last, seed):
    mode = 'rr' if args.model == 'han_rr' else 'fixed'

    ds_iter = OnTheFlyBucketedBatches(
        df=df,
        indices=indices,
        mimic_path=args.mimic_path,
        pre_samples=99,
        post_samples=201,
        classes=classes,
        denoise_fn=denoise,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        sampling_rate=500,
        seed=seed,
        cache_dir=args.cache_dir,
        mode=mode,
    )

    return DataLoader(
        ds_iter,
        batch_size=None,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

def multilabel_metrics(logits, targets, thresh=0.36):
    preds   = (logits.sigmoid() > thresh).float().cpu().numpy()
    probs   = logits.sigmoid().cpu().numpy()
    targets = targets.cpu().numpy()

    f1        = f1_score(targets, preds, average='weighted', zero_division=0)
    hamming   = hamming_loss(targets, preds)
    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall    = recall_score(targets, preds, average='weighted', zero_division=0)

    auc_scores = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) > 1:
            try:
                auc_scores.append(roc_auc_score(targets[:, i], probs[:, i]))
            except (ValueError, Exception):
                continue

    auc = np.mean(auc_scores) if auc_scores else 0.0
    return f1, hamming, precision, recall, auc

def forward_batch(model, batch, device, args):
    if args.model == "han_rr":
        beats       = batch["beats"].to(device)
        beat_counts = batch["beat_counts"].to(device)
        rr          = batch["rr"].to(device)
        return model(beats, beat_counts, rr)
    else:  
        signal = batch["signal"].to(device)
        rr     = batch["rr"].to(device)
        return model(signal, rr)


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler, args):
    model.train()
    running_loss = 0.0
    num_samples  = 0
    all_logits   = []
    all_labels   = []

    running_f1 = running_auc = running_hamming = 0.0
    steps = 0

    for batch in (pbar := tqdm(loader, desc="Train")):
        labels = batch["label"].to(device)
        logits = forward_batch(model, batch, device, args)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        num_samples  += labels.size(0)
        all_logits.append(logits.detach())
        all_labels.append(labels)

        f1, hamming, _, _, auc = multilabel_metrics(logits.detach(), labels)
        running_f1      += f1
        running_hamming += hamming
        running_auc     += auc
        steps           += 1

        pbar.set_postfix(
            loss=running_loss / num_samples,
            f1=running_f1 / steps,
            auc=running_auc / steps,
            hamming=running_hamming / steps,
        )

    if scheduler is not None:
        scheduler.step()

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    f1, hamming, prec, rec, auc = multilabel_metrics(all_logits, all_labels)

    return {
        "loss": running_loss / max(1, num_samples),
        "f1": f1, "hamming": hamming,
        "precision": prec, "recall": rec, "auc": auc,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, args):
    model.eval()
    running_loss = 0.0
    num_samples  = 0
    all_logits   = []
    all_labels   = []

    for batch in tqdm(loader, desc="Eval"):
        labels = batch["label"].to(device)
        logits = forward_batch(model, batch, device, args)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        num_samples  += labels.size(0)
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    f1, hamming, prec, rec, auc = multilabel_metrics(all_logits, all_labels)

    return {
        "loss": running_loss / max(1, num_samples),
        "f1": f1, "hamming": hamming,
        "precision": prec, "recall": rec, "auc": auc,
    }


def main():
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[pid {os.getpid()}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print(f"[pid {os.getpid()}] torch sees {torch.cuda.device_count()} GPU(s)")
    if torch.cuda.is_available():
        print(f"[pid {os.getpid()}] using device: {torch.cuda.get_device_name(0)}")

    print(f"[pid {os.getpid()}] model={args.model}  "
          f"mode={'rr' if args.model == 'han_rr' else 'fixed'}")

    classes = [
        "Chronic Ischemic Heart Disease (I25)",
        "Atrial Fibrillation (I48)",
        "Heart Failure (I50)",
        "Hypertensive Heart Diseases (I11-I16)",
        "Acute Myocardial Infarction (I21)",
        "Valve Disorders (I07,I08,I34,I35)",
        "Other"
    ]
    num_classes = len(classes)

    df = load_mimic_metadata(args.records_path, classes).reset_index(drop=True)
    df["orig_idx"] = df.index

    idx_fname = "good_indices_rr.npy" if args.model == "han_rr" else "good_indices.npy"
    good_idx  = np.load(os.path.join(args.cache_dir, idx_fname))
    good_mask = np.zeros(len(df), dtype=bool)
    good_mask[good_idx] = True
    df = df.loc[good_mask].reset_index(drop=True)
    print(f"Using {len(df)} records ({(~good_mask).sum()} dropped) [{idx_fname}]")

    batch_size   = 64
    seeds        = args.seeds
    metric_names = ["loss", "f1", "hamming", "precision", "recall", "auc"]
    all_results  = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag  = f"s{min(seeds)}-{max(seeds)}"
    out_path  = os.path.join(
        args.output_dir,
        f"results_mimic_{args.model}_{seed_tag}_{timestamp}.txt"
    )

    with open(out_path, "w") as f:
        f.write(f"Model: {args.model} | Dataset: MIMIC\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Cache: {args.cache_dir}\n")
        f.write("=" * 60 + "\n\n")

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}  (device: {args.device})")
        print(f"{'='*60}\n")

        set_seed(seed)

        Y = df[classes].astype(int).values

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=seed
        )
        trainval_idx, test_indices = next(msss.split(df.index, Y))

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=seed
        )
        train_idx_raw, val_idx_raw = next(msss.split(trainval_idx, Y[trainval_idx]))
        train_indices = trainval_idx[train_idx_raw]
        val_indices   = trainval_idx[val_idx_raw]

        train_loader = make_bucketed_loader(args, classes, df, train_indices,
                                            batch_size, shuffle=True,  drop_last=False, seed=seed)
        val_loader   = make_bucketed_loader(args, classes, df, val_indices,
                                            batch_size, shuffle=False, drop_last=False, seed=seed)
        test_loader  = make_bucketed_loader(args, classes, df, test_indices,
                                            batch_size, shuffle=False, drop_last=False, seed=seed)

        model     = build_model(args, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc    = 0
        patience        = 10
        patience_counter = 0
        epochs          = 50
        ckpt_path       = os.path.join(args.output_dir, f"best_model_{seed_tag}_seed{seed}.pt")

        for epoch in range(epochs):
            train_loader.dataset.set_epoch(epoch)
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer,
                                            device, scheduler, args)
            val_metrics   = evaluate(model, val_loader, criterion, device, args)

            print(f"[Seed {seed}] Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f} "
                  f"AUC: {val_metrics['auc']:.4f} Hamming: {val_metrics['hamming']:.4f}")

            if val_metrics['auc'] > best_val_auc:
                best_val_auc     = val_metrics['auc']
                torch.save(model.state_dict(), ckpt_path)
                print(f"  Saved best model (val_auc={val_metrics['auc']:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load(ckpt_path))
        test_metrics = evaluate(model, test_loader, criterion, device, args)

        print(f"[Seed {seed}] TEST | "
              f"Loss: {test_metrics['loss']:.4f} F1: {test_metrics['f1']:.4f} "
              f"Hamming: {test_metrics['hamming']:.4f} Precision: {test_metrics['precision']:.4f} "
              f"Recall: {test_metrics['recall']:.4f} AUC: {test_metrics['auc']:.4f}")

        all_results.append(test_metrics)

        with open(out_path, "a") as f:
            f.write(f"Seed {seed}\n")
            f.write("-" * 40 + "\n")
            for k in metric_names:
                f.write(f"  {k:>10s}: {test_metrics[k]:.4f}\n")
            f.write("\n")

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    with open(out_path, "a") as f:
        f.write("=" * 60 + "\n")
        f.write("Averaged Results (mean +/- std) for this process's seeds\n")
        f.write("-" * 60 + "\n")
        for m in metric_names:
            vals = [r[m] for r in all_results]
            f.write(f"  {m:>10s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}\n")

    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()