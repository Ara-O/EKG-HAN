import argparse
import sys 
import os

sys.path.append("../")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from tqdm import tqdm
import numpy as np 
from datetime import datetime
# import torch.distributed as dist

from data.datasets.sph import load_sph_dataset, ECGDataset
from data.datasets.cinc2020 import load_cinc_dataset
from data.utils.sampler import LengthBucketBatchSampler
from data.datasets.sph import default_collate, collate_by_num_beats, collate_variable_len
from data.utils.r_peak_processing import extract_beats_multi_lead, extract_rr_beats_multi_lead
from utils.set_seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["sph", "cinc", "mimic"])
    p.add_argument("--model", type=str, required=True, choices=["han_fixed", "han_rr", "catnet"])
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()

def extract_all_beats(data_list, fs=500, pre=99, post=201):
    out = []
    for ecg, label in data_list:
        beats, rr = extract_beats_multi_lead(ecg, fs=fs, pre_samples=pre, post_samples=post)
        if beats is None:
            continue
        out.append((beats, label, rr))
    return out

def extract_all_beats_rr(data_list, fs=500):
    out = []
    for ecg, label in data_list:
        beats, rr = extract_rr_beats_multi_lead(ecg, fs=fs)
        if beats is None:
            continue
        out.append((beats, label, rr))
    return out

def build_loaders(args, data_list, uses_beats, seed, batch_size):
    # Split indices
    indices = list(range(len(data_list)))
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)

    if args.dataset == 'sph':
        ds = ECGDataset(data_list, uses_beats=uses_beats)
    elif args.dataset == 'cinc':
        ds = ECGDataset(data_list, uses_beats=uses_beats)
    
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    if args.model == "han_rr":
        collate_fn = collate_variable_len
    elif args.model == "han_fixed":
        collate_fn = collate_by_num_beats
    else:
        collate_fn = default_collate

    def make_loader(subset, shuffle):
        if uses_beats:
            lengths = [ds.num_beats[i] for i in subset.indices]
            sampler = LengthBucketBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                drop_last=True, 
            )
            
            return DataLoader(subset, batch_sampler=sampler,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
        else:
            return DataLoader(subset, batch_size=batch_size,
                              shuffle=shuffle, collate_fn=collate_fn,
                              num_workers=0, pin_memory=True)

    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False)

def build_model(args, num_classes):
    if args.model == "han_fixed":
        from models.han_fixed import HANFixed

        if args.dataset == 'sph':
            return HANFixed(num_classes=num_classes, segment_hidden=128, sequence_hidden=512, fc_hidden=512)
        if args.dataset == 'cinc':
            return HANFixed(num_classes=num_classes, segment_hidden=256, sequence_hidden=256, fc_hidden=512)
    elif args.model == "han_rr":
        from models.han_rr import HAN_RR
        
        if args.dataset == 'sph':
            return HAN_RR(num_classes=num_classes, seg_hidden=128, seq_hidden=512, fc_dim=512)
        if args.dataset == 'cinc':
            return HAN_RR(num_classes=num_classes, seg_hidden=256, seq_hidden=256, fc_dim=512)
    elif args.model == "catnet":
        from models.catnet import CatNet
        return CatNet(num_classes=num_classes, dropout=args.dropout)


def train_one_epoch(model, loader, criterion, optimizer, device, args, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in (pbar := tqdm(loader, desc="Train")):
        labels = batch["label"].to(device)

        if args.model == "han_rr":
            beats = batch["beats"].to(device)
            beat_counts = batch["beat_counts"].to(device)
            rr = batch["rr"].to(device)
            logits = model(beats, beat_counts, rr)
        elif args.model == "han_fixed":
            beats = batch["beats"].to(device)
            rr = batch["rr"].to(device)
            logits = model(beats, rr)
        else:
            signal = batch["signal"].to(device)
            logits = model(signal)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()

        pbar.set_postfix(loss=total_loss / total, acc=correct / total)
    
    if scheduler is not None:
        try:
            scheduler.step()
        except Exception:
            pass

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, args, num_classes):
    model.eval()
    total_loss = 0
    total = 0
    all_labels, all_preds, all_probs = [], [], []

    for batch in tqdm(loader, desc="Eval"):
        labels = batch["label"].to(device)

        if args.model == "han_rr":
            beats = batch["beats"].to(device)
            beat_counts = batch["beat_counts"].to(device)
            rr = batch["rr"].to(device)
            logits = model(beats, beat_counts, rr)
        elif args.model == "han_fixed":
            beats = batch["beats"].to(device)
            rr = batch["rr"].to(device)
            logits = model(beats, rr)
        else:
            signal = batch["signal"].to(device)
            logits = model(signal)

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        all_labels.extend(labels_np)
        all_preds.extend(preds)
        all_probs.extend(probs)

    avg_loss = total_loss / total
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    try:
        y_true = np.eye(num_classes)[all_labels]
        y_score = np.array(all_probs)
        auc_list = []
        for i in range(num_classes):
            if np.any(y_true[:, i]):
                auc_list.append(roc_auc_score(y_true[:, i], y_score[:, i]))
        auc = np.mean(auc_list) if auc_list else float("nan")
    except ValueError:
        auc = float("nan")

    return avg_loss, acc, precision, recall, auc, f1


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
  
    # Load raw data
    if args.dataset == "sph":
        data_list, num_classes, class_mapping = load_sph_dataset(args.data_path)
    elif args.dataset == "cinc":
        data_list, num_classes, class_mapping = load_cinc_dataset(args.data_path)

    # Extract beats if needed (once, shared across all seeds)
    uses_beats = args.model in ("han_fixed", "han_rr")
    
    if uses_beats and args.model == 'han_rr':
        data_list = extract_all_beats_rr(data_list)
    elif uses_beats and args.model == 'han_fixed':
        data_list = extract_all_beats(data_list)

    # Prepare output file
    seeds = list(range(0,10))
    metric_names = ["loss", "acc", "precision", "recall", "f1", "auc"]
    all_results = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"results_{args.dataset}_{args.model}_{timestamp}.txt")

    # Write header
    with open(out_path, "w") as f:
        f.write(f"Model: {args.model} | Dataset: {args.dataset}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write("=" * 60 + "\n\n")

    # Loop over 10 seeds
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}\n")

        set_seed(seed)

        # Build loaders
        train_loader, val_loader, test_loader = build_loaders(args, data_list, uses_beats, seed, batch_size=16)

        # Model, optimizer, loss
        model = build_model(args, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=9e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        # Train
        best_val_acc = 0
        epochs = 25
        ckpt_path = os.path.join(args.output_dir, f"best_model_seed{seed}.pt")
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args, scheduler)
            val_loss, val_acc, val_prec, val_rc, val_auc, val_f1 = evaluate(model, val_loader, criterion, device, args, num_classes)

            print(f"[Seed {seed}] Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                  f"Prec: {val_prec:.4f} Rec: {val_rc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                print(f"  Saved best model (val_acc={val_acc:.4f})")

        # Test
        model.load_state_dict(torch.load(ckpt_path))
        test_loss, test_acc, test_prec, test_rc, test_auc, test_f1 = evaluate(model, test_loader, criterion, device, args, num_classes)
        print(f"[Seed {seed}] Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} "
              f"Prec: {test_prec:.4f} Rec: {test_rc:.4f} F1: {test_f1:.4f} AUC: {test_auc:.4f}")

        results = {"loss": test_loss, "acc": test_acc, "precision": test_prec,
                    "recall": test_rc, "f1": test_f1, "auc": test_auc}
        all_results.append(results)

        # Append this seed's results to file immediately
        with open(out_path, "a") as f:
            f.write(f"Seed {seed}\n")
            f.write("-" * 40 + "\n")
            for k in metric_names:
                f.write(f"  {k:>10s}: {results[k]:.4f}\n")
            f.write("\n")

        # Clean up checkpoint
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    # Append summary with mean +/- std
    with open(out_path, "a") as f:
        f.write("=" * 60 + "\n")
        f.write("Averaged Results (mean +/- std)\n")
        f.write("-" * 60 + "\n")
        for m in metric_names:
            vals = [r[m] for r in all_results]
            f.write(f"  {m:>10s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}\n")

    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()