import sys
import os
sys.path.insert(0, os.path.join(".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from data.utils.r_peak_processing import detect_r_peaks

SINUS_ARRHYTHMIA_IDX = 1 

ORIG_PATH = os.path.join("model_pths", "han_original.pth")
UPD_PATH  = os.path.join("model_pths", "han_updated.pt")
DATA_PATH = os.path.join("C.npy")

FS = 500
PRE_SAMPLES  = 99
POST_SAMPLES = 201


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        mid = max(1, channels // ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=True),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=True),
        )

    def forward(self, x):
        avg_out = self.mlp(torch.mean(x, dim=2))
        max_out = self.mlp(torch.max(x, dim=2).values)
        return x * torch.sigmoid(avg_out + max_out).unsqueeze(2)


class SegmentAttention(nn.Module):
    def __init__(self, input_dim, units):
        super().__init__()
        self.linear = nn.Linear(input_dim, units, bias=True)
        self.u = nn.Parameter(torch.randn(units))

    def forward(self, inputs):
        v = torch.tanh(self.linear(inputs))
        alphas = F.softmax(torch.matmul(v, self.u), dim=1)
        return torch.sum(inputs * alphas.unsqueeze(-1), dim=1), alphas


class TimeDistributedSegmentAttention(nn.Module):
    def __init__(self, input_dim, units):
        super().__init__()
        self.segment_attention = SegmentAttention(input_dim, units)

    def forward(self, inputs):
        B, S, T, D = inputs.shape
        out, alphas = self.segment_attention(inputs.view(B * S, T, D))
        return out.view(B, S, D), alphas.view(B, S, T)


class HANOriginal(nn.Module):
    def __init__(self, num_classes=11, conv_channels=128, segment_hidden=128,
                 sequence_hidden=512, fc_hidden=512, dropout=0.4):
        super().__init__()
        self.conv1d = nn.Conv1d(12, conv_channels, kernel_size=25, padding=12)
        self.channel_attention = ChannelAttention(conv_channels, ratio=8)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.lstm_segment = nn.LSTM(conv_channels, segment_hidden, batch_first=True)
        self.time_distributed_attention = TimeDistributedSegmentAttention(
            segment_hidden, segment_hidden
        )
        self.lstm_sequence = nn.LSTM(segment_hidden, sequence_hidden, batch_first=True)
        self.final_attention = SegmentAttention(sequence_hidden, sequence_hidden)
        self.fc = nn.Linear(sequence_hidden, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        # x: (B, S, T, C)
        B, S, T, C = x.shape
        x = x.view(B * S, T, C).permute(0, 2, 1)
        x = self.pool(self.channel_attention(self.conv1d(x))).permute(0, 2, 1)
        seg_out, _ = self.lstm_segment(x)
        seg_out = seg_out.view(B, S, seg_out.shape[1], seg_out.shape[2])
        seg_feats, _ = self.time_distributed_attention(seg_out)
        seq_out, _ = self.lstm_sequence(seg_feats)
        final_vec, _ = self.final_attention(seq_out)
        return self.classifier(self.dropout(F.relu(self.fc(final_vec))))


class HANUpdated(nn.Module):
    def __init__(self, num_classes=11, conv_channels=128, segment_hidden=128,
                 sequence_hidden=512, fc_hidden=512, dropout=0.4):
        super().__init__()
        self.conv1d = nn.Conv1d(12, conv_channels, kernel_size=25, padding=12)
        self.channel_attention = ChannelAttention(conv_channels, ratio=8)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.lstm_segment = nn.LSTM(conv_channels, segment_hidden, batch_first=True)
        self.time_distributed_attention = TimeDistributedSegmentAttention(
            segment_hidden, segment_hidden
        )
        self.lstm_sequence = nn.LSTM(segment_hidden + 1, sequence_hidden, batch_first=True)
        self.final_attention = SegmentAttention(sequence_hidden, sequence_hidden)
        self.fc = nn.Linear(sequence_hidden, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fc_hidden, num_classes)

    def forward(self, x, rr_intervals):
        # x: (B, S, T, C) | rr_intervals: (B, S)
        B, S, T, C = x.shape
        x = x.view(B * S, T, C).permute(0, 2, 1)
        x = self.pool(self.channel_attention(self.conv1d(x))).permute(0, 2, 1)
        seg_out, _ = self.lstm_segment(x)
        seg_out = seg_out.view(B, S, seg_out.shape[1], seg_out.shape[2])
        seg_feats, _ = self.time_distributed_attention(seg_out)
        combined = torch.cat([seg_feats, rr_intervals.unsqueeze(-1)], dim=-1)
        seq_out, _ = self.lstm_sequence(combined)
        final_vec, _ = self.final_attention(seq_out)
        return self.classifier(self.dropout(F.relu(self.fc(final_vec))))


def compute_rr(sample):
    ecg = sample["ecg"]                          # (12, 5000)
    r_peaks = detect_r_peaks(ecg, FS)
    if len(r_peaks) == 0:
        return None

    valid_mask = (r_peaks - PRE_SAMPLES >= 0) & (r_peaks + POST_SAMPLES <= ecg.shape[1])
    valid_peaks = r_peaks[valid_mask]
    if len(valid_peaks) < 2:
        return None

    rr = np.diff(valid_peaks) / FS
    rr = np.insert(rr, 0, np.mean(rr)).astype(np.float32)   # (n_beats,)

    stored_n_beats = sample["beats"].shape[0]
    if stored_n_beats != rr.shape[0]:
        return None

    return rr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading models...")
    han_orig = HANOriginal().to(device)
    han_orig.load_state_dict(torch.load(ORIG_PATH, map_location=device))
    han_orig.eval()

    han_upd = HANUpdated().to(device)
    han_upd.load_state_dict(torch.load(UPD_PATH, map_location=device))
    han_upd.eval()
    print("Models loaded.")

    samples = np.load(DATA_PATH, allow_pickle=True)
    print(f"Loaded {len(samples)} sinus arrhythmia samples from C.npy")

    improved   = []
    skipped_rr = 0
    orig_correct = 0
    upd_correct  = 0

    for sample in tqdm(samples, desc="Evaluating"):
        rr = compute_rr(sample)
        if rr is None:
            skipped_rr += 1
            continue

        beats = (
            torch.tensor(sample["beats"], dtype=torch.float32)
            .unsqueeze(0)          # (1, S, T, C)
            .to(device)
        )
        rr_tensor = (
            torch.tensor(rr, dtype=torch.float32)
            .unsqueeze(0)          # (1, S)
            .to(device)
        )

        with torch.no_grad():
            orig_logits = han_orig(beats)            # (1, 11)
            upd_logits  = han_upd(beats, rr_tensor)  # (1, 11)

        orig_pred = orig_logits.argmax(dim=1).item()
        upd_pred  = upd_logits.argmax(dim=1).item()

        if orig_pred == SINUS_ARRHYTHMIA_IDX:
            orig_correct += 1
        if upd_pred == SINUS_ARRHYTHMIA_IDX:
            upd_correct += 1

        if upd_pred == SINUS_ARRHYTHMIA_IDX and orig_pred != SINUS_ARRHYTHMIA_IDX:
            improved.append({
                "ecg_id":      sample["ecg_id"],
                "full_ecg": sample["ecg"],
                "orig_pred":   orig_pred,
                "upd_pred":    upd_pred,
                "orig_logits": orig_logits.squeeze().cpu().numpy(),
                "upd_logits":  upd_logits.squeeze().cpu().numpy(),
                "beats": beats,
                "rr_tensor": rr_tensor
            })

    evaluated = len(samples) - skipped_rr
    orig_acc = 100 * orig_correct / evaluated if evaluated else 0.0
    upd_acc  = 100 * upd_correct  / evaluated if evaluated else 0.0

    print(f"\nSkipped (RR mismatch): {skipped_rr}")
    print(f"Evaluated: {evaluated}")
    print(f"Original HAN accuracy:  {orig_correct}/{evaluated} ({orig_acc:.1f}%)")
    print(f"Updated  HAN accuracy:  {upd_correct}/{evaluated} ({upd_acc:.1f}%)")
    print(f"Corrected by updated HAN: {len(improved)} / {evaluated}")
    
    np.save("han_updated_v_original_sinus_arrhythmia", improved)

    return improved


if __name__ == "__main__":
    main()
