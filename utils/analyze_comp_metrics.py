import sys
sys.path.append("../")

from models.han_fixed import HANFixed
from models.han_rr import HAN_RR
from computational_metrics import analyze_model_complexity, run_inference_benchmark, calculate_flops

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HANFixedWrapper(torch.nn.Module):
    def __init__(self, model, S=10):
        super().__init__()
        self.model = model
        self.S = S

    def forward(self, x):
        # x: (B, S, T, C)
        B = x.shape[0]
        rr = torch.ones(B, self.S, device=x.device)
        return self.model(x, rr)


class HANRRWrapper(torch.nn.Module):
    def __init__(self, model, S=10):
        super().__init__()
        self.model = model
        self.S = S

    def forward(self, x):
        # x: (B, S, T, C)
        B, S, T, C = x.shape
        beats_list  = x.view(B * S, T, C)                          # (B*S, T, C)
        beat_counts = torch.full((B,), S, dtype=torch.long,
                                 device=x.device)                   # (B,) all = S
        rr          = torch.ones(B * S, device=x.device)            # (B*S,)
        return self.model(beats_list, beat_counts, rr)


INPUT_SHAPE = (1, 10, 300, 12)   # (batch, beats, timesteps, leads)
BATCH_SIZE  = 16

# ── HANFixed ──────────────────────────────────────────────────────────────────
base_model = HANFixed(num_classes=11, segment_hidden=256, sequence_hidden=256, fc_hidden=512)
model = HANFixedWrapper(base_model, S=10)

# ── HAN_RR ────────────────────────────────────────────────────────────────────
# base_model = HAN_RR(num_classes=11, seg_hidden=128, seq_hidden=512, fc_dim=512)
# model = HANRRWrapper(base_model, S=10).to(device)
model.eval()

analyze_model_complexity(model, input_size=INPUT_SHAPE, device=device)
run_inference_benchmark(model, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, device=device)
calculate_flops(model)