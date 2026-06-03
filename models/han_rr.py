import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchinfo import summary

class ChannelAttention(nn.Module):
    # Expects x shape = (batch, channels, seq_len)
    def __init__(self, channels, ratio=8):
        super().__init__()
        mid = max(1, channels // ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=True),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=True)
        )

    def forward(self, x):
        # x: (B, C, L)
        avg_pool = torch.mean(x, dim=2)           # (B, C)
        max_pool, _ = torch.max(x, dim=2)         # (B, C)
        avg_out = self.mlp(avg_pool)              # (B, C)
        max_out = self.mlp(max_pool)              # (B, C)
        att = torch.sigmoid(avg_out + max_out)    # (B, C)
        att = att.unsqueeze(2)                    # (B, C, 1)
        return x * att                             # broadcast multiply -> (B, C, L)                     

class BeatEncoder(nn.Module):
    def __init__(self, in_channels=12, mid_channels=64, out_channels=128,
                 fixed_len=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=15, padding=7)
        self.in1 = nn.InstanceNorm1d(mid_channels, affine=True)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=11, padding=5)
        self.in2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.channel_attention = ChannelAttention(out_channels, ratio=8)
        self.pool = nn.AdaptiveAvgPool1d(fixed_len)

    def forward(self, x):
        # pad_sequence → (N, T_max, 12), then permute → (N, 12, T_max)
        x = x.permute(0, 2, 1)                      # (N, 12, T_max)
        x = F.relu(self.in1(self.conv1(x)))               # (N, mid, T_max)
        x = F.relu(self.in2(self.conv2(x)))               # (N, out, T_max)
        x = self.channel_attention(x)
        x = self.pool(x)                                  # (N, out, fixed_len)
        return x.permute(0, 2, 1)                         # (N, fixed_len, out)
    
class SegmentAttention(nn.Module):
    def __init__(self, input_dim, units):
        super().__init__()
        self.linear = nn.Linear(input_dim, units, bias=True)
        self.u = nn.Parameter(torch.randn(units))

    def forward(self, inputs):
        v = torch.tanh(self.linear(inputs))             
        vu = torch.matmul(v, self.u)                   
        alphas = F.softmax(vu, dim=1)                 
        alphas = torch.nan_to_num(alphas, nan=0.0)
        output = torch.sum(inputs * alphas.unsqueeze(-1), dim=1)  
        return output, alphas

class HAN_RR(nn.Module):
    def __init__(self, num_classes=11, in_channels=12, encoder_channels=128,
                 fixed_len=32, seg_hidden=256, seq_hidden=512, fc_dim=2048,
                 dropout=0.4):
        super().__init__()
        self.beat_encoder = BeatEncoder(
            in_channels=in_channels,
            out_channels=encoder_channels,
            fixed_len=fixed_len
        )

        # Level 1: within-beat (segment) attention
        self.lstm_segment = nn.LSTM(
            input_size=encoder_channels, hidden_size=seg_hidden, batch_first=True
        )
        self.segment_attention = SegmentAttention(
            input_dim=seg_hidden, units=seg_hidden
        )

        # Level 2: across-beat (sequence) attention
        self.lstm_sequence = nn.LSTM(
            input_size=seg_hidden + 1, hidden_size=seq_hidden, batch_first=True
        )
        
        self.final_attention = SegmentAttention(
            input_dim=seq_hidden, units=seq_hidden
        )

        # Classifier
        self.fc = nn.Linear(seq_hidden, fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fc_dim, num_classes)

    def forward(self, beats_list, beat_counts, rr):
        # Encode all beats: list of (T_i, C) -> (num_beats_in_batch, fixed_len_timesteps, encoder_channels)
        # T_i = number of samples for beat i, fixed_len_timesteps = temporal dimension, encoder_channels = features per timestep (num of channels in output)
        encoded = self.beat_encoder(beats_list)

        # Segment LSTM over the fixed_len timesteps within each beat
        seg_out, _ = self.lstm_segment(encoded)             # (num_beats_in_batch, fixed_len_timesteps, seg_hidden)

        # Segment attention: collapse fixed_len -> single vector per beat
        beat_vecs, seg_alphas = self.segment_attention(seg_out)  # (num_beats_in_batch, seg_hidden)

        # Split back into per-sample sequences
        per_sample = torch.split(beat_vecs, beat_counts.tolist())  # tuple of (num_beats, seg_hidden)
        per_sample_rr = torch.split(rr, beat_counts.tolist())

        # Stack the sequences 
        sequence = torch.stack(per_sample, dim=0)    # (B, num_beats, seg_hidden)
        rr_seq = torch.stack(per_sample_rr, dim=0).unsqueeze(-1) # (B, num_beats, 1)
        sequence = torch.cat([sequence, rr_seq], dim=-1) 
        
        # Sequence LSTM + attention across beats
        seq_out, _ = self.lstm_sequence(sequence)                # (B, S_max, seq_hidden)
    
        record_vec, final_alphas = self.final_attention(seq_out)  # (B, seq_hidden)

        # Classify
        x = F.relu(self.fc(record_vec))
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits

# def print_han_summary(device):
#     model = HAN_RR(num_classes=11).to(device)
    
#     summary(model,
#         input_size=(2, 10, 300, 12),    # (batch, segments, timesteps, channels)
#         col_names=("input_size", "output_size", "num_params", "trainable"),
#         depth=4,
#         device=device.type)    