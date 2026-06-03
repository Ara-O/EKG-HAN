import torch.nn as nn
import torch.nn.functional as F
import torch
class ChannelAttention(nn.Module):
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

class SegmentAttention(nn.Module):
    def __init__(self, input_dim, units):
        super().__init__()
        self.linear = nn.Linear(input_dim, units, bias=True) 
        # u vector for scoring
        self.u = nn.Parameter(torch.randn(units))

    def forward(self, inputs):
        # inputs: (B, T, D)
        v = torch.tanh(self.linear(inputs))           # (B, T, units)
        # score each time step
        vu = torch.matmul(v, self.u)                  # (B, T)
        alphas = F.softmax(vu, dim=1)                 # softmax over time dimension
        output = torch.sum(inputs * alphas.unsqueeze(-1), dim=1)  # (B, D)
        return output, alphas

class TimeDistributedSegmentAttention(nn.Module):
    def __init__(self, input_dim, units):
        super().__init__()
        self.segment_attention = SegmentAttention(input_dim, units)

    def forward(self, inputs):
        # inputs: (B, S, T, D)
        B, S, T, D = inputs.shape
        flat = inputs.view(B * S, T, D)                       # (B*S, T, D)
        outputs, alphas = self.segment_attention(flat)        # (B*S, D), (B*S, T)
        outputs = outputs.view(B, S, D)                       # (B, S, D)
        alphas = alphas.view(B, S, T)                         # (B, S, T)
        return outputs, alphas

class HANFixed(nn.Module):
    def __init__(
        self,
        num_classes,
        conv_channels=128,
        segment_hidden=256,
        sequence_hidden=512,
        fc_hidden=512,
        dropout=0.4,
    ):
        super().__init__()

        # 1D conv over 12 leads
        self.conv1d = nn.Conv1d(12, conv_channels, kernel_size=25, padding=12)
        self.channel_attention = ChannelAttention(conv_channels, ratio=8)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Beat-level LSTM
        self.lstm_segment = nn.LSTM(
            input_size=conv_channels,
            hidden_size=segment_hidden,
            batch_first=True
        )
        self.time_distributed_attention = TimeDistributedSegmentAttention(
            segment_hidden,
            segment_hidden
        )

        # Sequence-level LSTM
        self.lstm_sequence = nn.LSTM(
            input_size=segment_hidden + 1,
            hidden_size=sequence_hidden,
            batch_first=True
        )
        self.final_attention = SegmentAttention(sequence_hidden, sequence_hidden)

        self.fc = nn.Linear(sequence_hidden, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fc_hidden, num_classes)

    def forward(self, x, rr_intervals):
        logits, _, _ = self.forward_with_attention(x, rr_intervals)
        return logits

    def forward_with_attention(self, x, rr_intervals):
        # x: (B, S, T, C) = (batch, beats, time, leads)
        B, S, T, C = x.shape

        # process beats individually: (B*S, C, T)
        x = x.view(B * S, T, C).permute(0, 2, 1)

        conv = self.conv1d(x)
        att  = self.channel_attention(conv)
        pooled = self.pool(att).permute(0, 2, 1)     # (B*S, T2, conv_channels)

        seg_out, _ = self.lstm_segment(pooled)       # (B*S, T2, segment_hidden)

        # regroup beats: (B, S, T2, segment_hidden)
        seg_out = seg_out.view(B, S, seg_out.shape[1], seg_out.shape[2])

        # TimeDistributed attention over each beat
        seg_feats, seg_alphas = self.time_distributed_attention(seg_out)  # (B,S,segment_hidden)

        # adding how much time passed since the previous beat
        combined_feats = torch.cat([seg_feats, rr_intervals.unsqueeze(-1)], dim=-1)
        
        # Sequence-level LSTM over beats
        seq_out, _ = self.lstm_sequence(combined_feats)                        # (B,S,sequence_hidden)
        final_vec, final_alphas = self.final_attention(seq_out)           # (B,sequence_hidden), (B,S)

        x = F.relu(self.fc(final_vec))
        x = self.dropout(x)
        logits = self.classifier(x)                                       # (B,C)
        return logits, final_alphas, seg_alphas
