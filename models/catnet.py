import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# Channel Attention (CBAM)
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(1, channels // ratio)
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True)
        )

    def forward(self, x):
        # x: (B, C, L)
        b, c, l = x.shape

        avg_pool = F.adaptive_avg_pool1d(x, 1).view(b, c)   # (B, C)
        max_pool = F.adaptive_max_pool1d(x, 1).view(b, c)   # (B, C)
        avg_out = self.mlp(avg_pool)  # (B, C)
        max_out = self.mlp(max_pool)  # (B, C)

        out = torch.sigmoid(avg_out + max_out).unsqueeze(-1)  # (B, C, 1)
        return x * out  # broadcasting over length

# Spatial Attention (CBAM)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels=2, out_channels=1,
                              kernel_size=kernel_size, padding=padding, bias=False)
        # we will apply sigmoid after conv

    def forward(self, x):
        # x: (B, C, L)
        # channel-wise avg and max -> (B, 1, L)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        cat = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, L)
        attn = torch.sigmoid(self.conv(cat))  # (B, 1, L)
        return x * attn  # broadcast over channels

# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.2):
        super().__init__()
        # Use batch_first=True so inputs are (B, L, D)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
                                         dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(inplace=True),
            nn.Linear(dff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # x: (B, L, D)
        attn_out, _ = self.mha(x, x, x,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        x = self.norm2(x + ffn_out)
        return x  # (B, L, D)

# Positional Encoding (sinusoidal)
def sinusoidal_positional_encoding(seq_len, d_model, device=None, dtype=None):
    if device is None:
        device = torch.device('cpu')
    pe = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float)
                         * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)

class CatNet(nn.Module):
    def __init__(self, num_channels=12,
                 d_model=128, num_heads=4, dff=128, dropout_rate=0.2,
                 num_classes=11, apply_softmax=False):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=21, padding=(21-1)//2)
        # self.bn1 = nn.BatchNorm1d(32)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.ca1 = ChannelAttention(32, ratio=8)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # approx 'same' behaviour

        self.conv2 = nn.Conv1d(32, 32, kernel_size=23, padding=(23-1)//2)
        # self.bn2 = nn.BatchNorm1d(32)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.ca2 = ChannelAttention(32, ratio=8)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=25, padding=(25-1)//2)
        # self.bn3 = nn.BatchNorm1d(64)
        self.bn3 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.ca3 = ChannelAttention(64, ratio=8)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=27, padding=(27-1)//2)
        # self.bn4 = nn.BatchNorm1d(128)
        self.bn4 = nn.GroupNorm(num_groups=8, num_channels=128)
        self.ca4 = ChannelAttention(128, ratio=8)

        if 128 != d_model:
            self.project_to_d_model = nn.Conv1d(128, d_model, kernel_size=1)
        else:
            self.project_to_d_model = None

        # Transformer encoder
        self.transformer_block = TransformerEncoderBlock(d_model=d_model, num_heads=num_heads,
                                                         dff=dff, dropout_rate=dropout_rate)

        self.fc1 = None 
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(128, num_classes)

        self._init_weights()

        self._configured = False
        self.d_model = d_model

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_fc_head(self, sample_batch, device):
        self.eval()
        with torch.no_grad():
            x = sample_batch.to(device)
            features = self._forward_features(x)  # returns (B, L, D) after transformer
            flat = features.reshape(features.size(0), -1)
            in_features = flat.size(1)
            self.fc1 = nn.Linear(in_features, 128).to(device)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            self._configured = True


    def _forward_features(self, x):
        x = x.permute(0, 2, 1)  # -> (B, C, L)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.ca1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.ca2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.ca3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        x = self.ca4(x)

        if self.project_to_d_model is not None:
            x = self.project_to_d_model(x)  # (B, d_model, L_final)

        # Transpose for transformer: (B, L, D)
        x = x.permute(0, 2, 1)

        seq_len = x.size(1)
        pe = sinusoidal_positional_encoding(seq_len, self.d_model, device=x.device, dtype=x.dtype)
        x = x + pe  # (B, L, D)

        x = self.transformer_block(x)  # (B, L, D)
        return x

    def forward(self, x):
        device = x.device

        if not self._configured:
            self._init_fc_head(x, device)

        features = self._forward_features(x)  # (B, L, D)
        pooled = features.reshape(features.size(0), -1) 
        pooled = self.fc1(pooled)                  
        pooled = F.relu(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc_out(pooled)

        if self.apply_softmax:
            return F.softmax(logits, dim=-1)
        return logits