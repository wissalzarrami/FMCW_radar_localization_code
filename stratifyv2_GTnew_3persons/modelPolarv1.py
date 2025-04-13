
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

# Positional Encoding for transformers (to capture temporal/spatial relations)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=262144):  # Adjust max_len to match your largest seq_len
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

# CNN Block for feature extraction with additional convolution layer
class CNNBlock(nn.Module):
    def __init__(self, in_channels):
        super(CNNBlock, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)  # Downsample by a factor of 8
        # Additional convolution layer for further feature extraction
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)  # Increase channels after pooling

    def forward(self, x):
        # Apply first convolution
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = self.pool(x)  # Further downsampling
        # Apply second convolution for more complex feature extraction
        x = F.leaky_relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Change to [batch_size, seq_len, features]
        return x

# MLP to process magnitude and phase features separately for each antenna
class FeatureFusionMLP(nn.Module):
    def __init__(self, input_size):
        super(FeatureFusionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512)  # Output size will match the transformer input size
        )

    def forward(self, magnitude_feat, phase_feat):
        # Concatenate magnitude and phase features and pass through MLP
        combined_feat = torch.cat([magnitude_feat, phase_feat], dim=2)  # Concatenate features along the last dimension
        return self.mlp(combined_feat)

# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size):
        super(TransformerEncoderBlock, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model=input_size)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True
        )

    def forward(self, x):
        x = self.positional_encoding(x)
        x = checkpoint(self.transformer, x)
        return x

class HeatmapPredictionModel(nn.Module):
    def __init__(self, num_heads, hidden_size, heatmap_size=(200, 200)):
        super(HeatmapPredictionModel, self).__init__()

        # Adjust in_channels to match input data channels if necessary
        self.cnn_magnitude_antenna1 = CNNBlock(in_channels=1)
        self.cnn_phase_antenna1 = CNNBlock(in_channels=1)
        self.cnn_magnitude_antenna2 = CNNBlock(in_channels=1)
        self.cnn_phase_antenna2 = CNNBlock(in_channels=1)

        # MLPs for each antenna (for magnitude and phase fusion)
        self.mlp_fusion_antenna1 = FeatureFusionMLP(input_size=1024)  # Adjust input size for concatenation
        self.mlp_fusion_antenna2 = FeatureFusionMLP(input_size=1024)

        # Transformer blocks for each antenna
        self.transformer_antenna1 = TransformerEncoderBlock(512, num_heads, hidden_size)
        self.transformer_antenna2 = TransformerEncoderBlock(512, num_heads, hidden_size)

        # Fully connected layers to predict a 2D heatmap (for range and angle)
        self.fc_heatmap = nn.Linear(512, heatmap_size[0] * heatmap_size[1])  # 512 -> 40000

        # Additional linear layer to reduce dimensionality to 512 before positional encoding
        self.linear_after_concat = nn.Linear(1024, 512)  # Reduce from 1024 to 512

        # Store heatmap dimensions
        self.heatmap_size = heatmap_size

    def forward(self, magnitude1, phase1, magnitude2, phase2):
        # Ensure inputs have the correct shape
        if len(magnitude1.shape) == 2:
            magnitude1 = magnitude1.unsqueeze(1)  # Shape: [batch_size, 1, seq_len]
        if len(phase1.shape) == 2:
            phase1 = phase1.unsqueeze(1)
        if len(magnitude2.shape) == 2:
            magnitude2 = magnitude2.unsqueeze(1)
        if len(phase2.shape) == 2:
            phase2 = phase2.unsqueeze(1)

        # Process magnitude and phase for antenna 1 through CNNs
        magnitude1_feat = self.cnn_magnitude_antenna1(magnitude1)  # [batch_size, seq_len, 512]
        phase1_feat = self.cnn_phase_antenna1(phase1)  # [batch_size, seq_len, 512]

        # Process magnitude and phase for antenna 2 through CNNs
        magnitude2_feat = self.cnn_magnitude_antenna2(magnitude2)  # [batch_size, seq_len, 512]
        phase2_feat = self.cnn_phase_antenna2(phase2)  # [batch_size, seq_len, 512]

        # Use MLP to fuse features for both antennas
        # Concatenate the features before passing them through MLP (size will be [batch_size, seq_len, 1024])
        antenna1_fused_feat = self.mlp_fusion_antenna1(magnitude1_feat, phase1_feat)
        antenna2_fused_feat = self.mlp_fusion_antenna2(magnitude2_feat, phase2_feat)

        # Apply positional encoding and transformer to fused features
        antenna1_transformer_out = self.transformer_antenna1(antenna1_fused_feat)  # [batch_size, seq_len, 512]
        antenna2_transformer_out = self.transformer_antenna2(antenna2_fused_feat)  # [batch_size, seq_len, 512]

        # Concatenate the transformer outputs for both antennas along feature dimension
        # Now, fusion_out should have shape [batch_size, seq_len, 1024]
        fusion_out = torch.cat([antenna1_transformer_out, antenna2_transformer_out], dim=2)  # [batch_size, seq_len, 1024]

        # Reduce dimensionality from 1024 to 512
        fusion_out = self.linear_after_concat(fusion_out)  # Now fusion_out is [batch_size, seq_len, 512]

        # Apply global average pooling over seq_len dimension
        fusion_out = fusion_out.mean(dim=1)  # Now fusion_out is [batch_size, 512]

        # Predict the heatmap from the fusion_out
        heatmap_pred = self.fc_heatmap(fusion_out)  # [batch_size, heatmap_size[0]*heatmap_size[1]]

        # Reshape the heatmap prediction to the correct 2D shape
        heatmap_pred = heatmap_pred.view(fusion_out.size(0), self.heatmap_size[0], self.heatmap_size[1])

        return heatmap_pred
