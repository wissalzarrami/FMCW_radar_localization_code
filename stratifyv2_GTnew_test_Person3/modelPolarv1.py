 ############# hedha loul elli me fichou mlp 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

from torch.utils.checkpoint import checkpoint

# Transformer Encoder Block for each antenna
class CNNBlock(nn.Module):
    def __init__(self, in_channels):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)  # Downsample by a factor of 8

    def forward(self, x):
        x = F.leaky_relu(self.conv(x))
        x = self.pool(x)
        x = self.pool(x)# Reduce sequence length
        x = x.permute(0, 2, 1)  # Change to [batch_size, seq_len, features]
        return x

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


# Heatmap Prediction Model with adjusted CNNBlock in_channels
class HeatmapPredictionModel(nn.Module):
    def __init__(self, num_heads, hidden_size, heatmap_size=(200, 200)):
        super(HeatmapPredictionModel, self).__init__()

        # Adjust in_channels to match input data channels if necessary
        self.cnn_magnitude_antenna1 = CNNBlock(in_channels=1)
        self.cnn_phase_antenna1 = CNNBlock(in_channels=1)
        self.cnn_magnitude_antenna2 = CNNBlock(in_channels=1)
        self.cnn_phase_antenna2 = CNNBlock(in_channels=1)

        # Transformer blocks for each antenna
        self.transformer_antenna1 = TransformerEncoderBlock(512, num_heads, hidden_size)
        self.transformer_antenna2 = TransformerEncoderBlock(512, num_heads, hidden_size)

        # Fully connected layers to predict a 2D heatmap (for range and angle)
        self.fc_projection = nn.Linear(1024, 1024)
        self.fc_heatmap = nn.Linear(1024, heatmap_size[0] * heatmap_size[1])

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
        
       # print(f"magnitude1 shape: {magnitude1.shape}")
       # print(f"phase1 shape: {phase1.shape}")
       # print(f"magnitude2 shape: {magnitude2.shape}")
       # print(f"phase2 shape: {phase2.shape}")

        # Process magnitude and phase for antenna 1 through CNNs
        magnitude1_feat = self.cnn_magnitude_antenna1(magnitude1)
        phase1_feat = self.cnn_phase_antenna1(phase1)

        # Process magnitude and phase for antenna 2 through CNNs
        magnitude2_feat = self.cnn_magnitude_antenna2(magnitude2)
        phase2_feat = self.cnn_phase_antenna2(phase2)

        # Concatenate magnitude and phase features for each antenna along feature dimension
        antenna1_feat = torch.cat([magnitude1_feat, phase1_feat], dim=2)  # [batch_size, seq_len, 512]
        antenna2_feat = torch.cat([magnitude2_feat, phase2_feat], dim=2)  # [batch_size, seq_len, 512]

        # Pass the features through the transformer blocks for both antennas
        antenna1_transformer_out = self.transformer_antenna1(antenna1_feat)
        antenna2_transformer_out = self.transformer_antenna2(antenna2_feat)

        # Concatenate the transformer outputs for both antennas along feature dimension
        fusion_out = torch.cat([antenna1_transformer_out, antenna2_transformer_out], dim=2)  # [batch_size, seq_len, 1024]

        # Apply global average pooling over seq_len dimension
        fusion_out = fusion_out.mean(dim=1)  # Now fusion_out is [batch_size, 1024]

        # Pass through the projection layer
        fusion_out = self.fc_projection(fusion_out)

        # Predict the heatmap from the fusion_out
        heatmap_pred = self.fc_heatmap(fusion_out)

        # Reshape the heatmap prediction to the correct 2D shape
        heatmap_pred = heatmap_pred.view(fusion_out.size(0), self.heatmap_size[0], self.heatmap_size[1])

        return heatmap_pred
