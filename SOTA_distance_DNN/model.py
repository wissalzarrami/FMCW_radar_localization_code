import torch
import torch.nn as nn

class SimpleDNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=10):  # 10 classes
        super(SimpleDNNClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)  # 10 output neurons
        )
    
    def forward(self, magnitude1, phase1, magnitude2, phase2):
        # Flatten and concatenate inputs
        x = torch.cat([
            magnitude1.view(magnitude1.size(0), -1),
            phase1.view(phase1.size(0), -1),
            magnitude2.view(magnitude2.size(0), -1),
            phase2.view(phase2.size(0), -1)
        ], dim=1)
        return self.fc(x)