import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import csv
import os
import os
import re
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader
# Model and data parameters
input_size = 256
num_heads = 4
hidden_size = 512
heatmap_size = (200, 200)  # Ground truth heatmap size
batch_size = 1
num_epochs = 50
distance_variance = 1.0  # Gaussian distribution variance for distance
angle_variance = 10.0    # Gaussian distribution variance for angle
error_threshold = 4.0

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HeatmapPredictionModel(num_heads=4, hidden_size=2048)
model.summary()
model.to(device)

# Xavier initialization function
def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Apply Xavier initialization
model.apply(initialize_weights)
from sklearn.model_selection import StratifiedShuffleSplit


# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

import re
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit



import os
from sklearn.model_selection import train_test_split

# Fonction pour charger les chemins des sous-dossiers
def load_file_paths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

# Chemin des données
data_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files'

# Charger tous les chemins
all_paths = load_file_paths(data_dir)

# Diviser les données en 80% train et 20% validation
train_paths, val_paths = train_test_split(all_paths, test_size=0.2, random_state=42)

# Vérification des répartitions
print(f"Total train: {len(train_paths)}")
print(f"Total val: {len(val_paths)}")

# Fonction pour sauvegarder les chemins dans des fichiers texte
def save_paths_to_txt(paths, filename):
    with open(filename, 'w') as f:
        for path in paths:
            f.write(path + '\n')

# Sauvegarder les chemins
output_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_3persons'

save_paths_to_txt(train_paths, os.path.join(output_dir, 'train_paths.txt'))
save_paths_to_txt(val_paths, os.path.join(output_dir, 'val_paths.txt'))

print(f"Train paths saved to: {os.path.join(output_dir, 'train_paths.txt')}")
print(f"Validation paths saved to: {os.path.join(output_dir, 'val_paths.txt')}")

train_datasets = [RadarDataset(sub_dir) for sub_dir in train_paths]
val_datasets = [RadarDataset(sub_dir) for sub_dir in val_paths]

train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False)

# Vérification des DataLoaders
print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Validation DataLoader: {len(val_loader)} batches")






# Training and validation loop
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    # Training loop
    for i, (magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap) in enumerate(train_loader):
        magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap = \
            magnitude1.to(device), phase1.to(device), magnitude2.to(device), phase2.to(device), ground_truth_heatmap.to(device)

        optimizer.zero_grad()
        heatmap_pred = model(magnitude1, phase1, magnitude2, phase2)
        loss = criterion(heatmap_pred, ground_truth_heatmap)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    # Average training loss for the epoch
    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap in val_loader:
            magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap = \
                magnitude1.to(device), phase1.to(device), magnitude2.to(device), phase2.to(device), ground_truth_heatmap.to(device)

            heatmap_pred = model(magnitude1, phase1, magnitude2, phase2)
            val_loss = criterion(heatmap_pred, ground_truth_heatmap)
            running_val_loss += val_loss.item()

    # Average validation loss for the epoch
    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)

    # Step the learning rate scheduler
    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Save the model after training
model_save_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_3persons/modelpolar_stratifyv2_GTnew_3persons_v2.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Visualize loss curves
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
loss_curve_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew/loss_curve_stratifyv2_GTnew1_3persons_v2.png'
plt.savefig(loss_curve_path)
plt.show()  # Optionally display the plot
print(f"Loss curve saved to {loss_curve_path}")
