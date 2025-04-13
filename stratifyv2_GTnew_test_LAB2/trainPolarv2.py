import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import json
import csv

from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset

from sklearn.model_selection import train_test_split

# === Hyperparameters and configuration ===
input_size = 256
num_heads = 4
hidden_size = 512
heatmap_size = (200, 200)  # Dimension of the heatmap
batch_size = 1
num_epochs = 50

distance_variance = 1.0
angle_variance = 10.0
error_threshold = 4.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Model ===
model = HeatmapPredictionModel(num_heads=4, hidden_size=2048).to(device)

# Xavier initialization
def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(initialize_weights)

# Optimizer and scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# === Utility functions ===
def load_file_paths(directory):
    """
    Returns a list of subfolder paths within 'directory'
    """
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]

def save_paths_to_txt(paths, filename):
    with open(filename, 'w') as f:
        for path in paths:
            f.write(path + '\n')

# === Root data directory ===
data_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files'

# 1) Get all subfolders
all_paths = load_file_paths(data_dir)

# 2) Build "test_paths" for those containing 'LAB2' only
test_paths = [p for p in all_paths if 'LAB2' in p]

# 3) Remaining go to "train_val_paths"
train_val_paths = [p for p in all_paths if 'LAB2' not in p]

print(f"Nombre total de dossiers : {len(all_paths)}")
print(f"  -> test_paths  : {len(test_paths)}  (folders containing 'LAB2')")
print(f"  -> train+val   : {len(train_val_paths)} (all other folders)")

# 4) Split train/val on these train_val_paths (e.g., 20% validation)
train_paths, val_paths = train_test_split(train_val_paths, test_size=0.2, random_state=42)

print(f"  -> train_paths : {len(train_paths)}")
print(f"  -> val_paths   : {len(val_paths)}")

# 5) Save paths to .txt files (optional)
save_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_LAB2'
#os.makedirs(save_dir, exist_ok=True)

save_paths_to_txt(train_paths, os.path.join(save_dir, 'train_paths.txt'))
save_paths_to_txt(val_paths,   os.path.join(save_dir, 'val_paths.txt'))
save_paths_to_txt(test_paths,  os.path.join(save_dir, 'test_paths.txt'))

# 6) Build the Datasets and DataLoaders
train_datasets = [RadarDataset(sub_dir) for sub_dir in train_paths]
val_datasets   = [RadarDataset(sub_dir) for sub_dir in val_paths]
test_datasets  = [RadarDataset(sub_dir) for sub_dir in test_paths]

train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(ConcatDataset(val_datasets),   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(ConcatDataset(test_datasets),  batch_size=batch_size, shuffle=False)

print(f"Train DataLoader : {len(train_loader)} batches")
print(f"Val   DataLoader : {len(val_loader)} batches")
print(f"Test  DataLoader : {len(test_loader)} batches")

# === Training & validation loop ===
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    for (m1, p1, m2, p2, gt_heatmap) in train_loader:
        m1, p1, m2, p2, gt_heatmap = m1.to(device), p1.to(device), m2.to(device), p2.to(device), gt_heatmap.to(device)

        optimizer.zero_grad()
        heatmap_pred = model(m1, p1, m2, p2)
        loss = criterion(heatmap_pred, gt_heatmap)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for (m1, p1, m2, p2, gt_heatmap) in val_loader:
            m1, p1, m2, p2, gt_heatmap = m1.to(device), p1.to(device), m2.to(device), p2.to(device), gt_heatmap.to(device)
            heatmap_pred = model(m1, p1, m2, p2)
            val_loss = criterion(heatmap_pred, gt_heatmap)
            running_val_loss += val_loss.item()

    epoch_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    val_losses.append(epoch_val_loss)

    # Scheduler
    scheduler.step(epoch_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

# === Save the model ===
model_save_path = os.path.join(save_dir, 'modelpolar_stratifyv2_GTnew1_testlab2.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# === Plotting the loss curve ===
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses,   label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
loss_curve_path = os.path.join(save_dir, 'loss_curve_stratifyv2_GTnew1_testlab2.png')
plt.savefig(loss_curve_path)
plt.close()
print(f"Loss curve saved to {loss_curve_path}")
