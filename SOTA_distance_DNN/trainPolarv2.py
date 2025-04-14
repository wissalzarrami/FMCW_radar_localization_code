import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from Polarloader import RadarDataset  # Make sure this file contains your RadarDataset class.
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ---------------------------
# Model and Data Parameters
# ---------------------------

batch_size = 500
num_epochs = 100

# ---------------------------
# Device Setup
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------
# Xavier Initialization Function
# ---------------------------
def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ---------------------------
# Functions for Stratified Splitting
# ---------------------------
def load_file_paths(directory):
    # Return all subdirectory paths
    return [os.path.join(directory, d) for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))]

def extract_position_and_person_from_path(path):
    folder_name = os.path.basename(path)
    match = re.search(r'(\d+)_degres_([\d.]+)m.*?(v\d+)?_rep(\d+)', folder_name)
    person_match = re.search(r'personnes(v\d+)?', folder_name)
    if match:
        angle = int(match.group(1))
        distance = float(match.group(2))
        category = match.group(3) if match.group(3) else "v1"  # default category = v1 if not specified
        repetition = int(match.group(4))
        person = person_match.group(1) if person_match and person_match.group(1) else "v1"
        return (distance, angle, category, person, repetition)
    else:
        raise ValueError(f"Cannot extract attributes from folder name: {folder_name}")

# ---------------------------
# Load and Stratify Data Folders
# ---------------------------
data_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files'
all_paths = load_file_paths(data_dir)

# Group paths by (distance, angle, category, person)
position_person_groups = defaultdict(list)
for path in all_paths:
    key = extract_position_and_person_from_path(path)[:4]  # ignore repetition for grouping
    position_person_groups[key].append(path)

# Create stratified splits ensuring training > validation
train_paths = []
val_paths = []
test_paths = []

for key, paths in position_person_groups.items():
    distance, angle, category, person = key
    print(f"Processing group: distance={distance}, angle={angle}, category={category}, person={person} with {len(paths)} repetitions")
    
    # Sort paths by repetition for consistency.
    paths = sorted(paths, key=lambda p: extract_position_and_person_from_path(p)[4])
    n = len(paths)
    
    if n < 3:
        # If a group has fewer than 3 samples, assign all to training.
        train_paths.extend(paths)
    elif n == 3:
        # For 3 samples: use 2 for training and 1 for validation.
        train, temp = train_test_split(paths, test_size=1/3, random_state=42)
        train_paths.extend(train)
        val_paths.extend(temp)
    else:
        # For groups with 4 or more samples, split 70% training, 15% validation, 15% test.
        train, temp = train_test_split(paths, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train_paths.extend(train)
        val_paths.extend(val)
        test_paths.extend(test)

print(f"Total train: {len(train_paths)}")
print(f"Total val: {len(val_paths)}")
print(f"Total test: {len(test_paths)}")

# Function to save paths to text files
def save_paths_to_txt(paths, filename):
    with open(filename, 'w') as f:
        for path in paths:
            f.write(path + '\n')

# Save the paths
output_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/SOTA'
os.makedirs(output_dir, exist_ok=True)
save_paths_to_txt(train_paths, os.path.join(output_dir, 'train_paths.txt'))
save_paths_to_txt(val_paths, os.path.join(output_dir, 'val_paths.txt'))
save_paths_to_txt(test_paths, os.path.join(output_dir, 'test_paths.txt'))

print(f"Train paths saved to: {os.path.join(output_dir, 'train_paths.txt')}")
print(f"Validation paths saved to: {os.path.join(output_dir, 'val_paths.txt')}")
print(f"Test paths saved to: {os.path.join(output_dir, 'test_paths.txt')}")

# ---------------------------
# Create Datasets and DataLoaders
# ---------------------------
train_datasets = [RadarDataset(sub_dir) for sub_dir in train_paths]
val_datasets = [RadarDataset(sub_dir) for sub_dir in val_paths]
test_datasets = [RadarDataset(sub_dir) for sub_dir in test_paths]

train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=batch_size, shuffle=False)

print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Validation DataLoader: {len(val_loader)} batches")
print(f"Test DataLoader: {len(test_loader)} batches")

# ---------------------------
# Infer Input Dimension from a Sample
# ---------------------------
sample = train_datasets[0][0]  # (magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap)
mag1, ph1, mag2, ph2, _ = sample
input_dim = mag1.numel() + ph1.numel() + mag2.numel() + ph2.numel()
print(f"Input dimension: {input_dim}")

# ---------------------------
# Initialize the Model, Loss, Optimizer, and Scheduler
# ---------------------------
from model import SimpleDNNClassifier  # Make sure your model is defined in model.py

# Initialize model with output_dim=10
model = SimpleDNNClassifier(input_dim=input_dim, output_dim=10)
model.to(device)

criterion = nn.CrossEntropyLoss()  # Classification loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ... [Previous code up to model initialization] ...

# Initialize lists to store losses
train_losses = []
val_losses = []
val_maes = []

# Class centers for MAE calculation
class_centers = torch.linspace(1.0, 5.0, 10).to(device)  # [1.0, 1.5, ..., 5.0]

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_train_loss = 0.0
    for batch in train_loader:
        magnitude1, phase1, magnitude2, phase2, labels = batch
        magnitude1 = magnitude1.to(device)
        phase1 = phase1.to(device)
        magnitude2 = magnitude2.to(device)
        phase2 = phase2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(magnitude1, phase1, magnitude2, phase2)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * labels.size(0)
    
    # Calculate average training loss
    train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    running_val_loss = 0.0
    running_mae = 0.0
    with torch.no_grad():
        for batch in val_loader:
            magnitude1, phase1, magnitude2, phase2, labels = batch
            magnitude1 = magnitude1.to(device)
            phase1 = phase1.to(device)
            magnitude2 = magnitude2.to(device)
            phase2 = phase2.to(device)
            labels = labels.to(device)
            
            logits = model(magnitude1, phase1, magnitude2, phase2)
            
            # Calculate validation loss
            loss = criterion(logits, labels)
            running_val_loss += loss.item() * labels.size(0)
            
            # Calculate MAE
            pred_classes = torch.argmax(logits, dim=1)
            pred_distances = class_centers[pred_classes]
            true_distances = class_centers[labels]
            running_mae += torch.sum(torch.abs(pred_distances - true_distances)).item()
    
    # Calculate validation metrics
    val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    
    val_mae = running_mae / len(val_loader.dataset)
    val_maes.append(val_mae)
    
    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"  Val MAE: {val_mae:.4f} meters")
    print("-" * 50)

# ... [Rest of the code remains the same] ...
# Save the trained model
model_save_path = os.path.join(output_dir, 'simple_dnn_model_stratified.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# ---------------------------
# Plot Loss Curves
# ---------------------------
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
loss_curve_path = os.path.join(output_dir, 'loss_curve_stratified.png')
plt.savefig(loss_curve_path)
plt.show()
print(f"Loss curve saved to {loss_curve_path}")
