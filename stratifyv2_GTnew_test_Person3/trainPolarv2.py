"""
############################# this is where person 3 in the test set and person1+2 in the train/val###############
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

# === Hyperparamètres et configuration ===
input_size = 256
num_heads = 4
hidden_size = 512
heatmap_size = (200, 200)  # Dimension de la heatmap
batch_size = 1
num_epochs = 50

distance_variance = 1.0
angle_variance = 10.0
error_threshold = 4.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Modèle ===
model = HeatmapPredictionModel(num_heads=4, hidden_size=2048).to(device)

# Xavier initialization
def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(initialize_weights)

# Optimiseur et scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# === Fonctions utilitaires ===
def load_file_paths(directory):
    
    #Retourne la liste des sous-dossiers (paths) dans 'directory'
    
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]

def save_paths_to_txt(paths, filename):
    with open(filename, 'w') as f:
        for path in paths:
            f.write(path + '\n')

# === Chemin racine des données ===
data_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files'

# 1) Récupère tous les sous-dossiers
all_paths = load_file_paths(data_dir)

# 2) Sépare en "test_paths" (pers3/v3) et "train_val_paths" (pers1+p2)
test_paths = [p for p in all_paths if 'v3' in p or 'pers3' in p]
train_val_paths = [p for p in all_paths if p not in test_paths]

print(f"Nombre total de dossiers : {len(all_paths)}")
print(f"  -> test_paths  : {len(test_paths)} (pers3/v3)")
print(f"  -> train+val : {len(train_val_paths)} (pers1 + pers2)")

# 3) Split train/val sur le set pers1+p2
#    Ici, par exemple, on met 20% en validation
train_paths, val_paths = train_test_split(train_val_paths, test_size=0.2, random_state=42)

print(f"  -> train_paths : {len(train_paths)}")
print(f"  -> val_paths   : {len(val_paths)}")

# 4) Sauvegarde des chemins dans des fichiers .txt (optionnel)
save_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_Person3'
os.makedirs(save_dir, exist_ok=True)

save_paths_to_txt(train_paths, os.path.join(save_dir, 'train_paths.txt'))
save_paths_to_txt(val_paths,   os.path.join(save_dir, 'val_paths.txt'))
save_paths_to_txt(test_paths,  os.path.join(save_dir, 'test_paths.txt'))

# 5) Construit les Datasets puis DataLoaders
train_datasets = [RadarDataset(sub_dir) for sub_dir in train_paths]
val_datasets   = [RadarDataset(sub_dir) for sub_dir in val_paths]
test_datasets  = [RadarDataset(sub_dir) for sub_dir in test_paths]

train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(ConcatDataset(val_datasets),   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(ConcatDataset(test_datasets),  batch_size=batch_size, shuffle=False)

print(f"Train DataLoader : {len(train_loader)} batches")
print(f"Val   DataLoader : {len(val_loader)} batches")
print(f"Test  DataLoader : {len(test_loader)} batches")

# === Boucle d'entraînement & validation ===
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

# === Sauvegarde du modèle ===
model_save_path = os.path.join(save_dir, 'modelpolar_stratifyv2_GTnew1_jdid.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# === Visualisation de la courbe de loss ===
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses,   label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
loss_curve_path = os.path.join(save_dir, 'loss_curve_stratifyv2_GTnew1_jdid.png')
plt.savefig(loss_curve_path)
plt.close()
print(f"Loss curve saved to {loss_curve_path}")



############################# this is where person 2 in the test set and person1+3 in the train/val###############
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import csv

from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset
from sklearn.model_selection import train_test_split

# === Hyperparameters and config ===
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

# Optimizer and Scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# === Utility functions ===
def load_file_paths(directory):
    # Returns a list of subfolder paths in 'directory'
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]

def save_paths_to_txt(paths, filename):
    with open(filename, 'w') as f:
        for path in paths:
            f.write(path + '\n')

# === Root directory for data ===
data_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files'

# 1) Get all subdirectories
all_paths = load_file_paths(data_dir)

# 2) We'll use Person 2 as the test set, i.e., subfolders that contain 'v2' or 'pers2'
test_paths = [p for p in all_paths if 'v2' in p or 'pers2' in p]
train_val_paths = [p for p in all_paths if p not in test_paths]

print(f"Total folders: {len(all_paths)}")
print(f"  -> test_paths : {len(test_paths)} (Person 2)")
print(f"  -> train+val  : {len(train_val_paths)} (Persons 1 + 3)")

# 3) Split train/val from persons 1+3
train_paths, val_paths = train_test_split(train_val_paths, test_size=0.2, random_state=42)

print(f"  -> train_paths : {len(train_paths)}")
print(f"  -> val_paths   : {len(val_paths)}")

# 4) Save subfolder paths to .txt (optional)
save_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_Person2'
os.makedirs(save_dir, exist_ok=True)

save_paths_to_txt(train_paths, os.path.join(save_dir, 'train_paths.txt'))
save_paths_to_txt(val_paths,   os.path.join(save_dir, 'val_paths.txt'))
save_paths_to_txt(test_paths,  os.path.join(save_dir, 'test_paths.txt'))

# 5) Build Datasets and DataLoaders
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

    # Scheduler step
    scheduler.step(epoch_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

# === Save the model ===
model_save_path = os.path.join(save_dir, 'modelpolar_stratifyv2_GTnew1_jdid2.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# === Save loss curve ===
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses,   label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
loss_curve_path = os.path.join(save_dir, 'loss_curve_stratifyv2_GTnew1_jdid2.png')
plt.savefig(loss_curve_path)
plt.close()
print(f"Loss curve saved to {loss_curve_path}")
"""
###### Test Personne1 and train + val on Personne2+Personne3############################
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import csv

from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset
from sklearn.model_selection import train_test_split

# === Hyperparameters and config ===
input_size = 256
num_heads = 4
hidden_size = 512
heatmap_size = (200, 200)  # Dimension de la heatmap
batch_size = 1
num_epochs = 50

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

# Optimizer and Scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# === Utility functions ===
def load_file_paths(directory):
    # Returns a list of subfolder paths in 'directory'
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]

def save_paths_to_txt(paths, filename):
    with open(filename, 'w') as f:
        for path in paths:
            f.write(path + '\n')

# === Root directory for data ===
data_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files'

# 1) Get all subdirectories
all_paths = load_file_paths(data_dir)

# 2) We'll use Person 2 + Person 3 as train/val (i.e., subfolders that contain 'v2' or 'v3').
#    The rest is for test.
train_val_paths = [
    p for p in all_paths
    if ('v2' in p.lower() or 'v3' in p.lower())
]
test_paths = [p for p in all_paths if p not in train_val_paths]

print(f"Total folders: {len(all_paths)}")
print(f"  -> train+val : {len(train_val_paths)} (contain v2 or v3)")
print(f"  -> test      : {len(test_paths)} (the rest)")

# 3) Split train/val from v2 + v3
train_paths, val_paths = train_test_split(train_val_paths, test_size=0.2, random_state=42)

print(f"  -> train_paths : {len(train_paths)}")
print(f"  -> val_paths   : {len(val_paths)}")

# 4) Save subfolder paths to .txt (optional)
save_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_person1'
os.makedirs(save_dir, exist_ok=True)

save_paths_to_txt(train_paths, os.path.join(save_dir, 'train_paths.txt'))
save_paths_to_txt(val_paths,   os.path.join(save_dir, 'val_paths.txt'))
save_paths_to_txt(test_paths,  os.path.join(save_dir, 'test_paths.txt'))

# 5) Build Datasets and DataLoaders
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

    # Scheduler step
    scheduler.step(epoch_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

# === Save the model ===
model_save_path = os.path.join(save_dir, 'modelpolar_stratifyv2_trVal_Pers2_3_test_REST.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# === Save loss curve ===
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses,   label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss (Train+Val on v2+v3, Test on remaining folders)')
loss_curve_path = os.path.join(save_dir, 'loss_curve_stratifyv2_trVal_Pers2_3_test_REST.png')
plt.savefig(loss_curve_path)
plt.close()
print(f"Loss curve saved to {loss_curve_path}")
