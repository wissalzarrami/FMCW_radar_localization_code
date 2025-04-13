""" 
################ no person 3, train and validation = person 1,2 , test=person3
import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit

# Importation du modèle et du dataset
from modelPolarv1 import DualAntennaSiameseModel  # Assurez-vous que ce modèle correspond bien à votre version pour 2 antennes
from PolarLoader import RadarDataset

# ------------------------------
# Paramètres du modèle et de l'entraînement
# ------------------------------
input_length = 256                # Longueur du vecteur 1D en entrée
num_heads = 4                     # Nombre de têtes du Transformer
base_transformer_d_model = 512    # Doit correspondre à la sortie désirée du DualAntennaFullModel
heatmap_size = 200                # Taille (hauteur = largeur) de la heatmap en sortie
batch_size = 1
num_epochs = 50
learning_rate = 1e-4
weight_decay = 1e-5

# ------------------------------
# Initialisation du modèle et du device
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DualAntennaSiameseModel(
    in_channels=1, 
    base_channels=32, 
    transformer_d_model=64,   # ou 512 si nécessaire
    nhead=4, 
    heatmap_size=200, 
    input_length=input_length,
    target_seq_length=256
)
model.to(device)

# ------------------------------
# Fonction d'initialisation Xavier
# ------------------------------
def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(initialize_weights)

# ------------------------------
# Définition de la fonction de perte, de l'optimiseur et du scheduler
# ------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# ------------------------------
# Fonctions utilitaires pour charger et stratifier les chemins des données
# ------------------------------
def load_file_paths(directory):
    #Charge les chemins de tous les sous-dossiers (chaque sous-dossier correspond à un enregistrement radar).
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

def extract_position_and_repetition(path):
    
    #Extrait l'angle, la distance et la répétition à partir du nom du dossier.
  #  Exemple de nom attendu : "stand_30_degres_2.5m_..._rep3"
    
    match = re.search(r'stand_(\d+)_degres_([\d.]+)m_.*?_rep(\d+)', path)
    if match:
        angle = int(match.group(1))
        distance = float(match.group(2))
        repetition = int(match.group(3))
        return angle, distance, repetition
    return None, None, None

def stratify_non_v3_data(paths, val_size=0.2):
    
    #Stratifie les dossiers ne contenant pas 'v3' (personnes 1 et 2) pour obtenir des ensembles d'entraînement et de validation.
    #Si une classe (position) est présente avec moins de 2 exemplaires, on effectue une séparation aléatoire.
    
    positions = []
    for path in paths:
        angle, distance, _ = extract_position_and_repetition(path)
        if angle is not None and distance is not None:
            positions.append(f"{angle}_{distance}")
        else:
            positions.append("unknown")
    positions = np.array(positions)
    paths = np.array(paths)

    # Vérifier la fréquence de chaque classe
    unique, counts = np.unique(positions, return_counts=True)
    if np.any(counts < 2):
        print("Warning: Some classes have less than 2 members; using a random split without stratification.")
        from sklearn.model_selection import train_test_split
        train_paths, val_paths = train_test_split(paths, test_size=val_size, random_state=42)
        return train_paths.tolist(), val_paths.tolist()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    for train_idx, val_idx in sss.split(paths, positions):
        train_paths = paths[train_idx]
        val_paths = paths[val_idx]
    
    return train_paths.tolist(), val_paths.tolist()


# ------------------------------
# Chargement et séparation des données
# ------------------------------
data_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files'
all_paths = load_file_paths(data_dir)

# Séparation en deux groupes :
# - Les dossiers de la personne 3 contiennent 'v3'
# - Les dossiers des personnes 1 et 2 ne contiennent pas 'v3'
v3_paths = [path for path in all_paths if 'v3' in path]         # Personne 3 -> test
non_v3_paths = [path for path in all_paths if 'v3' not in path]   # Personnes 1 & 2 -> entraînement et validation

#print("v3_paths",len(v3_paths))
print ("non_v3_paths",len(non_v3_paths))
# Stratification sur les dossiers des personnes 1 & 2 pour obtenir train et validation
train_paths, val_paths = stratify_non_v3_data(non_v3_paths, val_size=0.2)

# Le test sera effectué uniquement sur les dossiers de la personne 3
test_paths = v3_paths

# Affichage de la répartition
print(f"Total train (personnes 1 & 2): {len(train_paths)}")
print(f"Total validation (personnes 1 & 2): {len(val_paths)}")
print(f"Total test (personne 3): {len(test_paths)}")

# Optionnel : Sauvegarder les chemins dans des fichiers texte
def save_paths_to_txt(paths, filename):
    with open(filename, 'w') as f:
        for path in paths:
            f.write(path + '\n')

save_paths_to_txt(train_paths, '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/train_paths.txt')
save_paths_to_txt(val_paths, '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/val_paths.txt')
save_paths_to_txt(test_paths, '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/test_paths.txt')

# Création des datasets et DataLoaders
train_datasets = [RadarDataset(sub_dir) for sub_dir in train_paths]
val_datasets = [RadarDataset(sub_dir) for sub_dir in val_paths]
test_datasets = [RadarDataset(sub_dir) for sub_dir in test_paths]

train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=batch_size, shuffle=False)

print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Validation DataLoader: {len(val_loader)} batches")
print(f"Test DataLoader: {len(test_loader)} batches")

# ------------------------------
# Boucle d'entraînement et de validation
# ------------------------------
train_loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    # Boucle d'entraînement
    for i, (magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap) in enumerate(train_loader):
        # Envoi des données sur le device
        magnitude1 = magnitude1.to(device)
        phase1 = phase1.to(device)
        magnitude2 = magnitude2.to(device)
        phase2 = phase2.to(device)
        ground_truth_heatmap = ground_truth_heatmap.to(device)

        optimizer.zero_grad()
        heatmap_pred = model(magnitude1, phase1, magnitude2, phase2)
        loss = criterion(heatmap_pred, ground_truth_heatmap)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)
    train_loss_history.append(train_loss)

    # Boucle de validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap in val_loader:
            magnitude1 = magnitude1.to(device)
            phase1 = phase1.to(device)
            magnitude2 = magnitude2.to(device)
            phase2 = phase2.to(device)
            ground_truth_heatmap = ground_truth_heatmap.to(device)

            heatmap_pred = model(magnitude1, phase1, magnitude2, phase2)
            loss_val = criterion(heatmap_pred, ground_truth_heatmap)
            running_val_loss += loss_val.item()

    val_loss = running_val_loss / len(val_loader)
    val_loss_history.append(val_loss)

    scheduler.step(val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Sauvegarde du modèle entraîné
model_save_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/modelpolar_stratifyv2_GTnew1.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# ------------------------------
# Visualisation des courbes de perte
# ------------------------------
plt.figure()
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_loss_history, label='Train Loss')
plt.plot(epochs, val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
loss_curve_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/loss_curve_stratifyv2_GTnew1.png'
plt.savefig(loss_curve_path)
print(f"Loss curve saved to {loss_curve_path}")

################ train, val=person1,person2, person3   test= labo 2 ####################################################################################################
import os
import re
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit

# Importation du modèle et du dataset
from modelPolarv1 import DualAntennaSiameseModel  # Assurez-vous que ce modèle correspond bien à votre version pour 2 antennes
from PolarLoader import RadarDataset

# ------------------------------
# Paramètres du modèle et de l'entraînement
# ------------------------------
input_length = 256                # Longueur du vecteur 1D en entrée
num_heads = 4                     # Nombre de têtes du Transformer
base_transformer_d_model = 512    # Doit correspondre à la sortie désirée du DualAntennaFullModel
heatmap_size = 200                # Taille (hauteur = largeur) de la heatmap en sortie
batch_size = 1
num_epochs = 50
learning_rate = 1e-4
weight_decay = 1e-5

# ------------------------------
# Initialisation du modèle et du device
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DualAntennaSiameseModel(
    in_channels=1, 
    base_channels=32, 
    transformer_d_model=64,   # ou 512 si nécessaire
    nhead=4, 
    heatmap_size=200, 
    input_length=input_length,
    target_seq_length=256
)
model.to(device)

# ------------------------------
# Fonction d'initialisation Xavier
# ------------------------------
def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(initialize_weights)

# ------------------------------
# Définition de la fonction de perte, de l'optimiseur et du scheduler
# ------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# ------------------------------
# Fonctions utilitaires pour charger et stratifier les chemins des données
# ------------------------------
def load_file_paths(directory):
    
    #Charge les chemins de tous les sous-dossiers (chaque sous-dossier correspond à un enregistrement radar).
    
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

def extract_position_and_repetition(path):
    
    #Extrait l'angle, la distance et la répétition à partir du nom du dossier.
   # Exemple de nom attendu : "stand_30_degres_2.5m_..._rep3"
    
    match = re.search(r'stand_(\d+)_degres_([\d.]+)m_.*?_rep(\d+)', path)
    if match:
        angle = int(match.group(1))
        distance = float(match.group(2))
        repetition = int(match.group(3))
        return angle, distance, repetition
    return None, None, None

def stratify_data(paths, val_size=0.2):
    
   # Stratifie les dossiers (hors LAB2) pour obtenir des ensembles d'entraînement et de validation.
   # Pour chaque dossier, on extrait la position (angle et distance) qui sert de classe.
    #Si une classe a moins de 2 exemplaires, une séparation aléatoire est effectuée.
    
    positions = []
    for path in paths:
        angle, distance, _ = extract_position_and_repetition(path)
        if angle is not None and distance is not None:
            positions.append(f"{angle}_{distance}")
        else:
            positions.append("unknown")
    positions = np.array(positions)
    paths = np.array(paths)

    # Vérifier la fréquence de chaque classe
    unique, counts = np.unique(positions, return_counts=True)
    if np.any(counts < 2):
        print("Warning: Certaines classes ont moins de 2 membres; utilisation d'une séparation aléatoire sans stratification.")
        from sklearn.model_selection import train_test_split
        train_paths, val_paths = train_test_split(paths, test_size=val_size, random_state=42)
        return train_paths.tolist(), val_paths.tolist()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    for train_idx, val_idx in sss.split(paths, positions):
        train_paths = paths[train_idx]
        val_paths = paths[val_idx]
    
    return train_paths.tolist(), val_paths.tolist()

# ------------------------------
# Chargement et séparation des données
# ------------------------------
data_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files'
all_paths = load_file_paths(data_dir)
print("Nombre total de dossiers :", len(all_paths))

# Séparation des dossiers :
# - Les dossiers contenant "LAB2" vont dans le test set
# - Tous les autres dossiers seront utilisés pour l'entraînement et la validation
test_paths = [path for path in all_paths if "LAB2" in path]
non_test_paths = [path for path in all_paths if "LAB2" not in path]

print("Nombre de dossiers LAB2 (test) :", len(test_paths))
print("Nombre de dossiers non-LAB2 (pour train+validation) :", len(non_test_paths))

# Stratification sur les dossiers non-LAB2
train_paths, val_paths = stratify_data(non_test_paths, val_size=0.2)

print(f"Total train: {len(train_paths)}")
print(f"Total validation: {len(val_paths)}")
print(f"Total test: {len(test_paths)}")

# Optionnel : Sauvegarder les chemins dans des fichiers texte
def save_paths_to_txt(paths, filename):
    with open(filename, 'w') as f:
        for path in paths:
            f.write(path + '\n')

save_paths_to_txt(train_paths, '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/train_paths_EXPlab.txt')
save_paths_to_txt(val_paths, '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/val_paths_EXPlab.txt')
save_paths_to_txt(test_paths, '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/test_paths_EXPlab.txt')

# Création des datasets et DataLoaders
train_datasets = [RadarDataset(sub_dir) for sub_dir in train_paths]
val_datasets = [RadarDataset(sub_dir) for sub_dir in val_paths]
test_datasets = [RadarDataset(sub_dir) for sub_dir in test_paths]

train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=batch_size, shuffle=False)

print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Validation DataLoader: {len(val_loader)} batches")
print(f"Test DataLoader: {len(test_loader)} batches")

# ------------------------------
# Boucle d'entraînement et de validation
# ------------------------------
train_loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    # Boucle d'entraînement
    for i, (magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap) in enumerate(train_loader):
        # Envoi des données sur le device
        magnitude1 = magnitude1.to(device)
        phase1 = phase1.to(device)
        magnitude2 = magnitude2.to(device)
        phase2 = phase2.to(device)
        ground_truth_heatmap = ground_truth_heatmap.to(device)

        optimizer.zero_grad()
        heatmap_pred = model(magnitude1, phase1, magnitude2, phase2)
        loss = criterion(heatmap_pred, ground_truth_heatmap)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)
    train_loss_history.append(train_loss)

    # Boucle de validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap in val_loader:
            magnitude1 = magnitude1.to(device)
            phase1 = phase1.to(device)
            magnitude2 = magnitude2.to(device)
            phase2 = phase2.to(device)
            ground_truth_heatmap = ground_truth_heatmap.to(device)

            heatmap_pred = model(magnitude1, phase1, magnitude2, phase2)
            loss_val = criterion(heatmap_pred, ground_truth_heatmap)
            running_val_loss += loss_val.item()

    val_loss = running_val_loss / len(val_loader)
    val_loss_history.append(val_loss)

    scheduler.step(val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Sauvegarde du modèle entraîné
model_save_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/modelpolar_stratifyv2_GTnew1_EXPlab.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# ------------------------------
# Visualisation des courbes de perte
# ------------------------------
plt.figure()
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_loss_history, label='Train Loss')
plt.plot(epochs, val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
loss_curve_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/loss_curve_stratifyv2_GTnew1_EXPlab.png'
plt.savefig(loss_curve_path)
print(f"Loss curve saved to {loss_curve_path}")
"""
###################################### test with all ##########################################################
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
import os
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
# Importation du modèle et du dataset
from modelPolarv1 import DualAntennaSiameseModel  # Vérifiez que ce modèle correspond bien à votre version pour 2 antennes
from PolarLoader import RadarDataset

# ------------------------------
# Paramètres du modèle et de l'entraînement
# ------------------------------
input_length = 256                # Longueur du vecteur 1D en entrée
num_heads = 4                     # Nombre de têtes du Transformer
base_transformer_d_model = 512    # Doit correspondre à la sortie désirée du DualAntennaFullModel
heatmap_size = 200                # Taille (hauteur = largeur) de la heatmap en sortie
batch_size = 1
num_epochs = 50
learning_rate = 1e-4
weight_decay = 1e-5

# ------------------------------
# Initialisation du modèle et du device
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DualAntennaSiameseModel(
    in_channels=1, 
    base_channels=32, 
    transformer_d_model=64,   # ou 512 si nécessaire
    nhead=4, 
    heatmap_size=200, 
    input_length=input_length,
    target_seq_length=256
)
model.to(device)

# ------------------------------
# Fonction d'initialisation Xavier
# ------------------------------
def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(initialize_weights)

# ------------------------------
# Définition de la fonction de perte, de l'optimiseur et du scheduler
# ------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)


# Fonction pour charger les chemins des sous-dossiers
def load_file_paths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

# Fonction pour extraire les positions (distance, angle, catégorie, personne, répétition) depuis les noms des dossiers
def extract_position_and_person_from_path(path):
    folder_name = os.path.basename(path)
    match = re.search(r'(\d+)_degres_([\d.]+)m.*?(v\d+)?_rep(\d+)', folder_name)
    person_match = re.search(r'personnes(v\d+)?', folder_name)
    
    if match:
        angle = int(match.group(1))
        distance = float(match.group(2))
        category = match.group(3) if match.group(3) else "v1"  # Par défaut, catégorie = "v1" si non spécifié
        repetition = int(match.group(4))
        person = person_match.group(1) if person_match else "v1"  # Si pas de personne spécifiée, "v1"
        return (distance, angle, category, person, repetition)
    else:
        raise ValueError(f"Impossible d'extraire la position depuis le chemin: {folder_name}")

# Charger tous les chemins
data_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files'
all_paths = load_file_paths(data_dir)

# Regrouper les chemins par position (distance, angle, catégorie, personne)
position_person_groups = defaultdict(list)
for path in all_paths:
    position_and_person = extract_position_and_person_from_path(path)[:4]  # Ignorer la répétition pour le regroupement
    position_person_groups[position_and_person].append(path)

# Répartition stratifiée par position/personne
train_paths = []
val_paths = []
test_paths = []

for (distance, angle, category, person), paths in position_person_groups.items():
    # Affichage pour déboguer
    print(f"Traitement de la position: distance={distance}, angle={angle}, catégorie={category}, personne={person}")
    
    # Trier les chemins par répétition pour consistance
    paths = sorted(paths, key=lambda p: extract_position_and_person_from_path(p)[4])  # Trier par répétition

    if len(paths) == 1:
        # Si une seule répétition, elle va dans l'entraînement
        train_paths.extend(paths)
    elif len(paths) == 2:
        # Si deux répétitions, une va dans l'entraînement, l'autre dans val/test
        train, val = train_test_split(paths, test_size=0.5, random_state=42)
        train_paths.extend(train)
        val_paths.extend(val)
    elif len(paths) == 3:
        # Si trois répétitions, répartir 1 ou 2 dans train, et les autres entre val/test
        train, temp = train_test_split(paths, test_size=1/3, random_state=42)
        if len(temp) > 1:
            val, test = train_test_split(temp, test_size=0.5, random_state=42)
            val_paths.extend(val)
            test_paths.extend(test)
        else:
            val_paths.extend(temp)
    else:
        # Cas général : répartir entre train, val et test
        train, temp = train_test_split(paths, test_size=0.4, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train_paths.extend(train)
        val_paths.extend(val)
        test_paths.extend(test)

# Vérification des répartitions
print(f"Total train: {len(train_paths)}")
print(f"Total val: {len(val_paths)}")
print(f"Total test: {len(test_paths)}")

# Fonction pour sauvegarder les chemins dans des fichiers texte
def save_paths_to_txt(paths, filename):
    with open(filename, 'w') as f:
        for path in paths:
            f.write(path + '\n')

# Sauvegarder les chemins
output_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY'

save_paths_to_txt(train_paths, os.path.join(output_dir, 'train_paths.txt'))
save_paths_to_txt(val_paths, os.path.join(output_dir, 'val_paths.txt'))
save_paths_to_txt(test_paths, os.path.join(output_dir, 'test_paths.txt'))

print(f"Train paths saved to: {os.path.join(output_dir, 'train_paths.txt')}")
print(f"Validation paths saved to: {os.path.join(output_dir, 'val_paths.txt')}")
print(f"Test paths saved to: {os.path.join(output_dir, 'test_paths.txt')}")

# Création des datasets et DataLoaders
train_datasets = [RadarDataset(sub_dir) for sub_dir in train_paths]
val_datasets = [RadarDataset(sub_dir) for sub_dir in val_paths]
test_datasets = [RadarDataset(sub_dir) for sub_dir in test_paths]

train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=batch_size, shuffle=False)

# Vérification des DataLoaders
print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Validation DataLoader: {len(val_loader)} batches")
print(f"Test DataLoader: {len(test_loader)} batches")



"""
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
model_save_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/modelpolar_stratifyv2_GTnew_ALL.pth'
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
loss_curve_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/loss_curve_stratifyv2_GTnew1_ALL.png'
plt.savefig(loss_curve_path)
plt.show()  # Optionally display the plot
print(f"Loss curve saved to {loss_curve_path}")
"""