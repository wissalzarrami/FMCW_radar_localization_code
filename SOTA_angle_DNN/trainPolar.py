import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Import du modèle complet avec détection et estimation
from model import FullDOANetwork
from Polarloader import CustomLoader

def load_file_paths(directory):
    """
    Retourne la liste des chemins vers les sous-dossiers présents dans 'directory'.
    Chaque sous-dossier doit contenir les fichiers nécessaires (_a1.cf32 et _a2.cf32).
    """
    subfolders = []
    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        if os.path.isdir(full_path):
            subfolders.append(full_path)
    return subfolders

def build_tensor_dataset(paths):
    """
    Pour une liste de chemins de dossiers, construit un TensorDataset (X, y).
      - X : tenseur de forme (num_samples, feature_dim) issu du vecteur y extrait.
      - y : tenseur de forme (num_samples,) contenant l'étiquette DOA (extrait du nom du dossier).
    """
    combined_X = []
    combined_y = []
    
    for folder in paths:
        # Instanciation du loader pour le dossier courant
        loader = CustomLoader(folder)
        loader.load_data()
        # Chargement du premier échantillon (on suppose un échantillon par dossier)
        sample = loader.load_sample(0)  # sample est un np.array de type float32
        X_tensor = torch.from_numpy(sample)  # forme : (feature_dim,)
        
        # Extraction de l'étiquette à partir du nom du dossier (exemple : "23_degres")
        match = re.search(r'(\d+)_degres', os.path.basename(folder))
        if match:
            label = int(match.group(1))
        else:
            label = 0  # valeur par défaut
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        combined_X.append(X_tensor.unsqueeze(0))
        combined_y.append(label_tensor.unsqueeze(0))
    
    X_all = torch.cat(combined_X, dim=0)  # forme : (num_samples, feature_dim)
    y_all = torch.cat(combined_y, dim=0)    # forme : (num_samples,)
    return TensorDataset(X_all, y_all)

def train_model(
    train_dataset, 
    val_dataset,
    input_dim,
    Q,
    hidden_dims,
    output_dim,
    epochs=10,
    batch_size=4,
    lr=1e-3
):
    """
    Entraîne le réseau complet FullDOANetwork (détection + estimation) en mode classification.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = FullDOANetwork(input_dim, Q, hidden_dims, output_dim)
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.view(X_batch.size(0), -1)  # mise à plat si besoin
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)

        # Évaluation sur le jeu de validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.view(X_val.size(0), -1)
                val_out = model(X_val)
                loss_val = criterion(val_out, y_val)
                val_loss += loss_val.item()
                predicted = val_out.argmax(dim=1)
                correct += (predicted == y_val).sum().item()
                total   += y_val.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model

if __name__ == "__main__":
    # 1) Chemin vers le dossier principal contenant plusieurs sous-dossiers
    data_dir = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files"
    all_paths = load_file_paths(data_dir)
    print(f"Found {len(all_paths)} sub-folders total.")

    # 2) Séparation en 80 % train et 20 % validation
    train_paths, val_paths = train_test_split(all_paths, test_size=0.2, random_state=42)
    print(f"Total train folders: {len(train_paths)}, val folders: {len(val_paths)}")

    # 3) Construction des TensorDatasets
    train_ds = build_tensor_dataset(train_paths)
    val_ds   = build_tensor_dataset(val_paths)
    print(f"train_ds length: {len(train_ds)} samples, val_ds length: {len(val_ds)} samples.")

    # 4) Définition des hyperparamètres
    # Pour M=2, la dimension du vecteur y est : 2 (diagonale) + 2 (élément hors diag) = 4.
    input_dim = 4  
    # Q : nombre de secteurs angulaires (à ajuster selon votre application, ici par exemple 10)
    Q = 10  
    # Pour le réseau d'estimation, deux couches cachées avec 50 et 30 neurones respectivement.
    hidden_dims = [50, 30]
    # output_dim : nombre de classes (angles discrets), par exemple 180 pour des angles de 0 à 179°.
    output_dim = 180  
      
    # 5) Entraînement du modèle complet
    model = train_model(
        train_dataset=train_ds,
        val_dataset=val_ds,
        input_dim=input_dim,
        Q=Q,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        epochs=1000,
        batch_size=8,
        lr=1e-4
    )
    
    # 6) Sauvegarde du modèle entraîné
    torch.save(model.state_dict(), "doa_estimation_network.pth")
    print("Model saved to doa_estimation_network.pth")
