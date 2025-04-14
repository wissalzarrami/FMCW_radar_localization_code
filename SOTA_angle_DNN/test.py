import os
import torch
import torch.nn as nn
import numpy as np
import csv
import re

from model import FullDOANetwork  # Utilisation du modèle complet (détection + estimation)
from Polarloader import CustomLoader

def load_paths_from_file(file_path):
    """
    Lit les chemins depuis un fichier et retourne une liste.
    """
    with open(file_path, 'r') as file:
        paths = file.readlines()
    return [path.strip() for path in paths]

def extract_angle_from_folder_name(folder_name):
    """
    Extrait l'angle du nom du dossier, en supposant un format tel que 'stand_90_degres_2.5m'.
    """
    match = re.search(r'stand_(\d+)_degres', folder_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Angle not found in folder name: {folder_name}")

def write_results_to_csv(results, csv_filename="inference_results.csv"):
    """
    Écrit les résultats dans un fichier CSV.
    """
    header = ["Folder", "Estimated DOA", "Ground Truth DOA", "Error (degrees)"]
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    # -------------------------------
    # 1) Charger les chemins depuis le fichier
    # -------------------------------
    paths_file = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew__2labs/test_paths_WORK_WITH_THIS.txt"
    inference_folders = load_paths_from_file(paths_file)
    print(f"Found {len(inference_folders)} folders to test.")

    # -------------------------------
    # 2) Charger le modèle pré-entraîné
    # -------------------------------
    # Les paramètres doivent être identiques à ceux utilisés lors de l'entraînement
    input_dim = 4          # Dimension du vecteur y (pour M=2)
    Q = 10                 # Nombre de secteurs angulaires
    hidden_dims = [50, 30] # Dimensions cachées pour le réseau d'estimation
    output_dim = 180       # Nombre de classes (angles de 0 à 179°)

    model = FullDOANetwork(input_dim, Q, hidden_dims, output_dim)
    model_path = "doa_estimation_network.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Modèle chargé et en mode évaluation.")

    # -------------------------------
    # 3) Inférence et collecte des résultats
    # -------------------------------
    results = []
    for folder in inference_folders:
        print(f"\nTraitement du dossier : {folder}")
        # Extraction de la ground truth à partir du nom du dossier
        try:
            ground_truth_doa = extract_angle_from_folder_name(folder)
        except ValueError as e:
            print(e)
            continue  # On passe ce dossier si l'extraction de l'angle échoue

        # Chargement des données avec le CustomLoader
        loader = CustomLoader(folder)
        loader.load_data()
        sample = loader.load_sample(0)  # vecteur numpy de type float32

        # Préparation de l'entrée pour le modèle
        x = torch.from_numpy(sample).float().unsqueeze(0)  # forme : (1, feature_dim)

        with torch.no_grad():
            output = model(x)  # La sortie est déjà softmaxée dans l'EstimationNetwork
            predicted_label = torch.argmax(output, dim=1).item()
            confidence = output[0, predicted_label].item()

        # Calcul de l'erreur en degrés
        error = abs(predicted_label - ground_truth_doa)
        print(f"Prédiction : {predicted_label} (confiance : {confidence:.4f}), Ground Truth : {ground_truth_doa}, Erreur : {error}°")
        results.append([folder, predicted_label, ground_truth_doa, error])

    # -------------------------------
    # 4) Sauvegarde des résultats dans un fichier CSV
    # -------------------------------
    csv_filename = "inference_results_LAST.csv"
    write_results_to_csv(results, csv_filename)
    print(f"Résultats enregistrés dans {csv_filename}")
