"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset

# Définir le périphérique
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chargez le modèle entraîné
model = HeatmapPredictionModel(num_heads=4, hidden_size=2048)
model.load_state_dict(torch.load('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_ALL/modelpolar_combined_1p_2p.pth'))
model.to(device)
model.eval()

# Spécifiez le chemin vers le dossier spécifique que vous souhaitez tester
test_folder = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files/stand_90_degres_4m_1personnesv3_rep3'

# Créez une instance de RadarDataset pour ce dossier
test_dataset = RadarDataset(test_folder)

# Itération sur toutes les samples
results_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew/inference_samples'
os.makedirs(results_dir, exist_ok=True)

# Initialisez les listes pour stocker les distances, angles et probabilités
pred_distances = []
pred_angles = []
prob_sums = []

# Initialiser les listes pour stocker les coordonnées des centres des clusters
cluster_distances = []
cluster_angles = []

for sample_idx in range(len(test_dataset)):
    print(f"\nTraitement de l'échantillon {sample_idx + 1}/{len(test_dataset)}...")

    # Récupération des données pour l'échantillon courant
    magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap = test_dataset[sample_idx]
    
    # Ajoutez une dimension batch pour correspondre aux attentes du modèle
    magnitude1 = magnitude1.unsqueeze(0).to(device)
    phase1 = phase1.unsqueeze(0).to(device)
    magnitude2 = magnitude2.unsqueeze(0).to(device)
    phase2 = phase2.unsqueeze(0).to(device)
    ground_truth_heatmap = ground_truth_heatmap.unsqueeze(0).to(device)
    
    # Passez les données à travers le modèle pour obtenir la heatmap prédite
    with torch.no_grad():
        predicted_heatmap = model(magnitude1, phase1, magnitude2, phase2)
    
    # Déplacez les données sur le CPU et convertissez-les en numpy
    pred_heatmap_np = predicted_heatmap.cpu().numpy()[0]
    
    # Définir les bins de distance et d'angle
    heatmap_size = (200, 200)
    range_bins = np.linspace(0, 10, heatmap_size[0])  # De 0 à 10 mètres
    angle_bins = np.linspace(0, np.pi, heatmap_size[1])  # De 0 à pi en radians
    
    # Appliquez un seuil pour sélectionner les bins significatifs
    threshold = np.max(pred_heatmap_np) * 0.2  # 20% du maximum pour éliminer les faibles valeurs
    significant_coords = np.array(np.nonzero(pred_heatmap_np > threshold)).T  # Coordonnées des bins sélectionnées

    # Appliquez K-means clustering pour identifier le cluster principal
    if len(significant_coords) > 0:
        kmeans = KMeans(n_clusters=1, random_state=42).fit(significant_coords)
        cluster_centers = kmeans.cluster_centers_  # Centre du cluster principal
        cluster_center_idx = cluster_centers[0].astype(int)
    else:
        print(f"Aucun cluster significatif trouvé pour l'échantillon {sample_idx + 1}.")
        continue

    # Coordonnées du centre du cluster principal
    cluster_range = range_bins[cluster_center_idx[0]]
    cluster_angle = angle_bins[cluster_center_idx[1]]

    # Ajout des coordonnées des centres de clusters dans les listes
    cluster_distances.append(cluster_range)
    cluster_angles.append(cluster_angle)

    # Calcul du barycentre directement à partir de la heatmap
    normalized_heatmap = pred_heatmap_np / np.sum(pred_heatmap_np)  # Normalisation
    range_barycenter = np.sum(range_bins[:, None] * normalized_heatmap)  # Moyenne pondérée sur l'axe des portées
    angle_barycenter = np.sum(angle_bins[None, :] * normalized_heatmap)  # Moyenne pondérée sur l'axe des angles

    # Titre avec les prédictions (Distance et Angle)
    pred_title = f"Cluster_Center_Dist_{cluster_range:.2f}_Angle_{np.rad2deg(cluster_angle):.2f}"
    bary_title = f"Bary_Dist_{range_barycenter:.2f}_Angle_{np.rad2deg(angle_barycenter):.2f}"
    
    # Affichage polaire pour la prédiction
    r = range_bins
    theta = angle_bins
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    c = ax.pcolormesh(Theta, R, pred_heatmap_np, cmap='jet', shading='auto')
    plt.colorbar(c, ax=ax, label='Probabilité')
    ax.plot(cluster_angle, cluster_range, 'ro', label='Centre du Cluster (K-means)')  # Point rouge
    ax.plot(angle_barycenter, range_barycenter, 'go', label='Barycentre (Heatmap)')  # Point vert
    plt.legend()
    plt.title(pred_title)
    plt.savefig(os.path.join(results_dir, f'{pred_title}_polar_sample_{sample_idx + 1}.png'))
    plt.close()

    # Comparaison des valeurs
    print(f"Centre du Cluster (K-means) - Distance: {cluster_range:.2f} m, Angle: {np.rad2deg(cluster_angle):.2f}°")
    print(f"Barycentre - Distance: {range_barycenter:.2f} m, Angle: {np.rad2deg(angle_barycenter):.2f}°")
    
    # Stockez les valeurs pour calculer la moyenne pondérée
    pred_distances.append(range_barycenter)
    pred_angles.append(angle_barycenter)
    prob_sums.append(np.sum(pred_heatmap_np))  # Somme des probabilités dans la heatmap


# Calcul de la moyenne pondérée des positions globales
global_distance = np.sum(np.array(pred_distances) * np.array(prob_sums)) / np.sum(prob_sums)
global_angle_rad = np.sum(np.array(pred_angles) * np.array(prob_sums)) / np.sum(prob_sums)
global_angle_deg = np.rad2deg(global_angle_rad)

print(f"Position Prédite Globale (Moyenne) - Distance: {global_distance:.2f} m, Angle: {global_angle_deg:.2f}°")

# Médiane des distances et des angles
global_distance_median = np.median(pred_distances)
global_angle_median_rad = np.median(pred_angles)
global_angle_median_deg = np.rad2deg(global_angle_median_rad)

print(f"Position Prédite Globale (Médiane) - Distance: {global_distance_median:.2f} m, Angle: {global_angle_median_deg:.2f}°")

# Calcul de la moyenne et de la médiane des centres des clusters
global_cluster_distance = np.mean(cluster_distances)
global_cluster_angle_rad = np.mean(cluster_angles)
global_cluster_angle_deg = np.rad2deg(global_cluster_angle_rad)

print(f"Centre de Cluster Prédit (Moyenne) - Distance: {global_cluster_distance:.2f} m, Angle: {global_cluster_angle_deg:.2f}°")

global_cluster_distance_median = np.median(cluster_distances)
global_cluster_angle_median_rad = np.median(cluster_angles)
global_cluster_angle_median_deg = np.rad2deg(global_cluster_angle_median_rad)

print(f"Centre de Cluster Prédit (Médiane) - Distance: {global_cluster_distance_median:.2f} m, Angle: {global_cluster_angle_median_deg:.2f}°")
"""

#################################################################
######################## celui la est pour le test de 2 positions de 2 personnes !! #################################

import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, maximum_position
from sklearn.cluster import DBSCAN
import scipy
from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset

# Définir le périphérique (GPU si disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 1) Charger le modèle entraîné pour la détection de 1 ou 2 personnes
# =============================================================================
model_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_ALL/modelpolar_combined_1p_2p.pth'
model = HeatmapPredictionModel(num_heads=4, hidden_size=2048)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# =============================================================================
# 2) Charger le dataset de test pour 2 personnes
# =============================================================================
test_folder = '/store/wizar/HAR_code/radardataM/capteur/build/2personnes/sitting_p1_90_degres_1m_p2_45_degres_2m_rep3'
test_dataset = RadarDataset(test_folder)

# Extraction de la ground truth (GT) depuis le nom du dossier
folder_name = os.path.basename(test_folder)
match_p1 = re.search(r'p1_(\d+)_degres_(\d+)m', folder_name)
match_p2 = re.search(r'p2_(\d+)_degres_(\d+)m', folder_name)
if match_p1 and match_p2:
    gt_angle1 = float(match_p1.group(1))
    gt_distance1 = float(match_p1.group(2))
    gt_angle2 = float(match_p2.group(1))
    gt_distance2 = float(match_p2.group(2))
    print(f"GT Personne 1: {gt_distance1} m, {gt_angle1}°")
    print(f"GT Personne 2: {gt_distance2} m, {gt_angle2}°")
else:
    gt_angle1 = gt_distance1 = gt_angle2 = gt_distance2 = None
    print("Ground truth non trouvé dans le nom du dossier.")

# Dossier de sortie pour sauvegarder les images d'inférence individuelles (optionnel)
results_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_ALL/inference_samples_2clusters'
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# Initialiser les listes pour stocker les résultats et définir les paramètres
# =============================================================================
pred1_distances = []
pred1_angles = []
pred2_distances = []
pred2_angles = []
all_heatmaps = []  # Pour stocker les heatmaps prédites de chaque échantillon

# Paramètre DBSCAN (eps_distance est utilisé pour les deux dimensions)
eps_distance = 0.2  # rayon en mètres

def calculate_iou(mask1, mask2):
    #Calcul de l'Intersection over Union (IoU) entre deux masques.
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

# =============================================================================
# Boucle principale de traitement des échantillons du dataset de test
# =============================================================================
for sample_idx in range(len(test_dataset)):
    print(f"\nTraitement de l'échantillon {sample_idx + 1}/{len(test_dataset)}...")

    # Récupération des données d'entrée et de la heatmap de ground truth (non utilisée ici)
    magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap = test_dataset[sample_idx]
    magnitude1, phase1, magnitude2, phase2 = map(lambda x: x.unsqueeze(0).to(device),
                                                  [magnitude1, phase1, magnitude2, phase2])

    # Passage des données dans le modèle
    with torch.no_grad():
        predicted_heatmap = model(magnitude1, phase1, magnitude2, phase2)
    pred_heatmap_np = predicted_heatmap.cpu().numpy()[0]
    all_heatmaps.append(pred_heatmap_np)  # Stocker la heatmap pour la moyenne finale

    # Définir les bins pour la heatmap (les mêmes que lors de l'entraînement)
    heatmap_size = (200, 200)
    range_bins = np.linspace(0, 10, heatmap_size[0])
    angle_bins = np.linspace(0, np.pi, heatmap_size[1])

    # Identification des régions significatives à partir d'un seuil
    threshold = np.max(pred_heatmap_np) * 0.5
    significant_mask = pred_heatmap_np > threshold
    labeled_array, num_features = scipy.ndimage.label(significant_mask)

    # Localisation des pics par région
    peak_coords = maximum_position(pred_heatmap_np, labeled_array, range(1, num_features + 1))
    # Conversion des coordonnées en distance (m) et angle (°)
    peak_positions = [(range_bins[c[0]], np.rad2deg(angle_bins[c[1]])) for c in peak_coords]

    if len(peak_positions) < 2:
        print("Nombre insuffisant de pics détectés pour identifier 2 personnes.")
        continue

    # Regroupement des pics détectés avec DBSCAN
    peak_positions_np = np.array(peak_positions)
    dbscan = DBSCAN(eps=eps_distance, min_samples=1)
    labels = dbscan.fit_predict(peak_positions_np)

    clustered_peaks = []
    for label_val in set(labels):
        cluster = peak_positions_np[labels == label_val]
        avg_range = np.mean(cluster[:, 0])
        avg_angle = np.mean(cluster[:, 1])
        clustered_peaks.append((avg_range, avg_angle))

    # Pour identifier 2 personnes, trier les clusters par angle et prendre les 2 premiers
    if len(clustered_peaks) >= 2:
        clustered_peaks = sorted(clustered_peaks, key=lambda x: x[1])  # tri par angle
        # Personne 1 (par convention, la plus petite valeur d'angle)
        pred1_distances.append(clustered_peaks[0][0])
        pred1_angles.append(clustered_peaks[0][1])
        # Personne 2
        pred2_distances.append(clustered_peaks[1][0])
        pred2_angles.append(clustered_peaks[1][1])
        print(f"Échantillon {sample_idx + 1}:")
        print(f"  Personne 1 -> Distance: {clustered_peaks[0][0]:.2f} m, Angle: {clustered_peaks[0][1]:.2f}°")
        print(f"  Personne 2 -> Distance: {clustered_peaks[1][0]:.2f} m, Angle: {clustered_peaks[1][1]:.2f}°")
    else:
        print("Nombre insuffisant de clusters pour la détection de 2 personnes.")

    # Visualisation individuelle en coordonnées polaires (optionnelle)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    focused_heatmap = np.where(pred_heatmap_np > threshold, pred_heatmap_np, 0)
    R, Theta = np.meshgrid(range_bins, angle_bins, indexing='ij')
    c = ax.pcolormesh(Theta, R, focused_heatmap, cmap='jet', shading='auto')
    for avg_range, avg_angle in clustered_peaks[:2]:
        ax.plot(np.deg2rad(avg_angle), avg_range, 'ro')
    ax.set_title(f"Échantillon {sample_idx + 1}")
    plt.colorbar(c, ax=ax, label='Probabilité')
    sample_output_file = os.path.join(results_dir, f"sample_{sample_idx+1}.png")
    plt.savefig(sample_output_file)
    plt.close()

# =============================================================================
# Calcul des statistiques globales et de la heatmap moyenne sur tous les échantillons
# =============================================================================
if pred1_distances and pred2_distances:
    global_distance1 = np.mean(pred1_distances)
    global_angle1 = np.mean(pred1_angles)
    global_distance2 = np.mean(pred2_distances)
    global_angle2 = np.mean(pred2_angles)

    print("\n=== Résultats finaux ===")
    print(f"Personne 1 - Moyenne globale : Distance = {global_distance1:.2f} m, Angle = {global_angle1:.2f}°")
    print(f"Personne 2 - Moyenne globale : Distance = {global_distance2:.2f} m, Angle = {global_angle2:.2f}°")
else:
    print("Aucune prédiction consolidée pour 2 personnes.")

# Calcul de la heatmap moyenne sur l'ensemble des échantillons
avg_heatmap = np.mean(all_heatmaps, axis=0)

# =============================================================================
# Visualisation de la heatmap finale moyenne en coordonnées polaires
# avec les prédictions globales superposées
# =============================================================================
heatmap_size = avg_heatmap.shape  # par exemple (200, 200)
range_bins = np.linspace(0, 10, heatmap_size[0])
angle_bins = np.linspace(0, np.pi, heatmap_size[1])
R, Theta = np.meshgrid(range_bins, angle_bins, indexing='ij')

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)
c = ax.pcolormesh(Theta, R, avg_heatmap, cmap='jet', shading='auto')
plt.colorbar(c, ax=ax, label='Probability')
# Superposition des prédictions globales
ax.plot(np.deg2rad(global_angle1), global_distance1, 'ro',
        label=f"Person 1: {global_distance1:.2f} m, {global_angle1:.2f}°")
ax.plot(np.deg2rad(global_angle2), global_distance2, 'bo',
        label=f"Person 2: {global_distance2:.2f} m, {global_angle2:.2f}°")
ax.set_title(f"Predicted: Person (1): {global_distance1:.2f} m, {global_angle1:.2f}°; Person (2): {global_distance2:.2f} m, {global_angle2:.2f}°")
ax.legend(loc='upper right')
# Ligne d'enregistrement de la heatmap finale
final_output_file = os.path.join(results_dir, "final_heatmap_moyenne.png")
plt.savefig(final_output_file)
plt.show()


"""
################## celui la est pour le calcul des positions du test.txt dans le tableau
import os
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import re
from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = HeatmapPredictionModel(num_heads=4, hidden_size=2048)
model.load_state_dict(torch.load('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_ALL/modelpolar_combined_1p_2p_v2.pth'))
model.to(device)
model.eval()

# Function to parse two ground truth pairs from the folder name.
# Expected format: e.g., "sitting_p1_45_degres_1.5m_p2_90_degres_2m_lab2_rep3"
def parse_folder_name(folder_name):
    # Find all matches for pattern: p{number}_{angle}_degres_{distance}m
    matches = re.findall(r'p\d_(\d+)_degres_([\d.]+)m', folder_name)
    if len(matches) != 2:
        raise ValueError(f"Folder name {folder_name} does not contain exactly two ground truth pairs.")
    # Each match is (angle, distance) as strings; convert them to floats
    # We return tuples as (distance, angle)
    gt_pairs = [(float(m[1]), float(m[0])) for m in matches]
    # Sort by angle (or any other convention) so that the ordering is consistent
    gt_pairs = sorted(gt_pairs, key=lambda x: x[1])
    return gt_pairs

# Load test paths from the file
test_paths_file = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_ALL/test_paths.txt'
with open(test_paths_file, 'r') as f:
    test_paths = [line.strip() for line in f.readlines()]

# Final results for the CSV
results = []

# Process each folder in the test paths
for test_folder in test_paths:
    print(f"\nProcessing folder: {test_folder}...")

    try:
        # Extract two ground truth pairs from folder name.
        gt_pairs = parse_folder_name(os.path.basename(test_folder))
        # After sorting, assign:
        # Each ground truth is a tuple: (distance, angle)
        gt1 = gt_pairs[0]
        gt2 = gt_pairs[1]

        # Load the dataset for this folder
        test_dataset = RadarDataset(test_folder)

        # Temporary storage for predictions (one prediction per sample per person)
        predicted_distances1 = []
        predicted_angles1 = []
        predicted_distances2 = []
        predicted_angles2 = []

        # Process each sample in the current folder
        for sample_idx in range(len(test_dataset)):
            print(f"  Sample {sample_idx + 1}/{len(test_dataset)}")
            magnitude1, phase1, magnitude2, phase2, _ = test_dataset[sample_idx]
            magnitude1, phase1, magnitude2, phase2 = map(lambda x: x.unsqueeze(0).to(device), 
                                                          [magnitude1, phase1, magnitude2, phase2])

            with torch.no_grad():
                predicted_heatmap = model(magnitude1, phase1, magnitude2, phase2)
            pred_heatmap_np = predicted_heatmap.cpu().numpy()[0]

            # Define bins (assuming a heatmap size of 200x200)
            heatmap_size = (200, 200)
            range_bins = np.linspace(0, 10, heatmap_size[0])
            angle_bins = np.linspace(0, np.pi, heatmap_size[1])

            # Threshold the heatmap to keep only significant points
            threshold = np.max(pred_heatmap_np) * 0.2
            significant_coords = np.array(np.nonzero(pred_heatmap_np > threshold)).T

            if len(significant_coords) > 0:
                dbscan = DBSCAN(eps=0.5, min_samples=1)
                labels = dbscan.fit_predict(significant_coords)

                # For each cluster, compute the cluster center and convert to (range, angle)
                clustered_peaks = []
                for lbl in set(labels):
                    cluster_coords = significant_coords[labels == lbl]
                    # Compute the mean index (cluster center)
                    cluster_center_idx = cluster_coords.mean(axis=0).astype(int)
                    cluster_range = range_bins[cluster_center_idx[0]]
                    cluster_angle = np.rad2deg(angle_bins[cluster_center_idx[1]])
                    clustered_peaks.append((cluster_range, cluster_angle))

                # For two persons, we need at least two clusters.
                if len(clustered_peaks) >= 2:
                    # Sort clusters by angle (or any chosen criterion)
                    clustered_peaks = sorted(clustered_peaks, key=lambda x: x[1])
                    # Assign first cluster to Person 1 and second cluster to Person 2
                    predicted_distances1.append(clustered_peaks[0][0])
                    predicted_angles1.append(clustered_peaks[0][1])
                    predicted_distances2.append(clustered_peaks[1][0])
                    predicted_angles2.append(clustered_peaks[1][1])
                else:
                    print(f"    Insufficient clusters for two persons in sample {sample_idx + 1}.")
            else:
                print(f"    No significant cluster detected for sample {sample_idx + 1}.")

        # Average the predictions for each person over the samples in the folder
        if predicted_distances1 and predicted_angles1 and predicted_distances2 and predicted_angles2:
            avg_distance1 = np.mean(predicted_distances1)
            avg_angle1 = np.mean(predicted_angles1)
            avg_distance2 = np.mean(predicted_distances2)
            avg_angle2 = np.mean(predicted_angles2)
            
            # Compute errors for Person 1
            distance_error1 = abs(avg_distance1 - gt1[0])
            angle_error1 = abs(avg_angle1 - gt1[1])
            localization_error1 = np.sqrt(distance_error1**2 + np.deg2rad(angle_error1)**2)
            # Compute errors for Person 2
            distance_error2 = abs(avg_distance2 - gt2[0])
            angle_error2 = abs(avg_angle2 - gt2[1])
            localization_error2 = np.sqrt(distance_error2**2 + np.deg2rad(angle_error2)**2)

            # Create tuples for ground truth, predictions, and errors
            gt_tuple = (gt1, gt2)
            predicted_tuple = ((avg_distance1, avg_angle1), (avg_distance2, avg_angle2))
            error_tuple = ((distance_error1, angle_error1, localization_error1),
                           (distance_error2, angle_error2, localization_error2))
            
            results.append({
                "Folder": os.path.basename(test_folder),
                "GT": gt_tuple,
                "Predicted": predicted_tuple,
                "Errors": error_tuple
            })
        else:
            print(f"Insufficient predictions for folder {os.path.basename(test_folder)}.")

    except Exception as e:
        print(f"Error processing folder {test_folder}: {e}")
        continue

# Save results to a CSV file
output_csv_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_ALL/predictions_summary_ALL.csv'
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)
print(f"\nResults saved to: {output_csv_path}")


"""