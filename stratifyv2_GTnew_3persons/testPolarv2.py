""""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, maximum_position
from sklearn.cluster import DBSCAN
from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset
import scipy

# Définir le périphérique
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chargez le modèle entraîné
model = HeatmapPredictionModel(num_heads=4, hidden_size=2048)
model.load_state_dict(torch.load('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_3persons/modelpolar_stratifyv2_GTnew_3persons.pth'))
model.to(device)
model.eval()

# Spécifiez le chemin vers le dossier spécifique que vous souhaitez tester
test_folder = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files/stand_90_degres_4m_1personnesv3_rep3'

# Créez une instance de RadarDataset pour ce dossier
test_dataset = RadarDataset(test_folder)

# Itération sur toutes les samples
results_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_3persons/inference_samples'
os.makedirs(results_dir, exist_ok=True)


# Initialiser les listes pour stocker les résultats
pred_distances = []
pred_angles = []

# Paramètres DBSCAN
eps_distance = 0.5  # Rayon en mètres
eps_angle = 10  # Rayon en degrés

# Fonction utilitaire
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

# Boucle principale pour traiter chaque échantillon
for sample_idx in range(len(test_dataset)):
    print(f"\nTraitement de l'échantillon {sample_idx + 1}/{len(test_dataset)}...")

    # Récupération des données
    magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap = test_dataset[sample_idx]
    magnitude1, phase1, magnitude2, phase2 = map(lambda x: x.unsqueeze(0).to(device), [magnitude1, phase1, magnitude2, phase2])

    # Passez les données au modèle
    with torch.no_grad():
        predicted_heatmap = model(magnitude1, phase1, magnitude2, phase2)
    pred_heatmap_np = predicted_heatmap.cpu().numpy()[0]

    # Définir les bins
    heatmap_size = (200, 200)
    range_bins = np.linspace(0, 10, heatmap_size[0])
    angle_bins = np.linspace(0, np.pi, heatmap_size[1])

    # Identification des régions significatives
    threshold = np.max(pred_heatmap_np) * 0.5
    significant_mask = pred_heatmap_np > threshold
    labeled_array, num_features = scipy.ndimage.label(significant_mask)

    # Localisation des pics
    peak_coords = maximum_position(pred_heatmap_np, labeled_array, range(1, num_features + 1))
    peak_positions = [(range_bins[c[0]], np.rad2deg(angle_bins[c[1]])) for c in peak_coords]

    if not peak_positions:
        print("Aucun pic détecté.")
        continue

    # Regroupement avec DBSCAN
    peak_positions_np = np.array(peak_positions)
    dbscan = DBSCAN(eps=eps_distance, min_samples=1)
    labels = dbscan.fit_predict(peak_positions_np)

    clustered_peaks = []
    for label in set(labels):
        cluster = peak_positions_np[labels == label]
        avg_range = np.mean(cluster[:, 0])
        avg_angle = np.mean(cluster[:, 1])
        clustered_peaks.append((avg_range, avg_angle))

    # Sauvegarde des résultats
    for avg_range, avg_angle in clustered_peaks:
        pred_distances.append(avg_range)
        pred_angles.append(avg_angle)

        # Afficher les résultats pour l'échantillon
        print(f"Échantillon {sample_idx + 1}: Distance moyenne du cluster = {avg_range:.2f} m, Angle moyen du cluster = {avg_angle:.2f}°")

    # Visualisation améliorée
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    
    # Filtrer la heatmap pour se concentrer sur des zones étroites
    focused_heatmap = np.where(pred_heatmap_np > threshold, pred_heatmap_np, 0)

    # Création de la visualisation polaire
    R, Theta = np.meshgrid(range_bins, angle_bins, indexing='ij')
    c = ax.pcolormesh(Theta, R, focused_heatmap, cmap='jet', shading='auto')

    # Ajouter les clusters détectés
    for avg_range, avg_angle in clustered_peaks:
        ax.plot(np.deg2rad(avg_angle), avg_range, 'ro', label=f"Cluster consolidé: {avg_range:.2f}m, {avg_angle:.2f}°")

    # Titre et échelle
    pred_title = f"Dist={avg_range:.2f}m, Angle={avg_angle:.2f}°"
    ax.set_title(pred_title)
    plt.colorbar(c, ax=ax, label='Probabilité')

    # Sauvegarder l'image avec le titre comme nom
    output_file = os.path.join(results_dir, f"sample_{sample_idx + 1}_dist_{avg_range:.2f}_angle_{avg_angle:.2f}.png")
    plt.savefig(output_file)
    plt.close()

# Moyennes globales
if pred_distances:
    global_distance = np.mean(pred_distances)
    global_angle = np.mean(pred_angles)
    global_distance_median = np.median(pred_distances)
    global_angle_median = np.median(pred_angles)

    print("\n=== Résultats finaux ===")
    print(f"Moyenne globale - Distance : {global_distance:.2f} m, Angle : {global_angle:.2f}°")
    print(f"Médiane globale - Distance : {global_distance_median:.2f} m, Angle : {global_angle_median:.2f}°")
else:
    print("Aucune prédiction consolidée.")

"""
################## celui la est pour le calcul des positions du test.txt dans le tableau

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset
import os
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.ndimage import label, maximum_position
from sklearn.cluster import DBSCAN
from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset
import scipy
import pandas as pd

# Définir le périphérique
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chargez le modèle entraîné
model = HeatmapPredictionModel(num_heads=4, hidden_size=2048)
model.load_state_dict(torch.load('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_3persons/modelpolar_stratifyv2_GTnew_3persons.pth'))
model.to(device)
model.eval()

# Function to parse ground truths from folder name
def parse_folder_name(folder_name):
    print(f"Parsing folder name: {folder_name}")  # Debug: display folder name
    match = re.search(r'(\d+)_degres_([\d.]+)m', folder_name)
    if match:
        angle = float(match.group(1))
        distance = float(match.group(2))
        return distance, angle
    else:
        raise ValueError(f"Folder name {folder_name} does not contain the ground truth distance and angle.")

# Load test paths from the file
test_paths_file = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_3persons/pos_inter_paths.txt'
with open(test_paths_file, 'r') as f:
    test_paths = [line.strip() for line in f.readlines()]

# Final results for the CSV
results = []

# Process each path in the file
for test_folder in test_paths:
    print(f"\nProcessing folder: {test_folder}...")

    try:
        # Extract ground truths from folder name
        gt_distance, gt_angle = parse_folder_name(os.path.basename(test_folder))

        # Load the dataset
        test_dataset = RadarDataset(test_folder)

        for sample_idx in range(len(test_dataset)):
            print(f"\nProcessing sample {sample_idx + 1}/{len(test_dataset)}...")

            # Retrieve the data
            magnitude1, phase1, magnitude2, phase2, _ = test_dataset[sample_idx]
            magnitude1, phase1, magnitude2, phase2 = map(lambda x: x.unsqueeze(0).to(device), [magnitude1, phase1, magnitude2, phase2])

            # Pass the data through the model
            with torch.no_grad():
                predicted_heatmap = model(magnitude1, phase1, magnitude2, phase2)
            pred_heatmap_np = predicted_heatmap.cpu().numpy()[0]

            # Define bins
            heatmap_size = (200, 200)
            range_bins = np.linspace(0, 10, heatmap_size[0])
            angle_bins = np.linspace(0, np.pi, heatmap_size[1])

            # Identify significant regions
            threshold = np.max(pred_heatmap_np) * 0.2
            significant_coords = np.array(np.nonzero(pred_heatmap_np > threshold)).T

            if len(significant_coords) > 0:
                # Clustering with DBSCAN
                dbscan = DBSCAN(eps=0.5, min_samples=1)
                labels = dbscan.fit_predict(significant_coords)

                # Calculate cluster centers
                clustered_peaks = []
                for label in set(labels):
                    cluster_coords = significant_coords[labels == label]
                    cluster_center_idx = cluster_coords.mean(axis=0).astype(int)
                    cluster_range = range_bins[cluster_center_idx[0]]
                    cluster_angle = np.rad2deg(angle_bins[cluster_center_idx[1]])
                    clustered_peaks.append((cluster_range, cluster_angle))

                # Calculate the median of clusters for the final position
                median_distance = np.median([peak[0] for peak in clustered_peaks])
                median_angle = np.median([peak[1] for peak in clustered_peaks])

                # Calculate errors
                distance_error = abs(median_distance - gt_distance)
                angle_error = abs(median_angle - gt_angle)
                localization_error = np.sqrt(distance_error**2 + np.deg2rad(angle_error)**2)

                # Add results to the list
                results.append({
                    "Folder": os.path.basename(test_folder),
                    "GT Distance (m)": round(gt_distance, 3),
                    "GT Angle (°)": round(gt_angle, 3),
                    "Predicted Distance (m)": round(median_distance, 3),
                    "Predicted Angle (°)": round(median_angle, 3),
                    "Distance Error (m)": round(distance_error, 3),
                    "Angle Error (°)": round(angle_error, 3),
                    "Localization Error (m)": round(localization_error, 3)
                })

            else:
                print(f"No significant cluster detected for sample {sample_idx + 1}.")
                continue

    except Exception as e:
        print(f"Error processing folder {test_folder}: {e}")
        continue

# Regrouper les résultats par dossier pour calculer la moyenne
results_df = pd.DataFrame(results)

# Calculer les moyennes des prédictions pour chaque dossier
grouped_results = results_df.groupby("Folder").agg({
    "GT Distance (m)": "mean",
    "GT Angle (°)": "mean",
    "Predicted Distance (m)": "mean",
    "Predicted Angle (°)": "mean"
}).reset_index()

# Calculer les erreurs finales pour chaque dossier
grouped_results["Distance Error (m)"] = abs(grouped_results["Predicted Distance (m)"] - grouped_results["GT Distance (m)"])
grouped_results["Angle Error (°)"] = abs(grouped_results["Predicted Angle (°)"] - grouped_results["GT Angle (°)"])
grouped_results["Localization Error (m)"] = np.sqrt(
    grouped_results["Distance Error (m)"]**2 + 
    np.deg2rad(grouped_results["Angle Error (°)"])**2
)

# Arrondir les valeurs à 2 décimales
columns_to_round = [
    "GT Distance (m)", 
    "GT Angle (°)", 
    "Predicted Distance (m)", 
    "Predicted Angle (°)", 
    "Distance Error (m)", 
    "Angle Error (°)", 
    "Localization Error (m)"
]
grouped_results[columns_to_round] = grouped_results[columns_to_round].round(2)

# Enregistrer les résultats dans un fichier CSV final
final_output_csv_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_3persons/final_predictions_3persons_pos_inter.csv'
grouped_results.to_csv(final_output_csv_path, index=False)

print(f"Résultats finaux enregistrés dans : {final_output_csv_path}")


