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
model.load_state_dict(torch.load('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew/modelpolar_stratifyv2_GTnew1.pth'))
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
#################################################################
######################## celui la est pour le test dune position seule !! #################################
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

# Charger le modèle entraîné
model = HeatmapPredictionModel(num_heads=4, hidden_size=2048)
model.load_state_dict(torch.load('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew/modelpolar_stratifyv2_GTnew1.pth'))
model.to(device)
model.eval()

# Spécifiez le chemin vers le dossier spécifique que vous souhaitez tester
test_folder = '/store/wizar/HAR_code/radardataM/capteur/build/Lab2/stand_90_degres_2m_1personnesLAB2_rep3'
test_dataset = RadarDataset(test_folder)

results_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew/inference_samples'
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
model.load_state_dict(torch.load('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew/modelpolar_stratifyv2_GTnew1.pth'))
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
#################################################################
######################## celui la est pour le test dune position seule !! #################################
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

# Charger le modèle entraîné
model = HeatmapPredictionModel(num_heads=4, hidden_size=2048)
model.load_state_dict(torch.load('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew/modelpolar_stratifyv2_GTnew1.pth'))
model.to(device)
model.eval()

# Spécifiez le chemin vers le dossier spécifique que vous souhaitez tester
test_folder = '/store/wizar/HAR_code/radardataM/capteur/build/Lab2/stand_90_degres_2m_1personnesLAB2_rep3'
test_dataset = RadarDataset(test_folder)

results_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew/inference_samples'
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
import pandas as pd
from sklearn.cluster import DBSCAN
import re
from modelPolarv1 import HeatmapPredictionModel
from PolarLoader import RadarDataset

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = HeatmapPredictionModel(num_heads=4, hidden_size=2048)
model.load_state_dict(torch.load('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_LAB2/modelpolar_stratifyv2_GTnew1_testlab2.pth'))
model.to(device)
model.eval()

# Function to parse ground truths from folder name
def parse_folder_name(folder_name):
    match = re.search(r'(\d+)_degres_([\d.]+)m', folder_name)
    if match:
        angle = float(match.group(1))
        distance = float(match.group(2))
        return distance, angle
    else:
        raise ValueError(f"Folder name {folder_name} does not contain the ground truth distance and angle.")

# Load test paths from the file
test_paths_file = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_LAB2/test_paths.txt'
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

        # Temporary storage for all predictions of current folder
        predicted_distances = []
        predicted_angles = []

        for sample_idx in range(len(test_dataset)):
            print(f"  Sample {sample_idx + 1}/{len(test_dataset)}")
            magnitude1, phase1, magnitude2, phase2, _ = test_dataset[sample_idx]
            magnitude1, phase1, magnitude2, phase2 = map(lambda x: x.unsqueeze(0).to(device), [magnitude1, phase1, magnitude2, phase2])

            with torch.no_grad():
                predicted_heatmap = model(magnitude1, phase1, magnitude2, phase2)
            pred_heatmap_np = predicted_heatmap.cpu().numpy()[0]

            # Define bins
            heatmap_size = (200, 200)
            range_bins = np.linspace(0, 10, heatmap_size[0])
            angle_bins = np.linspace(0, np.pi, heatmap_size[1])

            threshold = np.max(pred_heatmap_np) * 0.2
            significant_coords = np.array(np.nonzero(pred_heatmap_np > threshold)).T

            if len(significant_coords) > 0:
                dbscan = DBSCAN(eps=0.5, min_samples=1)
                labels = dbscan.fit_predict(significant_coords)

                clustered_peaks = []
                for label in set(labels):
                    cluster_coords = significant_coords[labels == label]
                    cluster_center_idx = cluster_coords.mean(axis=0).astype(int)
                    cluster_range = range_bins[cluster_center_idx[0]]
                    cluster_angle = np.rad2deg(angle_bins[cluster_center_idx[1]])
                    clustered_peaks.append((cluster_range, cluster_angle))

                median_distance = np.median([peak[0] for peak in clustered_peaks])
                median_angle = np.median([peak[1] for peak in clustered_peaks])

                predicted_distances.append(median_distance)
                predicted_angles.append(median_angle)

            else:
                print(f"    No significant cluster detected for sample {sample_idx + 1}.")

        # Average the predictions of this folder
        if predicted_distances and predicted_angles:
            avg_distance = np.mean(predicted_distances)
            avg_angle = np.mean(predicted_angles)
            distance_error = abs(avg_distance - gt_distance)
            angle_error = abs(avg_angle - gt_angle)
            localization_error = np.sqrt(distance_error**2 + np.deg2rad(angle_error)**2)

            results.append({
                "Folder": os.path.basename(test_folder),
                "GT Distance (m)": round(gt_distance, 3),
                "GT Angle (°)": round(gt_angle, 3),
                "Predicted Distance (m)": round(avg_distance, 3),
                "Predicted Angle (°)": round(avg_angle, 3),
                "Distance Error (m)": round(distance_error, 3),
                "Angle Error (°)": round(angle_error, 3),
                "Localization Error (m)": round(localization_error, 3)
            })

    except Exception as e:
        print(f"Error processing folder {test_folder}: {e}")
        continue

# Save results to a CSV file
output_csv_path = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_test_LAB2/predictions_summary_test_lab2.csv'
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)
print(f"\nResults saved to: {output_csv_path}")
