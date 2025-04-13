import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, maximum_position
from sklearn.cluster import DBSCAN
from PolarLoader import RadarDataset
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

# Chargement du dataset
test_folders = [
    '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files/stand_23_degres_2.5m_1personnes_rep3',
    '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files/stand_45_degres_1.5m_1personnesv2_rep3',
    '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files/stand_68_degres_1m_1personnesv3_rep2',
    '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files/stand_68_degres_3.5m_1personnesLAB2_rep2'
    # Ajoutez ici d'autres dossiers de test si nécessaire
]
results_dir = '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew_COPY/inference_samples'
os.makedirs(results_dir, exist_ok=True)

for folder in test_folders:
    print(f"\nTraitement du dossier: {folder}")
    # Création du dataset pour le dossier courant
    test_dataset = RadarDataset(folder)





    # --------------------------
    # Initialisation pour l'accumulation des heatmaps et positions
    # --------------------------
    heatmap_list = []   # Pour stocker chaque heatmap prédite (tableau 2D de dimension 200x200)
    pred_distances = [] # Pour stocker la (ou les) position(s) de cluster par échantillon (distance en m)
    pred_angles = []    # Pour stocker la (ou les) position(s) de cluster par échantillon (angle en °)

    # Paramètres
    heatmap_dims = (200, 200)
    range_bins = np.linspace(0, 10, heatmap_dims[0])
    angle_bins = np.linspace(0, np.pi, heatmap_dims[1])
    eps_distance = 0.5  # Paramètre pour DBSCAN (en mètres)

    # --------------------------
    # Boucle sur l'ensemble des échantillons
    # --------------------------
    num_samples = len(test_dataset)
    print(f"Nombre total d'échantillons à traiter : {num_samples}")

    for sample_idx in range(num_samples):
        print(f"\nTraitement de l'échantillon {sample_idx + 1}/{num_samples}...")
        
        # Récupération des données
        magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap = test_dataset[sample_idx]
        # Ajout d'une dimension batch et transfert sur le device
        magnitude1, phase1, magnitude2, phase2 = map(lambda x: x.unsqueeze(0).to(device),
                                                    [magnitude1, phase1, magnitude2, phase2])
        
        # Passage dans le modèle
        with torch.no_grad():
            predicted_heatmap = model(magnitude1, phase1, magnitude2, phase2)
        
        # Extraction de la heatmap en 2D (on supprime la dimension du canal)
        # Si la forme est (1, 1, 200, 200) on récupère le tableau 200x200
        pred_heatmap_np = predicted_heatmap.cpu().numpy()[0, 0, :, :]
        
        # Stockage de la heatmap dans la liste
        heatmap_list.append(pred_heatmap_np)
        
        # --- Traitement pour détecter les clusters sur cette heatmap ---
        # Seuil : on ne considère que les valeurs > 50% du maximum
        threshold = np.max(pred_heatmap_np) * 0.5
        significant_mask = pred_heatmap_np > threshold
        
        # Étiquetage des régions significatives
        labeled_array, num_features = scipy.ndimage.label(significant_mask)
        
        # Localisation des pics : pour chaque région, on calcule la position du maximum
        peak_coords = maximum_position(pred_heatmap_np, labeled_array, range(1, num_features + 1))
        # Conversion en positions (distance, angle) à l'aide des bins
        peak_positions = [(range_bins[c[0]], np.rad2deg(angle_bins[c[1]])) for c in peak_coords]
        
        if not peak_positions:
            print("Aucun pic détecté pour cet échantillon.")
            continue
        
        # Regroupement avec DBSCAN (on regroupe les pics proches)
        peak_positions_np = np.array(peak_positions)
        dbscan = DBSCAN(eps=eps_distance, min_samples=1)
        labels = dbscan.fit_predict(peak_positions_np)
        
        clustered_peaks = []
        for lab in set(labels):  # 'lab' évite de masquer la fonction intégrée label()
            cluster = peak_positions_np[labels == lab]
            avg_range = np.mean(cluster[:, 0])
            avg_angle = np.mean(cluster[:, 1])
            clustered_peaks.append((avg_range, avg_angle))
        
        # Pour cet échantillon, on ajoute toutes les positions de clusters détectées
        for avg_range, avg_angle in clustered_peaks:
            pred_distances.append(avg_range)
            pred_angles.append(avg_angle)
            print(f"Échantillon {sample_idx + 1}: Cluster détecté à {avg_range:.2f} m, {avg_angle:.2f}°")

    # --------------------------
    # Calcul des valeurs médianes globales
    # --------------------------
    if pred_distances:
        global_distance_median = np.median(pred_distances)
        global_angle_median = np.median(pred_angles)
        print("\n=== Résultats finaux ===")
        print(f"Médiane globale - Distance : {global_distance_median:.2f} m, Angle : {global_angle_median:.2f}°")
    else:
        print("Aucune prédiction consolidée pour les clusters.")

    # Calcul de la heatmap médiane (pixel par pixel) sur tous les échantillons traités
    if heatmap_list:
        median_heatmap = np.median(np.stack(heatmap_list, axis=0), axis=0)
    else:
        print("Aucune heatmap disponible.")
        exit()

    # --------------------------
    # Visualisation finale : une seule heatmap en coordonnées polaires
    # --------------------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)

    # Création de la grille de coordonnées pour l'affichage
    R, Theta = np.meshgrid(range_bins, angle_bins, indexing='ij')
    c = ax.pcolormesh(Theta, R, median_heatmap, cmap='jet', shading='auto')

    # Superposition du point correspondant à la position médiane globale des clusters
    ax.plot(np.deg2rad(global_angle_median), global_distance_median, 'ro', markersize=10,
            label=f"Médiane: {global_distance_median:.2f} m, {global_angle_median:.2f}°")
    ax.set_title(f"Heatmap Médiane\nDistance: {global_distance_median:.2f} m, Angle: {global_angle_median:.2f}°")
    plt.colorbar(c, ax=ax, label='Probabilité')
    plt.legend()

    # Sauvegarder et afficher la heatmap finale
    final_output = os.path.join(results_dir, "median_heatmap.png")
    plt.savefig(final_output)
    plt.show()
