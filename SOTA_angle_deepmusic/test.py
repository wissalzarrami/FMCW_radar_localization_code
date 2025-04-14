### voir juste une valeur dun angle
"""
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Import your partitioned model definition
from model import DeepMUSICPartitionedNet
# Import your dataset
from Polarloader import RadarMUSICDataset

if __name__ == "__main__":
    # -------------------------------
    # 1) Choix du dossier d'inférence
    # -------------------------------
    inference_folder = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files/stand_23_degres_1.5m_1personnesv2_rep2"

    # Hyperparamètres du modèle (doivent correspondre à l'entraînement)
    M = 2      # nombre de capteurs
    N = 180    # nombre total de bins d'angle
    Q = 8      # nombre de sous-régions (doit correspondre à l'entraînement)

    # -------------------------------
    # 2) Construction du dataset pour ce dossier (1 échantillon)
    # -------------------------------
    test_dataset = RadarMUSICDataset(
        data_dir=inference_folder,
        num_signals=1,
        angle_grid=None  # par défaut, 180 angles de -pi/2 à pi/2
    )

    # Récupérer l'échantillon unique : X_cov (entrée) et classical_label (label MUSIC classique)
    X_cov, classical_label = test_dataset[0]  # X_cov: (3, M, M), classical_label: (N,)

    # -------------------------------
    # 3) Chargement du modèle partitionné
    # -------------------------------
    model = DeepMUSICPartitionedNet(M=M, N=N, Q=Q)
    model_weights = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/SOTA_deepmusic/deepmusic_partitioned_model.pth"
    if not os.path.isfile(model_weights):
        raise FileNotFoundError(f"Fichier du modèle {model_weights} introuvable!")
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    # -------------------------------
    # 4) Passage forward pour obtenir le pseudo-spectre prédit
    # -------------------------------
    with torch.no_grad():
        X_cov_batch = X_cov.unsqueeze(0)   # forme: (1, 3, M, M)
        pred_spectrum = model(X_cov_batch)   # forme: (1, N)
        pred_spectrum = pred_spectrum[0]     # forme: (N,)

    # Conversion en NumPy
    pred_spectrum_np = pred_spectrum.cpu().numpy()  # devrait être de forme (180,)
    classical_label_np = classical_label.cpu().numpy()  # (180,)
    print("Pseudo-spectre prédit :", pred_spectrum_np)

    # -------------------------------
    # 5) Détection des pics (on souhaite K = 1 pic, comme demandé)
    # -------------------------------
    # Ici, on utilise une contrainte d'espacement plus importante avec distance=5 pour éviter les pics trop rapprochés.
    peaks, properties = find_peaks(pred_spectrum_np, distance=5, height=0.1)

    # Si plus d'un pic est trouvé, on ne conserve que le pic de plus forte amplitude
    K = 1
    if len(peaks) > K:
        peak_amplitudes = pred_spectrum_np[peaks]
        sorted_indices = np.argsort(peak_amplitudes)[-K:]
        peaks = peaks[sorted_indices]
        peaks = np.sort(peaks)  # Tri pour avoir les angles dans l'ordre croissant

    # -------------------------------
    # 6) Préparation de la grille d'angles
    # -------------------------------
    angle_grid = test_dataset.angle_grid   # (180,) en radians
    angle_grid_deg = np.degrees(angle_grid)  # Conversion en degrés

    # Si besoin, interpolation pour que pred_spectrum_np ait la même longueur que angle_grid_deg
    if len(pred_spectrum_np) != len(angle_grid_deg):
        interp_func = interp1d(np.linspace(0, 1, len(pred_spectrum_np)),
                               pred_spectrum_np,
                               kind='linear',
                               fill_value='extrapolate')
        pred_spectrum_np = interp_func(np.linspace(0, 1, len(angle_grid_deg)))

    est_angles_deg = angle_grid_deg[peaks]
    print("Pic estimé (DOA en degrés) :", est_angles_deg)

    # -------------------------------
    # 7) Affichage du spectre prédit et du pseudo-spectre classique
    # -------------------------------
    plt.figure(figsize=(8,6))
    plt.plot(angle_grid_deg, pred_spectrum_np, label="Spectre prédit par DeepMUSIC", color="blue")
    plt.plot(angle_grid_deg, classical_label_np, '--', label="Pseudo-spectre MUSIC classique", color="green")
    plt.plot(angle_grid_deg[peaks], pred_spectrum_np[peaks], 'ro', markersize=8, label="Pic estimé")

    # Annoter le pic avec sa valeur (en degrés)
    for pk in peaks:
        doa_deg = angle_grid_deg[pk]
        val = pred_spectrum_np[pk]
        plt.text(doa_deg, val , f"{doa_deg:.1f}°", color='red', ha='center', fontsize=12)

    plt.xlabel("Angle (°)")
    plt.ylabel("Amplitude du pseudo-spectre")
    plt.title(f"Spectre prédit & pseudo-spectre classique \nDossier: {os.path.basename(inference_folder)}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("inference_peaks_only.png", dpi=300)
    plt.show()

"""
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import csv

# Import your partitioned model definition
from model import DeepMUSICPartitionedNet
# Import your dataset
from Polarloader import RadarMUSICDataset

def load_paths_from_file(file_path):
    
    #Reads paths from a file and returns them as a list.
    
    with open(file_path, 'r') as file:
        paths = file.readlines()
    return [path.strip() for path in paths]  # remove any leading/trailing whitespaces

def extract_angle_from_folder_name(folder_name):
    
    #Extracts the angle from the folder name, assuming the format is like 'stand_90_degres_2.5m'.
    
    match = re.search(r'stand_(\d+)_degres', folder_name)
    if match:
        return int(match.group(1))  # Return the extracted angle as an integer
    else:
        raise ValueError(f"Angle not found in folder name: {folder_name}")

def write_results_to_csv(results, csv_filename="inference_results_LAST.csv"):
    
   # Write the results to a CSV file.
    
    header = ["Folder", "Estimated DOA", "Ground Truth DOA", "Error (degrees)"]
    
    # Write the header and results
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    # -------------------------------
    # 1) Load paths from the provided file
    # -------------------------------
    #paths_file = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew__2labs/test_paths.txt"
    paths_file = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew__2labs/test_paths_WORK_WITH_THIS.txt"
    inference_folders = load_paths_from_file(paths_file)
    print(f"Found {len(inference_folders)} folders to test.")

    # Model hyperparams (must match training)
    M = 2      # number of sensors
    N = 180    # total angle bins
    Q = 8      # number of subregions (must match how you trained)

    # -------------------------------
    # 2) Load your partitioned model
    # -------------------------------
    model = DeepMUSICPartitionedNet(M=M, N=N, Q=Q)
    model_weights = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/SOTA2_deepmusic/deepmusic_partitioned_model.pth"  # your trained checkpoint
    if not os.path.isfile(model_weights):
        raise FileNotFoundError(f"Model file {model_weights} not found!")
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    # List to store results for CSV
    results = []

    # Loop through each folder and perform inference
    for inference_folder in inference_folders:
        print(f"Processing folder: {inference_folder}")

        # -------------------------------
        # 3) Build a dataset for that folder (just 1 sample)
        # -------------------------------
        test_dataset = RadarMUSICDataset(
            data_dir=inference_folder,
            num_signals=1,
            angle_grid=None  # => default 180 angles from -pi/2..+pi/2
        )

        # We have only 1 item: X_cov, classical_label
        X_cov, classical_label = test_dataset[0]  # (3,M,M), (N,)

        # -------------------------------
        # 4) Forward pass to get predicted pseudo-spectrum
        # -------------------------------
        with torch.no_grad():
            X_cov_batch = X_cov.unsqueeze(0)   # shape (1,3,M,M)
            pred_spectrum = model(X_cov_batch) # shape (1,N)
            pred_spectrum = pred_spectrum[0]   # shape (N,)

        # Convert to NumPy for analysis
        pred_spectrum_np = pred_spectrum.cpu().numpy()  # shape (180,)

        # The classical MUSIC label from dataset
        classical_label_np = classical_label.cpu().numpy()  # shape (180,)

        # -------------------------------
        # 5) Peak finding => DOA estimate
        # -------------------------------
        peaks, _ = find_peaks(pred_spectrum_np, distance=2, height=0.1)

        # Retrieve angle grid from the dataset
        angle_grid = test_dataset.angle_grid  # shape (180,) in radians
        angle_grid_deg = np.degrees(angle_grid)

        # Interpolate pred_spectrum_np to match the length of angle_grid_deg
        if len(pred_spectrum_np) != len(angle_grid_deg):
            interp_func = interp1d(np.linspace(0, 1, len(pred_spectrum_np)), pred_spectrum_np, kind='linear', fill_value='extrapolate')
            pred_spectrum_np = interp_func(np.linspace(0, 1, len(angle_grid_deg)))

        est_angles_deg = angle_grid_deg[peaks]
        print("Estimated DOAs (degrees):", est_angles_deg)

        # -------------------------------
        # 6) Plot predicted vs. classical label
        # -------------------------------
        plt.figure(figsize=(8,6))
        plt.plot(angle_grid_deg, pred_spectrum_np, label="DeepMUSIC Predicted Spectrum")
        plt.plot(angle_grid_deg, classical_label_np, '--', label="Classical MUSIC Label")
        plt.plot(angle_grid_deg[peaks], pred_spectrum_np[peaks], 'ro', label="Estimated DOAs")

        # Annotate each detected DOA with its angle in degrees
        for pk in peaks:
            doa_deg = angle_grid_deg[pk]
            val = pred_spectrum_np[pk]
            plt.text(doa_deg, val + 0.02, f"{doa_deg:.1f}°", color='red', ha='center')

        plt.xlabel("Angle [degrees]")
        plt.ylabel("Pseudo-spectrum amplitude")
        plt.title(f"Inference on folder:\n{os.path.basename(inference_folder)}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"inference_partitioned_result_{os.path.basename(inference_folder)}.png", dpi=300)
        plt.show()

        # -------------------------------
        # 7) Extract ground truth angle from folder name
        # -------------------------------
        try:
            ground_truth_doa = extract_angle_from_folder_name(inference_folder)
        except ValueError as e:
            print(e)
            continue  # Skip this folder if the angle extraction fails

        # -------------------------------
        # 8) Calculate error and store results for CSV
        # -------------------------------
        doa_estimates = angle_grid_deg[peaks]  # Convert peak indices to degrees
        amplitudes = pred_spectrum_np[peaks]  # Get corresponding amplitudes

        # Calculate error (absolute difference between estimated DOAs and ground truth)
        errors = np.abs(doa_estimates - ground_truth_doa)

        # Assuming best DOA is the one with highest amplitude
        best_doa = doa_estimates[np.argmax(amplitudes)]
        error = np.min(errors)  # Minimum error between estimated and ground truth DOA

        # Add result to the CSV data
        results.append([inference_folder, best_doa, ground_truth_doa, error])

    # -------------------------------
    # 9) Write results to CSV file
    # -------------------------------
    write_results_to_csv(results)
    print(f"Results written to 'inference_results_LAST.csv'")
