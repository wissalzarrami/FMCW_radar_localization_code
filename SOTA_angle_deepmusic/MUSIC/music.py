import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

class DOAEstimator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.antenna1_files = []
        self.antenna2_files = []
        self.load_data()
    
    def load_data(self):
        print(f"Chargement des données depuis : {self.data_dir}")
        for file_name in os.listdir(self.data_dir):
            full_path = os.path.join(self.data_dir, file_name)
            if file_name.endswith('_a1.cf32'):
                self.antenna1_files.append(full_path)
            elif file_name.endswith('_a2.cf32'):
                self.antenna2_files.append(full_path)
        
        self.antenna1_files.sort()
        self.antenna2_files.sort()
        
        if not self.antenna1_files:
            raise ValueError("Aucun fichier '_a1.cf32' trouvé dans le dossier.")
        if not self.antenna2_files:
            raise ValueError("Aucun fichier '_a2.cf32' trouvé dans le dossier.")
        if len(self.antenna1_files) != len(self.antenna2_files):
            raise ValueError(
                f"Nombre de fichiers différent entre antenne1 ({len(self.antenna1_files)}) "
                f"et antenne2 ({len(self.antenna2_files)})."
            )
        
        print("Fichiers pour antenne 1 :")
        for f in self.antenna1_files:
            print(f"  {f}")
        print("Fichiers pour antenne 2 :")
        for f in self.antenna2_files:
            print(f"  {f}")
    
    def load_complex_data(self, file_path):
        # Suppose the files are stored in binary format with complex64 numbers.
        return np.fromfile(file_path, dtype=np.complex64)
    
    def run_music(self, save_fig_path=None):
        # Use only the first pair of files for simplicity.
        data_a1 = self.load_complex_data(self.antenna1_files[0])
        data_a2 = self.load_complex_data(self.antenna2_files[0])
        
        # Form a 2 x N data matrix (2 antennas, N snapshots)
        X = np.vstack([data_a1, data_a2])
        snapshots = X.shape[1]
        
        # Compute the covariance matrix (complex covariance includes both real and imaginary parts)
        R = np.dot(X, X.conj().T) / snapshots
        
        # Eigen-decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(R)
        idx = eigenvalues.argsort()[::-1]  # sort in descending order
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Assuming one signal, the noise subspace is the eigenvector corresponding to the smallest eigenvalue.
        noise_subspace = eigenvectors[:, 1:]
        
        # Define the scanning grid for angles (in degrees)
        angles = np.linspace(-90, 90, 181)
        P_music = []
        
        # Array parameters (assuming half-wavelength spacing)
        d = 0.5          # Sensor spacing in wavelengths
        wavelength = 1.0 # Normalized wavelength
        k = 2 * np.pi / wavelength
        
        # Compute the MUSIC pseudospectrum over the grid
        for theta in angles:
            theta_rad = np.deg2rad(theta)
            # Steering vector for a two-element array
            steering_vector = np.array([
                1,
                np.exp(-1j * k * d * np.sin(theta_rad))
            ]).reshape(-1, 1)
            # Projection onto the noise subspace (norm squared)
            denom = np.linalg.norm(np.dot(noise_subspace.conj().T, steering_vector))**2
            P_music.append(1.0 / denom if denom != 0 else 0)
        
        P_music = np.array(P_music)
        
        # Plot the MUSIC pseudospectrum
        plt.figure(figsize=(8, 4))
        plt.plot(angles, 10 * np.log10(P_music))
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Spatial Spectrum (dB)")
        plt.title("MUSIC DOA Estimation")
        plt.grid(True)
        if save_fig_path:
            plt.savefig(save_fig_path)
            plt.close()
        else:
            plt.show()
        
        # Identify the DOA as the angle with the maximum pseudospectrum value.
        max_idx = np.argmax(P_music)
        estimated_doa = angles[max_idx]
        print("Estimated DOA:", estimated_doa, "degrees")
        return estimated_doa

def extract_ground_truth(folder_path):
    """
    Extract the ground truth DOA from the folder name.
    Looks for a number preceding the term 'degres' (case insensitive).
    For example, if folder contains "stand_23_degres" it extracts 23.
    """
    match = re.search(r'(\d+)[ _-]*degres', folder_path, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        return None

def process_folders(test_paths_file, output_csv, output_fig_dir):
    # Read folder paths from the test paths file.
    with open(test_paths_file, 'r') as f:
        folder_paths = [line.strip() for line in f if line.strip()]
    
    results = []
    os.makedirs(output_fig_dir, exist_ok=True)
    
    for folder in folder_paths:
        print(f"\nProcessing folder: {folder}")
        try:
            estimator = DOAEstimator(folder)
            # Create a filename for the saved spectrum image using the folder name.
            folder_name = os.path.basename(folder.rstrip(os.sep))
            fig_filename = os.path.join(output_fig_dir, f"{folder_name}_spectrum.png")
            estimated_doa = estimator.run_music(save_fig_path=fig_filename)
            
            # Extract ground truth DOA from the folder name.
            ground_truth = extract_ground_truth(folder)
            if ground_truth is None:
                print("Ground truth DOA not found in folder name.")
                ground_truth = float('nan')
            error = abs(estimated_doa - ground_truth) if not np.isnan(ground_truth) else float('nan')
            
            results.append([folder, estimated_doa, ground_truth, error])
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            results.append([folder, float('nan'), float('nan'), float('nan')])
    
    # Write results to a CSV file.
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Folder", "Estimated DOA", "Ground Truth DOA", "Error (degrees)"])
        writer.writerows(results)
    print(f"\nResults saved to {output_csv}")

if __name__ == '__main__':
    # Path to the file containing folder paths (one folder per line)
    test_paths_file = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew__2labs/test_paths_WORK_WITH_THIS.txt"
    # Output CSV file to store results
    output_csv = "doa_results_LAST.csv"
    # Directory where the spectrum images will be saved
    output_fig_dir = "spectrum_figures"
    
    process_folders(test_paths_file, output_csv, output_fig_dir)
