
import os
import re
import torch
from torch.utils.data import Dataset
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

class RadarDataset(Dataset):
    def __init__(self, data_dir, sigma_range=1, sigma_angle=15):
        self.data_dir = data_dir.rstrip('/')  # Remove any trailing slashes from the path
        self.antenna1_files = []
        self.antenna2_files = []
        self.ground_truth_heatmap = None
        
        self.sigma_range = sigma_range
        self.sigma_angle = sigma_angle
        
        # Load the data from the directory
        self.load_data()

    def load_data(self):
        print(f"Loading data from: {self.data_dir}")

        # Load all _a1 and _a2 files
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('_a1.cf32'):
                self.antenna1_files.append(os.path.join(self.data_dir, file_name))
            elif file_name.endswith('_a2.cf32'):
                self.antenna2_files.append(os.path.join(self.data_dir, file_name))

        # Check if we have exactly 9 files for each antenna
        if len(self.antenna1_files) != 9 or len(self.antenna2_files) != 9:
            raise ValueError("Could not find 9 files for each antenna in the directory")

        # Sort files to ensure they are in the correct order
        self.antenna1_files.sort()
        self.antenna2_files.sort()

        print(f"Found antenna files: {self.antenna1_files}, {self.antenna2_files}")
        
        # Extract ground truth heatmap based on the folder name
        try:
            folder_name = os.path.basename(self.data_dir)
            if not folder_name:
                raise ValueError("Folder name is empty, please verify the path.")

            print(f"Folder name: {folder_name}")  # Debug: display folder name
            
            ground_truth_distance, ground_truth_angle = self.parse_folder_name(folder_name)
            print(f"Extracted ground truth distance: {ground_truth_distance}, angle: {ground_truth_angle}")
            
            # Generate the polar Gaussian heatmap for multiple values
            self.ground_truth_heatmap = self.generate_continuous_polar_heatmap(
                range_vals=[ground_truth_distance],  # Single target scenario
                angle_vals=[ground_truth_angle],  # Single angle scenario
                sigma_range=self.sigma_range,
                sigma_angle=self.sigma_angle
            )
            
            if self.ground_truth_heatmap is None:
                raise ValueError(f"Failed to generate heatmap for folder: {folder_name}")
            print(f"Generated ground truth heatmap for distance: {ground_truth_distance}, angle: {ground_truth_angle}")
        except ValueError as e:
            print(f"Error parsing folder name: {folder_name}. {e}")
            self.ground_truth_heatmap = None

    def parse_folder_name(self, folder_name):
        # Handle both integer and float distances
        print(f"Parsing folder name: {folder_name}")  # Debug: display folder name

        # Adjust regex to accept both integer and floating-point distances
        match = re.search(r'(\d+)_degres_([\d.]+)m', folder_name)
        if match:
            angle = float(match.group(1))
            distance = float(match.group(2))  # Can be either integer or float
            return distance, angle
        else:
            raise ValueError(f"Folder name {folder_name} does not contain the ground truth distance and angle.")

    def generate_continuous_polar_heatmap(self, range_vals, angle_vals, sigma_range, sigma_angle):
        """
        Génère une heatmap polaire basée sur des cibles définies par leurs distances et angles,
        avec une propagation gaussienne ajustée.

        :param range_vals: Liste des distances (en mètres) pour les cibles.
        :param angle_vals: Liste des angles (en degrés) pour les cibles.
        :param sigma_range: Écart-type de la gaussienne en distance.
        :param sigma_angle: Écart-type de la gaussienne en angle (en degrés).
        :return: Heatmap polaire normalisée.
       
       """

        # Définir les limites de la heatmap
        r_min = 0
        r_max = 10  # Distance maximale en mètres
        theta_min = 0
        theta_max = np.pi  # Angle maximal en radians (180°)

        # Résolution de la heatmap
        num_range_points = 200
        num_angle_points = 200

        # Créer une grille polaire: rep polaire ! 
        r = np.linspace(r_min, r_max, num_range_points, dtype=np.float32)
        theta = np.linspace(theta_min, theta_max, num_angle_points, dtype=np.float32)
        R, Theta = np.meshgrid(r, theta, indexing='ij')

        # Convertir les angles des cibles en radians
        angle_vals_rad = np.deg2rad(angle_vals)
        sigma_angle_rad = np.deg2rad(sigma_angle)

        # Initialiser la heatmap
        heatmap = np.zeros_like(R)

        # Parcourir chaque couple (distance, angle) pour ajouter des gaussiennes
        for range_val, angle_val_rad in zip(range_vals, angle_vals_rad):
            # Calculer la gaussienne centrée sur (range_val, angle_val_rad)
            gaussian = np.exp(-((R - range_val) ** 2) / (2 * sigma_range ** 2)) * \
                    np.exp(-((Theta - angle_val_rad) ** 2) / (2 * sigma_angle_rad ** 2))

            # Ajouter la gaussienne à la heatmap
            heatmap += gaussian

        # Normaliser la heatmap pour qu'elle soit dans la plage [0, 1]
        heatmap /= np.max(heatmap)

        return heatmap

    

    def __len__(self):
        # The total number of elements in the dataset is based on the number of files
        return len(self.antenna1_files)

    def __getitem__(self, idx):
        if idx >= len(self.antenna1_files):
            raise IndexError("Index out of range for the dataset")

        # Load the corresponding files for both antennas at the given index
        antenna1_data = self.load_cf32(self.antenna1_files[idx])
        antenna2_data = self.load_cf32(self.antenna2_files[idx])

        # Apply FFT and convert complex data to magnitude and phase
        magnitude1, phase1 = self.polar(antenna1_data)
        magnitude2, phase2 = self.polar(antenna2_data)

        # Normalize the data
        magnitude1, phase1 = self.normalize(magnitude1, phase1)
        magnitude2, phase2 = self.normalize(magnitude2, phase2)

        # Ensure the heatmap is generated before converting to PyTorch tensor
        if self.ground_truth_heatmap is None:
            raise ValueError("Ground truth heatmap is not generated or is missing.")
        
        # Convert magnitudes, phases, and heatmap to PyTorch tensors
        magnitude1 = torch.tensor(magnitude1, dtype=torch.float32)
        phase1 = torch.tensor(phase1, dtype=torch.float32)
        magnitude2 = torch.tensor(magnitude2, dtype=torch.float32)
        phase2 = torch.tensor(phase2, dtype=torch.float32)
        ground_truth_heatmap = torch.tensor(self.ground_truth_heatmap, dtype=torch.float32)

        return magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap

    def load_cf32(self, file_path):
        return np.fromfile(file_path, dtype=np.complex64)

    def polar(self, complex_data):
        # Apply FFT if required
        # fft_data = fft.fft(complex_data)
        magnitude = np.abs(complex_data)
        phase = np.angle(complex_data)
        return magnitude, phase

    def normalize(self, magnitude, phase):
        magnitude = (magnitude - magnitude.mean()) / magnitude.std()
        phase = (phase + np.pi) / (2 * np.pi)  # Scale phase to range [0, 1]
        phase = 2 * phase - 1  # Scale phase to range [-1, 1]
        return magnitude, phase