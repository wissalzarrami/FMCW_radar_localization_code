import os
import re
import torch
from torch.utils.data import Dataset
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

class RadarDataset(Dataset):
    #def __init__(self, data_dir, sigma_range=1, sigma_angle=12):
    def __init__(self, data_dir, sigma_range=0.7, sigma_angle=7): #jdid
        self.data_dir = data_dir.rstrip('/')
        self.antenna1_files = []
        self.antenna2_files = []
        self.ground_truth_heatmap = None
        
        self.sigma_range = sigma_range
        self.sigma_angle = sigma_angle
        
        self.load_data()

    def load_data(self):
        print(f"Loading data from: {self.data_dir}")

        # Charger tous les fichiers _a1 et _a2
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('_a1.cf32'):
                self.antenna1_files.append(os.path.join(self.data_dir, file_name))
            elif file_name.endswith('_a2.cf32'):
                self.antenna2_files.append(os.path.join(self.data_dir, file_name))

        # Vérifier qu'on a bien 9 fichiers par antenne
        if len(self.antenna1_files) != 9 or len(self.antenna2_files) != 9:
            raise ValueError("Could not find 9 files for each antenna in the directory")

        # Trier pour l'ordre cohérent
        self.antenna1_files.sort()
        self.antenna2_files.sort()

        print(f"Found antenna files:\n  A1: {self.antenna1_files}\n  A2: {self.antenna2_files}")
        
        # Extraire la heatmap
        try:
            folder_name = os.path.basename(self.data_dir)
            if not folder_name:
                raise ValueError("Folder name is empty, please verify the path.")

            print(f"Folder name: {folder_name}")
            
            targets = self.parse_folder_name(folder_name)
            # targets ex: [(68.0, 3.0), (90.0, 2.0)] pour deux personnes
            print(f"Extracted ground truth targets: {targets}")

            angle_vals = [t[0] for t in targets]
            range_vals = [t[1] for t in targets]

            self.ground_truth_heatmap = self.generate_continuous_polar_heatmap(
                range_vals=range_vals,
                angle_vals=angle_vals,
                sigma_range=self.sigma_range,
                sigma_angle=self.sigma_angle
            )
            
            if self.ground_truth_heatmap is None:
                raise ValueError(f"Failed to generate heatmap for folder: {folder_name}")
            print(f"Generated ground truth heatmap for targets at {targets}")
        except ValueError as e:
            print(f"Error parsing folder name: {folder_name}. {e}")
            self.ground_truth_heatmap = None

    def parse_folder_name(self, folder_name):
        """
        Retourne une liste de tuples (angle, distance).
        Accepte : p1_68_degres_3m, p2_90_degres_2m, ...
        """
        matches = re.findall(r'(?:p\d_)?(\d+)_degres_([\d.]+)m', folder_name)
        if not matches:
            raise ValueError(
                f"Folder name '{folder_name}' does not contain any valid angle/distance pairs."
            )
        results = []
        for angle_str, dist_str in matches:
            angle = float(angle_str)
            dist = float(dist_str)
            results.append((angle, dist))
        return results

    def generate_continuous_polar_heatmap(self, range_vals, angle_vals, sigma_range, sigma_angle):
        r_min = 0
        r_max = 10
        theta_min = 0
        theta_max = np.pi  # 180°

        num_range_points = 200
        num_angle_points = 200

        r = np.linspace(r_min, r_max, num_range_points, dtype=np.float32)
        theta = np.linspace(theta_min, theta_max, num_angle_points, dtype=np.float32)
        R, Theta = np.meshgrid(r, theta, indexing='ij')

        angle_vals_rad = np.deg2rad(angle_vals)
        sigma_angle_rad = np.deg2rad(sigma_angle)

        heatmap = np.zeros_like(R)

        for range_val, angle_val_rad in zip(range_vals, angle_vals_rad):
            gaussian = np.exp(-((R - range_val)**2) / (2*sigma_range**2)) * \
                       np.exp(-((Theta - angle_val_rad)**2) / (2*sigma_angle_rad**2))
            heatmap += gaussian

        heatmap /= np.max(heatmap)
        return heatmap

    def __len__(self):
        return len(self.antenna1_files)

    def __getitem__(self, idx):
        if idx >= len(self.antenna1_files):
            raise IndexError("Index out of range for the dataset")

        antenna1_data = self.load_cf32(self.antenna1_files[idx])
        antenna2_data = self.load_cf32(self.antenna2_files[idx])

        magnitude1, phase1 = self.polar(antenna1_data)
        magnitude2, phase2 = self.polar(antenna2_data)

        magnitude1, phase1 = self.normalize(magnitude1, phase1)
        magnitude2, phase2 = self.normalize(magnitude2, phase2)

        if self.ground_truth_heatmap is None:
            raise ValueError("Ground truth heatmap is not generated or is missing.")
        
        magnitude1 = torch.tensor(magnitude1, dtype=torch.float32)
        phase1 = torch.tensor(phase1, dtype=torch.float32)
        magnitude2 = torch.tensor(magnitude2, dtype=torch.float32)
        phase2 = torch.tensor(phase2, dtype=torch.float32)
        ground_truth_heatmap = torch.tensor(self.ground_truth_heatmap, dtype=torch.float32)

        return magnitude1, phase1, magnitude2, phase2, ground_truth_heatmap

    def load_cf32(self, file_path):
        return np.fromfile(file_path, dtype=np.complex64)

    def polar(self, complex_data):
        magnitude = np.abs(complex_data)
        phase = np.angle(complex_data)
        return magnitude, phase

    def normalize(self, magnitude, phase):
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-9)
        phase = (phase + np.pi) / (2 * np.pi)  # [0, 1]
        phase = 2 * phase - 1                 # [-1, 1]
        return magnitude, phase
