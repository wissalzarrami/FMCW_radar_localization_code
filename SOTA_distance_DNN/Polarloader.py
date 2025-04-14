import os
import re
import torch
from torch.utils.data import Dataset
import numpy as np

class RadarDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir.rstrip('/')
        self.antenna1_files = []
        self.antenna2_files = []
        self.class_centers = torch.linspace(1.0, 5.0, 10)  # 10 classes (1.0m to 5.0m in 0.5m steps)
        self.ground_truth_class = None  # Now using class index instead of raw distance

        # Load data
        self.load_data()

    def load_data(self):
        print(f"Loading data from: {self.data_dir}")

        # Load antenna files
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('_a1.cf32'):
                self.antenna1_files.append(os.path.join(self.data_dir, file_name))
            elif file_name.endswith('_a2.cf32'):
                self.antenna2_files.append(os.path.join(self.data_dir, file_name))

        # Validate file counts
        if len(self.antenna1_files) != 9 or len(self.antenna2_files) != 9:
            raise ValueError("Could not find 9 files for each antenna in the directory")

        # Sort files for consistency
        self.antenna1_files.sort()
        self.antenna2_files.sort()

        print(f"Found antenna files:\n  Antenna1: {self.antenna1_files}\n  Antenna2: {self.antenna2_files}")
        
        # Extract ground truth from folder name
        try:
            folder_name = os.path.basename(self.data_dir)
            if not folder_name:
                raise ValueError("Empty folder name, check path.")
            
            print(f"Folder name: {folder_name}")
            distance, angle = self.parse_folder_name(folder_name)
            
            # Convert distance to class index
            self.ground_truth_class = torch.argmin(torch.abs(self.class_centers - distance)).item()
            print(f"Ground truth: {distance}m → Class {self.ground_truth_class} ({self.class_centers[self.ground_truth_class]}m)")
            
        except ValueError as e:
            print(f"Error parsing folder name: {folder_name}. {e}")
            self.ground_truth_class = None

    def parse_folder_name(self, folder_name):
        match = re.search(r'(\d+)_degres_([\d.]+)m', folder_name)
        if match:
            angle = float(match.group(1))
            distance = float(match.group(2))
            return distance, angle
        raise ValueError(f"Invalid folder format: {folder_name}")

    def __len__(self):
        return len(self.antenna1_files)

    def __getitem__(self, idx):
        if idx >= len(self.antenna1_files):
            raise IndexError("Index out of range")

        # Load and process antenna data
        antenna1_data = self.load_cf32(self.antenna1_files[idx])
        antenna2_data = self.load_cf32(self.antenna2_files[idx])

        # Extract magnitude and phase
        mag1, ph1 = self.fft_and_polar(antenna1_data)
        mag2, ph2 = self.fft_and_polar(antenna2_data)

        # Normalize
        mag1, ph1 = self.normalize(mag1, ph1)
        mag2, ph2 = self.normalize(mag2, ph2)

        # Convert to tensors
        mag1 = torch.tensor(mag1, dtype=torch.float32)
        ph1 = torch.tensor(ph1, dtype=torch.float32)
        mag2 = torch.tensor(mag2, dtype=torch.float32)
        ph2 = torch.tensor(ph2, dtype=torch.float32)

        if self.ground_truth_class is None:
            raise ValueError("Ground truth class not set")

        return mag1, ph1, mag2, ph2, torch.tensor(self.ground_truth_class, dtype=torch.long)

    def load_cf32(self, file_path):
        return np.fromfile(file_path, dtype=np.complex64)

    def fft_and_polar(self, complex_data):
        return np.abs(complex_data), np.angle(complex_data)

    def normalize(self, magnitude, phase):
        magnitude = (magnitude - magnitude.mean()) / magnitude.std()
        phase = (phase + np.pi) / (2 * np.pi)  # Normalize to [-1, 1]
        return magnitude, phase