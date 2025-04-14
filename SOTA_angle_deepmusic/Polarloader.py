import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.linalg as la

def steering_vector(angle, element_positions):
    """
    Calcule le vecteur de pilotage a(theta) pour un réseau 1D.
    """
    M = element_positions.size
    phase_shifts = np.exp(1j * 2 * np.pi * element_positions * np.sin(angle))
    return phase_shifts / np.sqrt(M)

def compute_music_spectrum(cov_mat, num_signals, element_positions, angle_grid):
    """
    Calcule le pseudo-spectre MUSIC classique.
    """
    eigvals, eigvecs = la.eig(cov_mat)
    # Tri décroissant selon la magnitude des valeurs propres
    idx_sorted = np.argsort(-np.abs(eigvals))
    eigvecs = eigvecs[:, idx_sorted]
    
    M = cov_mat.shape[0]
    # Le sous-espace de bruit est constitué des vecteurs propres d'indice num_signals à M
    Qn = eigvecs[:, num_signals:M]
    
    pspectrum = np.zeros_like(angle_grid, dtype=np.float32)
    for i, ang in enumerate(angle_grid):
        a_vec = steering_vector(ang, element_positions)
        denom = Qn.conj().T @ a_vec
        pspectrum[i] = 1.0 / np.linalg.norm(denom)
    
    return pspectrum

class RadarMUSICDataset(Dataset):
    """
    Dataset construit à partir des fichiers .cf32 pour deux antennes.
    Pour chaque antenne, on suppose que chaque fichier contient au moins 500 snapshots
    (observations simultanées des capteurs). On extrait les 500 premiers échantillons 
    de chaque fichier et on moyenne sur l'ensemble des fichiers pour obtenir une 
    série temporelle de T=500 snapshots par antenne.
    
    L'entrée du réseau est constituée des 3 canaux (réel, imaginaire, phase) de la matrice de covariance,
    et le label est le pseudo-spectre MUSIC calculé à partir de cette matrice.
    """
    def __init__(self, data_dir, num_signals=1, angle_grid=None, num_snapshots=500):
        """
        Args:
          data_dir (str): répertoire contenant les fichiers .cf32 pour chaque antenne.
          num_signals (int): nombre de signaux (sources) à considérer.
          angle_grid (np.ndarray): grille d'angles en radians (par défaut 180 points de -pi/2 à pi/2).
          num_snapshots (int): nombre de snapshots à utiliser pour le calcul de la covariance (T).
        """
        super().__init__()
        self.data_dir = data_dir.rstrip('/')
        self.num_signals = num_signals
        self.num_snapshots = num_snapshots
        
        # Grille d'angles par défaut
        if angle_grid is None:
            self.angle_grid = np.linspace(-np.pi/2, np.pi/2, 180)
        else:
            self.angle_grid = angle_grid
        
        # Positions des capteurs pour 2 antennes (espacement de 0.5 wavelengths)
        self.element_positions = np.array([0.0, 0.5], dtype=np.float32)
        
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
    
    def __len__(self):
        # Ici, on construit une covariance unique par dossier, donc la taille du dataset est 1.
        return 1
    
    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(f"RadarMUSICDataset ne contient qu'un seul échantillon. Demande idx={idx}.")
        
        T = self.num_snapshots  # T = 500
        
        # Pour chaque antenne, on lit les 500 premiers snapshots de tous les fichiers
        # et on effectue une moyenne pour obtenir une série temporelle représentative.
        Y1_collection = []
        for fpath in self.antenna1_files:
            raw_data = np.fromfile(fpath, dtype=np.complex64)
            if len(raw_data) < T:
                raise ValueError(f"Le fichier {fpath} n'a pas au moins {T} snapshots.")
            Y1_collection.append(raw_data[:T])
        Y1_collection = np.stack(Y1_collection, axis=0)
        Y1 = np.mean(Y1_collection, axis=0)
        
        Y2_collection = []
        for fpath in self.antenna2_files:
            raw_data = np.fromfile(fpath, dtype=np.complex64)
            if len(raw_data) < T:
                raise ValueError(f"Le fichier {fpath} n'a pas au moins {T} snapshots.")
            Y2_collection.append(raw_data[:T])
        Y2_collection = np.stack(Y2_collection, axis=0)
        Y2 = np.mean(Y2_collection, axis=0)
        
        # Constitution de Y de forme (T, M) avec M = 2 capteurs.
        Y = np.stack([Y1, Y2], axis=1)
        
        # Calcul de la matrice de covariance : R = (Y^H @ Y) / T
        R = (Y.conj().T @ Y) / T
        
        # Normalisation de la matrice de covariance par sa norme de Frobenius
        frob_norm = np.linalg.norm(R, 'fro')
        if frob_norm != 0:
            R = R / frob_norm
        
        # Préparation des canaux d'entrée : réel, imaginaire et phase de R
        R_real = np.real(R)
        R_imag = np.imag(R)
        R_angle = np.angle(R)
        X = np.stack([R_real, R_imag, R_angle], axis=0)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Calcul du pseudo-spectre MUSIC (label) à partir de R
        pspectrum = compute_music_spectrum(
            cov_mat=R,
            num_signals=self.num_signals,
            element_positions=self.element_positions,
            angle_grid=self.angle_grid
        )
        label_tensor = torch.tensor(pspectrum, dtype=torch.float32)
        
        return X_tensor, label_tensor


"""
# -------------------------------------------------------------------------------
# DEMO USAGE:
# Suppose our data folder has:
#   file1_a1.cf32, file2_a1.cf32, ..., file9_a1.cf32
#   file1_a2.cf32, file2_a2.cf32, ..., file9_a2.cf32
#
# Then we create the dataset, retrieve item 0, and see the shapes:
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    def load_file_paths(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    data_folder = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files"  
    all_paths = load_file_paths(data_folder)
    dataset = [RadarMUSICDataset(sub_dir) for sub_dir in all_paths]
   # print(dataset[0][0])

    
    print(f"Dataset length: {len(dataset)} sample(s).")
    X_tensor, label_tensor = dataset[0][0] 
    print("Covariance input shape:", X_tensor.shape)   # (3, 2, 2)
    print("MUSIC label shape:", label_tensor.shape)    # (180,) if angle_grid=180 points
    
    import matplotlib.pyplot as plt

    angle_grid = dataset[0].angle_grid  # shape (180,)
    plt.plot(angle_grid, label_tensor.numpy(), label="MUSIC pseudo-spectrum")
    plt.title("MUSIC Label from Covariance")
    plt.xlabel("Angle (radians)")
    plt.legend()

    # Save the figure to a file, e.g. "music_spectrum.png"
    plt.savefig("music_spectrum.png", dpi=300, bbox_inches="tight")

    # Optionally, show the figure in an interactive window
    plt.show()
"""