import os
import numpy as np

class CustomLoader:
    def __init__(self, data_dir):
        """
        Initialise le data loader avec le répertoire contenant les fichiers.
        """
        self.data_dir = data_dir
        self.antenna1_files = []
        self.antenna2_files = []

    def load_data(self):
        """
        Parcourt le répertoire self.data_dir et répertorie les fichiers pour l'antenne 1 et l'antenne 2.
        Les fichiers doivent se terminer par '_a1.cf32' ou '_a2.cf32'.
        """
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

    def load_sample(self, index):
        """
        Pour l'index donné, charge les données associées aux deux antennes,
        calcule la matrice de corrélation Rx et en extrait le vecteur y 
        correspondant à la partie triangulaire supérieure (diag: réel, hors diag: réel et imaginaire).

        Hypothèses :
         - Chaque fichier contient un signal complexe (dtype=np.complex64)
         - Les deux fichiers associés contiennent T snapshots.
         - La matrice Rx est approximée par la moyenne sur T snapshots.

        Retourne :
            y : vecteur de dimension M2 (float32)
        """
        # Charger les données de l'antenne 1 et de l'antenne 2
        a1 = np.fromfile(self.antenna1_files[index], dtype=np.complex64)
        a2 = np.fromfile(self.antenna2_files[index], dtype=np.complex64)
        
        # Pour cet exemple, on suppose que chaque fichier contient T snapshots.
        # On empile les signaux pour former une matrice x de dimension (M, T)
        # Ici, M=2 (2 antennes), mais dans votre cas M peut être supérieur.
        x = np.vstack([a1, a2])
        
        # Nombre de snapshots (on suppose que les deux antennes fournissent le même nombre)
        T = x.shape[1]
        # Calcul de la matrice de corrélation Rx = E[x x^H] ≈ (1/T)*x*x^H
        Rx = (x @ x.conj().T) / T
        
        # Extraction du vecteur y à partir de la partie triangulaire supérieure de Rx.
        # Pour les éléments diagonaux, seule la partie réelle est utilisée.
        # Pour les éléments hors diagonaux, on stocke d'abord la partie réelle puis la partie imaginaire.
        M = Rx.shape[0]
        y = []
        for i in range(M):
            for j in range(i, M):
                if i == j:
                    # Diagonale : seule la partie réelle
                    y.append(np.real(Rx[i, j]))
                else:
                    y.append(np.real(Rx[i, j]))
                    y.append(np.imag(Rx[i, j]))
        y = np.array(y, dtype=np.float32)
        return y


"""
# Exemple d'utilisation :
if __name__ == "__main__":
    data_dir = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files/stand_23_degres_1.5m_1personnes_rep1"  # Remplacez par le chemin réel de vos données
    loader = CustomLoader(data_dir)
    loader.load_data()
    
    # Chargement du premier échantillon
    sample_vector = loader.load_sample(0)
    print("Vecteur d'entrée y extrait :", sample_vector)
"""