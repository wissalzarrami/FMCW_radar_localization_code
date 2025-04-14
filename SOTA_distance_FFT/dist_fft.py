import os
import re
import csv
import numpy as np

# -----------------------------
# Paramètres FMCW (à adapter au besoin)
# -----------------------------
c = 3e8           # Vitesse de la lumière (m/s)
B = 200e6         # Bande passante (Hz) : 200 MHz
fs = 320e3        # Fréquence d'échantillonnage (Hz) : 320 kHz
N_fast_time = 2048  # Nombre d'échantillons (fast time) par chirp


def extract_distance(folder_path):
    """
    Extrait la distance 'ground truth' à partir du nom du dossier.
    On cherche un nombre (entier ou flottant) avant le 'm' qui suit le terme 'degres'.
    Exemple : 'stand_68_degres_3m_1personnesLAB2_rep1' => on récupère 3.
    """
    match = re.search(r'degres[_-](\d+(?:\.\d*)?)m', folder_path, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        return None


def read_cf32(filename):
    """
    Lit un fichier binaire .cf32 (complex64).
    """
    return np.fromfile(filename, dtype=np.complex64)


def estimate_range_and_beatfreq(filename):
    """
    Lit un fichier .cf32, effectue la FFT en temps rapide (range),
    puis estime la fréquence de battement (peak) et en déduit une distance.
    
    Retourne un tuple : (distance_estimee, beat_frequency_estimee)
    """
    data = read_cf32(filename)
    
    # Vérifie la dimension
    if data.size % N_fast_time != 0:
        raise ValueError(f"Le fichier {filename} ne contient pas un multiple de {N_fast_time} échantillons.")

    # Nombre de 'chirps' (temps lent)
    N_slow_time = data.size // N_fast_time
    
    # On reshape sous forme (N_slow_time, N_fast_time)
    data_2d = data.reshape((N_slow_time, N_fast_time))
    
    # FFT sur fast time (range). Pas de fftshift pour cet exemple simplifié.
    # (Vous pouvez ajouter un fftshift si nécessaire.)
    fft_range = np.fft.fft(data_2d, axis=1) / N_fast_time
    
    # On construit un "range profile" en sommant (ou moyennant) l'amplitude sur le temps lent
    range_profile = np.sum(np.abs(fft_range), axis=0)
    
    # Indice du pic
    peak_idx = np.argmax(range_profile)
    
    # Fréquence de battement estimée
    # On suppose un axe fréquentiel variant de [0, fs), sans fftshift
    beat_freq = peak_idx * fs / N_fast_time
    
    # Hypothèse : la durée d'un chirp = (N_slow_time / fs) 
    # (i.e. on suppose que tout le fichier correspond à 1 seul chirp,
    #  ce qui est très simplifié. À adapter selon votre vrai T_sweep !)
    T_sweep = N_slow_time / fs
    
    # Relation distance pour FMCW : R = (c * T_sweep * beat_freq) / (2 * B)
    distance_estimee = (c * T_sweep * beat_freq) / (2 * B)
    
    return distance_estimee, beat_freq


def process_folder(folder_path):
    """
    Traite un dossier en :
      - Vérifiant la présence de 9 fichiers pour l'antenne 1 et 9 fichiers pour l'antenne 2.
      - Lisant chaque fichier pour construire un tableau de distances/beats.
      - Moyennant la distance et la beat frequency sur l'ensemble.
      - Comparant avec la ground truth extraite du nom du dossier.
    
    Retourne :
      (folder_path, average_range, average_beat_frequency, ground_truth, distance_error)
    ou None si le dossier n'a pas le nombre de fichiers attendu.
    """
    ground_truth = extract_distance(folder_path)
    
    # Liste des fichiers antenne 1
    antenna1_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith('_a1.cf32')
    ])
    # Liste des fichiers antenne 2
    antenna2_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith('_a2.cf32')
    ])
    
    # Vérification : 9 fichiers pour chaque antenne
    if len(antenna1_files) != 9 or len(antenna2_files) != 9:
        print(f"[SKIP] {folder_path} : on attend 9 fichiers par antenne, "
              f"mais on a {len(antenna1_files)} (a1) et {len(antenna2_files)} (a2).")
        return None
    
    distances = []
    beat_freqs = []
    
    # Traiter les 9 fichiers de l'antenne 1
    for f in antenna1_files:
        dist, bf = estimate_range_and_beatfreq(f)
        distances.append(dist)
        beat_freqs.append(bf)
    
    # Traiter les 9 fichiers de l'antenne 2
    for f in antenna2_files:
        dist, bf = estimate_range_and_beatfreq(f)
        distances.append(dist)
        beat_freqs.append(bf)
    
    # Moyenne sur les 18 fichiers
    average_range = np.mean(distances)
    average_beat_frequency = np.mean(beat_freqs)
    
    # Calcul de l'erreur (si la ground truth est trouvée)
    if ground_truth is not None:
        distance_error = abs(average_range - ground_truth)
    else:
        distance_error = None
    
    return (folder_path, average_range, ground_truth, distance_error)


def main():
    # Fichier texte contenant les chemins de dossiers, un par ligne
    #test_paths_file = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/SOTA2_deepmusic/test_paths_copy.txt"
    test_paths_file = "//store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew__2labs/test_paths_WORK_WITH_THIS.txt"
    with open(test_paths_file, 'r') as f:
        folder_paths = [line.strip() for line in f if line.strip()]
    
    results = []
    for folder in folder_paths:
        res = process_folder(folder)
        if res is not None:
            results.append(res)
    
    # Écriture des résultats dans un CSV
    csv_filename = 'results_LAST.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'folder', 
            'average_range (m)', 
            'ground_truth (m)', 
            'distance_error (m)'
        ])
        for row in results:
            csv_writer.writerow(row)
    
    print(f"Résultats écrits dans le fichier : {csv_filename}")


if __name__ == "__main__":
    main()
