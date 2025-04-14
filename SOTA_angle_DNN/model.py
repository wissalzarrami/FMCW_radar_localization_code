import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionNetwork(nn.Module):
    """
    Réseau de détection.
    Entrée : vecteur y de dimension M².
    Architecture :
      - Couche cachée : 50 neurones, activation ReLU.
      - Couche de sortie : Q neurones (linéaire) qui activent le secteur angulaire.
    """
    def __init__(self, input_dim, Q):
        super(DetectionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, Q)  # Q : nombre de secteurs angulaires

    def forward(self, x):
        x = F.relu(self.fc1(x))
        t = self.fc2(x)
        return t

class EstimationNetwork(nn.Module):
    """
    Réseau d'estimation DOA.
    Entrée : vecteur de dimension Q (sortie du module de détection).
    Architecture :
      - 1ère couche cachée : 50 neurones (ReLU).
      - 2ème couche cachée : 30 neurones (ReLU).
      - Couche de sortie : Z neurones avec softmax,
        où Z = (θ_max - θ_min)/Δθ + 1 (résolution angulaire).
    """
    def __init__(self, Q, hidden_dims, output_dim):
        super(EstimationNetwork, self).__init__()
        # hidden_dims est supposé être [50, 30]
        self.fc1 = nn.Linear(Q, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, t):
        x = F.relu(self.fc1(t))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # softmax pour obtenir une distribution de probabilités sur les angles
        oL = F.softmax(x, dim=-1)
        return oL

class FullDOANetwork(nn.Module):
    """
    Réseau complet combinant détection et estimation DOA.
    - Entrée : vecteur y (dimension M²)
    - Module de détection : sort une représentation t de dimension Q.
    - Module d'estimation : prend t et renvoie un vecteur de probabilités sur Z angles.
    """
    def __init__(self, input_dim, Q, hidden_dims, output_dim):
        super(FullDOANetwork, self).__init__()
        self.detection = DetectionNetwork(input_dim, Q)
        self.estimation = EstimationNetwork(Q, hidden_dims, output_dim)
    
    def forward(self, y):
        t = self.detection(y)
        oL = self.estimation(t)
        return oL
