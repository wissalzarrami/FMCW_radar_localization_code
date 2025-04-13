"""
import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################
# (Les classes FusionBlock1D, MagConvBlock1D et PCFEncoder1D restent identiques)
##########################################

class FusionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
      #  print("FusionBlock1D - after conv:", x.shape)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
     #   print("FusionBlock1D - after pool:", x.shape)
        return x

class MagConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MagConvBlock1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
       # print("MagConvBlock1D - after conv:", x.shape)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
       # print("MagConvBlock1D - after pool:", x.shape)
        return x

class PCFEncoder1D(nn.Module):
    
    #PCF-Encoder pour vecteurs 1D (traitement d'une antenne) :
     # - Convolution indépendante sur magnitude et phase,
      #- Concaténation initiale,
      #- Plusieurs blocs de fusion (fusion branch + mag-only branch).
    
    def __init__(self, in_channels=1, base_channels=32, num_blocks=3):
        super(PCFEncoder1D, self).__init__()
        # Convolutions initiales pour magnitude et phase
        self.mag_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.phase_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.initial_activation = nn.LeakyReLU(0.2)
        
        # Construction des blocs de fusion
        self.fusion_blocks = nn.ModuleList()
        self.mag_blocks = nn.ModuleList()
        
        # Après concaténation, la sortie initiale possède 2*base_channels canaux
        current_fused_channels = 2 * base_channels  # ex: 64
        current_mag_channels = base_channels         # ex: 32
        
        for i in range(num_blocks):
            target_channels = current_fused_channels
            self.fusion_blocks.append(
                FusionBlock1D(in_channels=current_fused_channels, out_channels=target_channels)
            )
            self.mag_blocks.append(
                MagConvBlock1D(in_channels=current_mag_channels, out_channels=target_channels)
            )
            # Mise à jour pour le bloc suivant (augmentation par concaténation)
            current_fused_channels = target_channels * 2
            current_mag_channels = target_channels

    def forward(self, mag, phase):
        # Assurer que les entrées ont la forme [B, 1, L]
        if mag.dim() == 2:
            mag = mag.unsqueeze(1)
        if phase.dim() == 2:
            phase = phase.unsqueeze(1)
        
        x_mag = self.initial_activation(self.mag_conv(mag))
       # print("PCFEncoder1D - x_mag after conv & activation:", x_mag.shape)
        x_phase = self.initial_activation(self.phase_conv(phase))
      #  print("PCFEncoder1D - x_phase after conv & activation:", x_phase.shape)
        
        fused = torch.cat([x_mag, x_phase], dim=1)
       # print("PCFEncoder1D - after initial concatenation (fused):", fused.shape)
        mag_feature = x_mag
        
        for idx, (fusion_block, mag_block) in enumerate(zip(self.fusion_blocks, self.mag_blocks)):
            fused_out = fusion_block(fused)
           # print(f"PCFEncoder1D - after fusion block {idx} (fused_out):", fused_out.shape)
            mag_out = mag_block(mag_feature)
           # print(f"PCFEncoder1D - after mag block {idx} (mag_out):", mag_out.shape)
            fused = torch.cat([fused_out, mag_out], dim=1)
           # print(f"PCFEncoder1D - after concatenation block {idx} (fused):", fused.shape)
            mag_feature = mag_out
      #  print("PCFEncoder1D - final output shape:", fused.shape)
        return fused

##########################################
# Branche complète : PCF-Encoder + Transformer
##########################################

class AntennaBranch(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_blocks=3,
                 transformer_d_model=512, nhead=4, input_length=262144, target_seq_length=256):
        
       # La branche traite les données d'une antenne.
        #  - PCFEncoder1D : extrait des features [B, transformer_d_model, L_reduit]
         # - Pooling additionnel pour réduire la séquence
          #- Transformer Encoder : traite la séquence réduite
          #- Pooling global sur la dimension temporelle pour obtenir un vecteur de dimension transformer_d_model
        
        super(AntennaBranch, self).__init__()
        self.encoder = PCFEncoder1D(in_channels, base_channels, num_blocks)
        # Calcul de la longueur réduite après les blocs de pooling du PCFEncoder1D
        self.seq_length = input_length // (2 ** num_blocks)
       # print("Initial sequence length after PCFEncoder1D pooling:", self.seq_length)
        # Pooling additionnel pour réduire la séquence avant Transformer
        self.additional_pool = nn.AdaptiveAvgPool1d(target_seq_length)
        # Le Transformer attend une entrée de forme [seq_length, B, d_model]
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=nhead)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, mag, phase):
        x = self.encoder(mag, phase)  # [B, transformer_d_model, L_reduit] 
       # print("AntennaBranch - after PCFEncoder1D:", x.shape)
        # Réduction supplémentaire de la longueur de séquence
        x = self.additional_pool(x)
       # print("AntennaBranch - after additional pooling:", x.shape)
        # Préparation pour le Transformer : [L_reduit_reduit, B, transformer_d_model]
        x = x.permute(2, 0, 1)
       # print("AntennaBranch - after permutation for Transformer:", x.shape)
        x = self.transformer_encoder(x)
       # print("AntennaBranch - after Transformer Encoder:", x.shape)
        # Retour à la forme [B, transformer_d_model, L_reduit_reduit]
        x = x.permute(1, 2, 0)
       # print("AntennaBranch - after re-permutation:", x.shape)
        # Pooling global sur la dimension temporelle pour obtenir un vecteur [B, transformer_d_model]
        x = self.pool(x)
       # print("AntennaBranch - after global pooling:", x.shape)
        x = x.squeeze(-1)
       # print("AntennaBranch - final feature vector:", x.shape)
        return x

##########################################
# Modèle complet Siamese pour 2 antennes
##########################################

class DualAntennaSiameseModel(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 base_channels=32, 
                 num_blocks=3, 
                 transformer_d_model=512, 
                 nhead=4, 
                 heatmap_size=200,
                 input_length=262144, 
                 target_seq_length=256):
        
        #Le modèle complet applique la même branche (siamese) aux deux antennes,
        #concatène les vecteurs de features, et prédit la heatmap 2D.
        
        super(DualAntennaSiameseModel, self).__init__()
        self.branch = AntennaBranch(in_channels, base_channels, num_blocks, 
                                    transformer_d_model, nhead, input_length, target_seq_length)
        # Après concaténation, la dimension est 2 * transformer_d_model (ex: 1024)
        self.fc = nn.Linear(2 * transformer_d_model, heatmap_size * heatmap_size)
        self.sigmoid = nn.Sigmoid()
        self.heatmap_size = heatmap_size
    
    def forward(self, mag1, phase1, mag2, phase2):
       # print("DualAntennaSiameseModel - processing antenna 1")
        feat1 = self.branch(mag1, phase1)  # [B, transformer_d_model]
       # print("DualAntennaSiameseModel - feature vector antenna 1:", feat1.shape)
       # print("DualAntennaSiameseModel - processing antenna 2")
        feat2 = self.branch(mag2, phase2)  # [B, transformer_d_model]
       # print("DualAntennaSiameseModel - feature vector antenna 2:", feat2.shape)
        combined = torch.cat([feat1, feat2], dim=1)
       # print("DualAntennaSiameseModel - after concatenation:", combined.shape)
        out = self.fc(combined)
       ## print("DualAntennaSiameseModel - after fully connected layer:", out.shape)
        out = self.sigmoid(out)
       # print("DualAntennaSiameseModel - after sigmoid:", out.shape)
        heatmap = out.view(-1, 1, self.heatmap_size, self.heatmap_size)
       # print("DualAntennaSiameseModel - final heatmap shape:", heatmap.shape)
        return heatmap

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################
# 1. Encodeur simple PCF avec un seul bloc de fusion
##############################################
class SimplePCFEncoder1D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        """
        Cette partie réalise :
          - Une première convolution sur la magnitude et la phase séparément.
          - Une fusion des deux branches via concaténation suivie d'une convolution.
          - Une convolution sur la branche magnitude seule.
          - Une fusion finale (concaténation des deux sorties) puis un dernier traitement.
          
        La sortie finale aura 2 * base_channels canaux.
        """
        super(SimplePCFEncoder1D, self).__init__()
        
        # Convolution initiale sur la magnitude
        self.conv_mag = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2)
        )
        # Convolution initiale sur la phase
        self.conv_phase = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2)
        )
        
        # Bloc de fusion : on concatène magnitude et phase pour former 2*base_channels canaux,
        # puis on applique une convolution, une batchnorm et un LeakyReLU.
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(2 * base_channels, 2 * base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(2 * base_channels),
            nn.LeakyReLU(0.2)
        )
        
        # Convolution sur la branche magnitude seule (pour extraire des features complémentaires)
        self.mag_conv2 = nn.Sequential(
            nn.Conv1d(base_channels, 2 * base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(2 * base_channels),
            nn.LeakyReLU(0.2)
        )
        
        # Fusion finale des deux branches (fusion et magnitude)
        # L'entrée a 4 * base_channels canaux (2*base_channels de fusion + 2*base_channels de magnitude)
        # La sortie sera ramenée à 2 * base_channels canaux.
        self.final_conv = nn.Sequential(
            nn.Conv1d(4 * base_channels, 2 * base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(2 * base_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, mag, phase):
        # S'assurer que les tenseurs ont la forme [B, 1, L]
        if mag.dim() == 2:
            mag = mag.unsqueeze(1)
        if phase.dim() == 2:
            phase = phase.unsqueeze(1)
        
        # Première convolution sur chaque branche
        x_mag = self.conv_mag(mag)       # [B, base_channels, L]
        x_phase = self.conv_phase(phase)   # [B, base_channels, L]
        
        # Fusion : concaténation des deux branches et passage par le bloc de fusion
        fusion_in = torch.cat([x_mag, x_phase], dim=1)  # [B, 2*base_channels, L]
        fusion_out = self.fusion_conv(fusion_in)          # [B, 2*base_channels, L]
        
        # Traitement de la branche magnitude seule
        mag_out = self.mag_conv2(x_mag)  # [B, 2*base_channels, L]
        
        # Concaténation finale des deux sorties
        combined = torch.cat([fusion_out, mag_out], dim=1)  # [B, 4*base_channels, L]
        out = self.final_conv(combined)                     # [B, 2*base_channels, L]
        
        return out

##############################################
# 2. Branche complète : Encodeur + Transformer
##############################################
class AntennaBranch(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, input_length=262144, target_seq_length=256,
                 transformer_d_model=64, nhead=4):
        """
        Cette branche traite les données d'une antenne :
          - Elle utilise l'encodeur PCF simplifié.
          - Un pooling (AdaptiveAvgPool1d) réduit la longueur de séquence à target_seq_length.
          - Un Transformer Encoder (simple, 1 couche) traite la séquence.
          - Un pooling global sur la dimension temporelle permet d'obtenir un vecteur de features.
        
        Remarque : transformer_d_model doit être égal à 2 * base_channels (la sortie de l'encodeur).
        """
        super(AntennaBranch, self).__init__()
        self.encoder = SimplePCFEncoder1D(in_channels, base_channels)
        
        # Réduction de la séquence à target_seq_length
        self.additional_pool = nn.AdaptiveAvgPool1d(target_seq_length)
        
        # Transformer Encoder simple
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=nhead)
        
        # Pooling global sur la dimension temporelle
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Vérification de la cohérence des dimensions
        assert transformer_d_model == 2 * base_channels, "transformer_d_model doit être égal à 2*base_channels"
        
    def forward(self, mag, phase):
        # Passage dans l'encodeur
        x = self.encoder(mag, phase)  # [B, 2*base_channels, L]
        
        # Réduction de la longueur de la séquence
        x = self.additional_pool(x)   # [B, 2*base_channels, target_seq_length]
        
        # Préparation pour le Transformer :
        # Le Transformer attend une entrée de forme [seq_length, B, d_model]
        x = x.permute(2, 0, 1)        # [target_seq_length, B, 2*base_channels]
        x = self.transformer_encoder(x)  # [target_seq_length, B, 2*base_channels]
        
        # Retour à la forme [B, 2*base_channels, target_seq_length]
        x = x.permute(1, 2, 0)
        
        # Pooling global sur la dimension temporelle pour obtenir un vecteur [B, 2*base_channels]
        x = self.pool(x)  # [B, 2*base_channels, 1]
        x = x.squeeze(-1) # [B, 2*base_channels]
        return x

##############################################
# 3. Modèle complet Siamese pour 2 antennes
##############################################
class DualAntennaSiameseModel(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, transformer_d_model=64, nhead=4,
                 heatmap_size=200, input_length=262144, target_seq_length=256):
        """
        Ce modèle applique la même branche (siamese) aux deux antennes, concatène les vecteurs de features
        et prédit une heatmap 2D.
        """
        super(DualAntennaSiameseModel, self).__init__()
        self.branch = AntennaBranch(in_channels, base_channels, input_length, target_seq_length,
                                    transformer_d_model, nhead)
        # Après concaténation des deux vecteurs (de dimension transformer_d_model chacun),
        # la dimension totale est 2 * transformer_d_model.
        self.fc = nn.Linear(2 * transformer_d_model, heatmap_size * heatmap_size)
        self.sigmoid = nn.Sigmoid()
        self.heatmap_size = heatmap_size
        
    def forward(self, mag1, phase1, mag2, phase2):
        feat1 = self.branch(mag1, phase1)  # [B, transformer_d_model]
        feat2 = self.branch(mag2, phase2)  # [B, transformer_d_model]
        combined = torch.cat([feat1, feat2], dim=1)  # [B, 2*transformer_d_model]
        out = self.fc(combined)
        out = self.sigmoid(out)
        heatmap = out.view(-1, 1, self.heatmap_size, self.heatmap_size)
        return heatmap

