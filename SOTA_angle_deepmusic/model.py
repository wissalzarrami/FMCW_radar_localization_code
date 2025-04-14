import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMUSICSubCNN(nn.Module):
    """
    17-layer CNN matching the article's description for a single subregion.

    Layer indexing from the paper:
      f(1):  Input layer  (not an actual nn layer, just input)
      f(2):  1st Conv     (5x5, out_channels=256)
      f(3):  BN after 1st conv
      f(4):  ReLU after 1st BN
      f(5):  2nd Conv     (5x5, out_channels=256)
      f(6):  BN
      f(7):  ReLU
      f(8):  3rd Conv     (3x3, out_channels=256)
      f(9):  BN
      f(10): ReLU
      f(11): 4th Conv     (3x3, out_channels=256)
      f(12): BN
      f(13): ReLU
      f(14): Fully connected
      f(15): Dropout
      f(16): Softmax
      f(17): Final output (regression) of size L_sub
    """

    def __init__(self, M=2, L_sub=45):
        """
        Args:
          M (int): Number of sensors => input shape is [3, M, M].
          L_sub (int): Output size for this subregion's MUSIC sub-spectrum.
        """
        super().__init__()
        self.M = M
        self.L_sub = L_sub
        
        # f(2): 1st conv (5x5, 256 filters)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, padding=2)
        # f(3): BN
        self.bn1   = nn.BatchNorm2d(256)
        
        # f(5): 2nd conv (5x5)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        # f(6): BN
        self.bn2   = nn.BatchNorm2d(256)
        
        # f(8): 3rd conv (3x3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # f(9): BN
        self.bn3   = nn.BatchNorm2d(256)
        
        # f(11): 4th conv (3x3)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # f(12): BN
        self.bn4   = nn.BatchNorm2d(256)
        
        # f(14): Fully-connected layer
        # After 4 conv layers, shape => (256, M, M) => flatten => 256*M*M
        self.fc1 = nn.Linear(256 * M * M, 1024)
        
        # f(15): Dropout
        #self.dropout = nn.Dropout(p=0.5)
        
        # f(16): Softmax (applied to an intermediate dimension, e.g. 1024)
        #   We'll do so as a separate layer: we can store it with nn.Softmax(dim=1).
       # self.softmx = nn.Softmax(dim=1)

        # f(17): Final output layer => regression => size L_sub
        self.fc2 = nn.Linear(1024, L_sub)

    def forward(self, x):
        """
        x shape: (batch_size, 3, M, M) => covariance matrix channels
        Returns: (batch_size, L_sub) => predicted MUSIC sub-spectrum
        """
        # f(2)->f(3)->f(4): conv1 -> BN1 -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # f(5)->f(6)->f(7): conv2 -> BN2 -> ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # f(8)->f(9)->f(10): conv3 -> BN3 -> ReLU
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # f(11)->f(12)->f(13): conv4 -> BN4 -> ReLU
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Flatten => shape (batch_size, 256*M*M)
        x = x.view(x.size(0), -1)
        
        # f(14): fully connected
        x = self.fc1(x)
        
        # f(15): dropout
       # x = self.dropout(x)
        
        # f(16): softmax
       # x = self.softmx(x)
        
        # f(17): final regression layer -> L_sub
        out = self.fc2(x)
        return out


class DeepMUSICPartitionedNet(nn.Module):
    """
    Q identical 17-layer sub-CNNs from the paper.
    Each sub-CNN outputs size L_sub. Combined => full size N=Q*L_sub.
    """
    def __init__(self, M=2, N=180, Q=3):
        super().__init__()
        self.N = N
        self.Q = Q
        self.L_sub = N // Q
        
        self.cnns = nn.ModuleList([
            DeepMUSICSubCNN(M=M, L_sub=self.L_sub) for _ in range(Q)
        ])
    
    def forward(self, x):
        """
        x shape: (batch_size, 3, M, M)
        returns: (batch_size, N)
        """
        outputs = []
        for q in range(self.Q):
            sub_out = self.cnns[q](x)   # shape (batch_size, L_sub)
            outputs.append(sub_out)
        return torch.cat(outputs, dim=1)  # (batch_size, Q*L_sub) = (batch_size, N)
