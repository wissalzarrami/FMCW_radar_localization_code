import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Import the partitioned model and dataset classes
from model import DeepMUSICPartitionedNet  # <--- Q sub-CNN architecture
from Polarloader import RadarMUSICDataset


def load_file_paths(directory):
    """
    Returns a list of **sub-directory** paths within 'directory'.
    Each sub-directory is expected to contain 9 '_a1.cf32' + 9 '_a2.cf32'.
    """
    subfolders = []
    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        if os.path.isdir(full_path):
            subfolders.append(full_path)
    return subfolders


def build_tensor_dataset(paths, num_signals=1, M=2, N=180):
    """
    Given a list of folder paths (each with 9 '_a1.cf32' + 9 '_a2.cf32'),
    builds a TensorDataset of (X, y).

    - X: shape (num_folders, 3, M, M), the covariance matrix channels
    - y: shape (num_folders, N), the MUSIC spectrum across N angles
    """
    combined_X = []
    combined_y = []
    for folder in paths:
        ds = RadarMUSICDataset(folder, num_signals=num_signals, angle_grid=None)
        # ds has length=1 => ds[0] = (X_cov, p_full)
        X_cov, p_full = ds[0]
        combined_X.append(X_cov.unsqueeze(0))  # shape => (1, 3, M, M)
        combined_y.append(p_full.unsqueeze(0)) # shape => (1, N)

    X_all = torch.cat(combined_X, dim=0)  # (num_samples, 3, M, M)
    y_all = torch.cat(combined_y, dim=0)  # (num_samples, N)

    return TensorDataset(X_all, y_all)


def train_deepmusic_partitioned(
    train_dataset,
    val_dataset=None,
    M=2,
    N=180,
    Q=8,
    epochs=10,
    batch_size=1,
    lr=1e-3
):
    """
    Train the Q-partitioned DeepMUSIC model, where each of the Q sub-CNNs
    predicts a subregion of length N/Q.

    Args:
      train_dataset (TensorDataset): (X, y) for training
      val_dataset (TensorDataset): (X, y) for validation (optional)
      M, N, Q: array size, total angles, number of subregions
      epochs, batch_size, lr: training hyperparameters
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    model = DeepMUSICPartitionedNet(M=M, N=N, Q=Q)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    L_sub = N // Q  # number of angles in each subregion

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # X_batch: (batch_size, 3, M, M)
            # y_batch: (batch_size, N)
            optimizer.zero_grad()

            pred_full = model(X_batch)  # shape => (batch_size, N)

            # Sum MSE over Q sub-spectra
            loss = 0.0
            for q in range(Q):
                pred_sub = pred_full[:, q*L_sub:(q+1)*L_sub]
                label_sub = y_batch[:, q*L_sub:(q+1)*L_sub]
                loss_sub = criterion(pred_sub, label_sub)
                loss += loss_sub

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # (Optional) Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    pred_val = model(X_val)
                    loss_val = 0.0
                    for q in range(Q):
                        pred_sub = pred_val[:, q*L_sub:(q+1)*L_sub]
                        label_sub = y_val[:, q*L_sub:(q+1)*L_sub]
                        loss_val += criterion(pred_sub, label_sub)
                    val_loss += loss_val.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{epochs}] -> train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] -> train_loss: {avg_train_loss:.4f}")

    return model


if __name__ == "__main__":
    # 1) Path to the main directory containing multiple sub-folders
    data_dir = "/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/separate_files"

    # 2) Gather all sub-folder paths
    all_paths = load_file_paths(data_dir)
    print(f"Found {len(all_paths)} sub-folders total.")

    # 3) Split into 80% train, 20% val
    train_paths, val_paths = train_test_split(all_paths, test_size=0.2, random_state=42)
    print(f"Total train folders: {len(train_paths)}, val folders: {len(val_paths)}")

    # 4) Build TensorDatasets
    train_ds = build_tensor_dataset(train_paths, num_signals=1, M=2, N=180)
    val_ds   = build_tensor_dataset(val_paths, num_signals=1, M=2, N=180)

    print(f"train_ds length: {len(train_ds)} samples, val_ds length: {len(val_ds)} samples.")

    # 5) Train the partitioned model (Q=8 subregions)
    model = train_deepmusic_partitioned(
        train_dataset=train_ds,
        val_dataset=val_ds,
        M=2,
        N=180,
        Q=8,
        epochs=10,
        batch_size=1,
        lr=1e-3
    )

    # 6) Save the model weights
    torch.save(model.state_dict(), "deepmusic_partitioned_model.pth")
    print("Saved model to deepmusic_partitioned_model.pth")
