import os
import torch
import numpy as np
import re
import pandas as pd
from torch.utils.data import DataLoader
from model import SimpleDNNClassifier
from Polarloader import RadarDataset

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_centers = np.linspace(1.0, 5.0, 10)  # 10 classes from 1.0m to 5.0m (0.5m steps)

# Load model
model = SimpleDNNClassifier(input_dim=1048576, output_dim=10)  # Update input_dim
model.load_state_dict(torch.load(
    '/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/SOTA1_DNN/simple_dnn_model_stratified.pth'
))
model.to(device)
model.eval()

"""
# Load test paths: loul hedha
with open('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/SOTA1/test_paths.txt', 'r') as f:
    test_paths = [line.strip() for line in f.readlines()]
"""
with open('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/stratifyv2_GTnew__2labs/test_paths_WORK_WITH_THIS.txt', 'r') as f:
    test_paths = [line.strip() for line in f.readlines()]
results = []


def parse_distance(folder_name):
    match = re.search(r'_([\d.]+)m', folder_name)
    return float(match.group(1)) if match else None

for folder in test_paths:
    try:
        # Get ground truth
        folder_name = os.path.basename(folder)
        gt_distance = parse_distance(folder_name)
        if gt_distance is None:
            continue

        # Load dataset
        dataset = RadarDataset(folder)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Run inference
        all_preds = []
        with torch.no_grad():
            for mag1, ph1, mag2, ph2, _ in loader:
                inputs = [t.to(device) for t in [mag1, ph1, mag2, ph2]]
                outputs = model(*inputs)
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        # Convert to distances
        pred_classes, counts = np.unique(all_preds, return_counts=True)
        main_class = pred_classes[np.argmax(counts)]
        pred_distance = class_centers[main_class]

        # Save results
        results.append({
            "Folder": folder_name,
            "GT Distance": gt_distance,
            "Predicted Distance": pred_distance,
            "Error": abs(pred_distance - gt_distance)
        })

    except Exception as e:
        print(f"Error processing {folder_name}: {e}")
        continue

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.round({"GT Distance": 3, "Predicted Distance": 3, "Error": 3})
results_df.to_csv('/store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/SOTA1_DNN/distance_results_LAST.csv', index=False)

print("Inference complete!")
print(f"Average error: {results_df['Error'].mean():.3f} meters")
print(f"Results saved to: /store/wizar/HAR_code_complex/radardataM/radar_detection1/wRadar/Polar/SOTA1_DNN/distance_results_LAST.csv")