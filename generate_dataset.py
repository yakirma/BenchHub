import numpy as np
import os
import zipfile
import json
import random
import shutil
import re
import hashlib

def get_next_dataset_number():
    """Finds the next available dataset number."""
    existing_datasets = [f for f in os.listdir('.') if re.match(r'dataset_\d+\.zip', f)]
    if not existing_datasets:
        return 1
    
    numbers = []
    for ds in existing_datasets:
        match = re.search(r'dataset_(\d+)\.zip', ds)
        if match:
            numbers.append(int(match.group(1)))
    
    if numbers:
        return max(numbers) + 1
    return 1

def generate_dataset(num_samples=5):
    """
    Generates a dummy dataset with the specified number of samples.
    """
    dataset_number = get_next_dataset_number()
    dataset_base_name = f"dataset_{dataset_number}"
    dataset_dir = dataset_base_name

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    # Create directories
    hist_dir = os.path.join(dataset_dir, 'hist')
    pick_dir = os.path.join(dataset_dir, 'pick')
    config_dir = os.path.join(dataset_dir, 'config')
    wave_shape_dir = os.path.join(dataset_dir, 'wave_shape')
    for d in [hist_dir, pick_dir, config_dir, wave_shape_dir]:
        os.makedirs(d)

    # Create git.info file
    git_info = {
        "commit": hashlib.sha1(os.urandom(24)).hexdigest(),
        "branch": random.choice(["main", "develop", "feature/new-thing"]),
        "message": f"Commit message for dataset {dataset_number}"
    }
    with open(os.path.join(dataset_dir, 'git.info'), 'w') as f:
        json.dump(git_info, f, indent=4)

    for i in range(1, num_samples + 1):
        sample_name = f"sample{i}"

        # 3. Signal shape (decide this first)
        shape = random.choice(['gaussian', 'square'])
        with open(os.path.join(wave_shape_dir, f"{sample_name}.txt"), 'w') as f:
            f.write(shape)

        # 1. Histogram data
        num_bins = random.randint(500, 1000)
        bins = np.linspace(0, 100, num_bins)
        peak_pos = random.uniform(20, 80)
        peak_height = random.randint(1000, 10000)
        
        if shape == 'square':
            square_width = random.uniform(3, 15) # Vary width for entropy range
            square_start = peak_pos - square_width / 2
            square_end = peak_pos + square_width / 2
            counts = np.where((bins >= square_start) & (bins <= square_end), peak_height, 0).astype(float)
        else: # Gaussian
            peak_width = random.uniform(0.5, 5.0) # Vary width for entropy range
            counts = peak_height * np.exp(-((bins - peak_pos)**2) / (2 * peak_width**2))

        # Add a secondary peak occasionally
        if random.random() < 0.15: # 15% chance of secondary peak
            sec_peak_pos = random.uniform(10, 90)
            sec_peak_height = peak_height * random.uniform(0.1, 0.4)
            sec_peak_width = random.uniform(1, 4)
            counts += sec_peak_height * np.exp(-((bins - sec_peak_pos)**2) / (2 * sec_peak_width**2))

        # Variable Noise Floor
        noise_level = random.choice([5, 50, 200, 500]) # Diverse noise levels for entropy
        counts += np.random.randint(0, noise_level, size=num_bins)
        
        # Inject occasional anomalies for Trend Overlay testing
        if random.random() < 0.2: # 20% chance of an anomaly
            anomaly_idx = random.randint(0, num_bins - 1)
            counts[anomaly_idx] += peak_height * 0.8 # Significant spike
            
        counts = np.maximum(0, counts).astype(int)

        
        np.savez_compressed(os.path.join(hist_dir, f"{sample_name}.npz"), bins=bins, counts=counts)

        # 2. Ground truth peak
        with open(os.path.join(pick_dir, f"{sample_name}.txt"), 'w') as f:
            f.write(str(peak_pos))

        # 4. Config file
        config = { "param1": random.random(), "param2": random.choice(['A', 'B', 'C']) }
        with open(os.path.join(config_dir, f"{sample_name}.json"), 'w') as f:
            json.dump(config, f, indent=4)

        # Custom fields, config etc. logic remains

    # Zip the directory
    zip_filename = f"{dataset_base_name}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(dataset_dir, '..'))
                zipf.write(os.path.join(root, file), arcname)
    
    # Clean up the directory
    shutil.rmtree(dataset_dir)
    print(f"Generated dataset: {zip_filename}")

if __name__ == "__main__":
    generate_dataset(num_samples=random.randint(10, 100))
