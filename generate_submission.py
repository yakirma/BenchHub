import numpy as np
import os
import zipfile
import random
import shutil
import argparse
import re
import json
import hashlib

def get_next_submission_number(dataset_base_name):
    """Finds the next available submission number for a given dataset."""
    pattern = re.compile(f"^{re.escape(dataset_base_name)}_submission_(\\d+)\\.zip$")
    existing_submissions = [f for f in os.listdir('.') if pattern.match(f)]
    
    if not existing_submissions:
        return 1
        
    numbers = [int(pattern.match(s).group(1)) for s in existing_submissions]
    return max(numbers) + 1 if numbers else 1

def generate_submission(dataset_zip_path):
    """
    Generates a numbered dummy submission for a given dataset zip file.
    """
    if not os.path.exists(dataset_zip_path):
        print(f"Error: Dataset zip file not found at '{dataset_zip_path}'")
        return

    dataset_base_name = os.path.splitext(os.path.basename(dataset_zip_path))[0]
    submission_number = get_next_submission_number(dataset_base_name)
    submission_name = f"{dataset_base_name}_submission_{submission_number}"
    submission_dir = submission_name

    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)

    # Create directories for submission
    metric_peak_dir = os.path.join(submission_dir, 'metric_peak')
    filtered_hist_dir = os.path.join(submission_dir, 'hist_filtered')
    os.makedirs(metric_peak_dir)
    os.makedirs(filtered_hist_dir)

    # ... (skipping git info) ...

    # ... (in loop) ...

            # 1. Predicted peak
            metric_peak = gt_pick + random.uniform(-2, 2) # Add some noise
            with open(os.path.join(metric_peak_dir, f"{sample_name}.txt"), 'w') as f:
                f.write(str(metric_peak))

            # 2. Filtered histogram
            # Read original histogram to base the filtered one on it
            hist_path = os.path.join(zip_root_dir, 'hist', f'{sample_name}.npz')
            with zip_ref.open(hist_path) as f:
                with np.load(f) as data:
                    bins = data['bins']
                    counts = data['counts']
            
            # Diverse Filtering Strategies
            strategy = random.choice(['clean', 'standard', 'noisy', 'aggressive'])
            
            if signal_shape == 'square':
                # Higher fidelity square signal simulation
                square_amplitude = random.randint(3000, 15000)
                square_width = random.uniform(3, 10)
                square_start = metric_peak - square_width / 2
                square_end = metric_peak + square_width / 2
                filtered_counts = np.where((bins >= square_start) & (bins <= square_end), square_amplitude, 0).astype(float)
                
                # Different noise additions based on strategy
                noise_map = {'clean': 5, 'standard': 20, 'noisy': 100, 'aggressive': 2}
                blur_map = {'clean': 1, 'standard': 3, 'noisy': 3, 'aggressive': 10}
                
                filtered_counts += np.random.randint(0, noise_map[strategy], size=len(bins))
                filtered_counts = np.convolve(filtered_counts, np.ones(blur_map[strategy])/blur_map[strategy], mode='same')
            else:
                # Gaussian or Mixed signals
                if strategy == 'clean':
                    # High quality smoothing
                    filtered_counts = np.convolve(counts, np.array([0.1, 0.8, 0.1]), mode='same')
                    filtered_counts += np.random.randint(0, 5, size=len(bins))
                elif strategy == 'aggressive':
                    # Heavy blur (lowers entropy)
                    window = 11
                    filtered_counts = np.convolve(counts, np.ones(window)/window, mode='same')
                elif strategy == 'noisy':
                    # Poor filtering (high entropy)
                    filtered_counts = counts * random.uniform(0.8, 1.2)
                    filtered_counts += np.random.randint(0, 80, size=len(bins))
                else: # standard
                    filtered_counts = np.convolve(counts, np.ones(5)/5, mode='same')
                    filtered_counts += np.random.randint(0, 20, size=len(bins))

            filtered_counts = np.maximum(0, filtered_counts).astype(int)

            
            np.savez_compressed(os.path.join(filtered_hist_dir, f"{sample_name}.npz"), bins=bins, counts=filtered_counts)

            np.savez_compressed(os.path.join(filtered_hist_dir, f"{sample_name}.npz"), bins=bins, counts=filtered_counts)

    # Zip the submission directory
    zip_filename = f"{submission_name}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(submission_dir):
            for file in files:
                arcname = os.path.relpath(os.path.join(root, file), submission_dir)
                zipf.write(os.path.join(root, file), arcname)

    # Clean up the directory
    shutil.rmtree(submission_dir)
    print(f"Generated submission: {zip_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dummy submission for a given dataset.")
    parser.add_argument("dataset_zip", help="Path to the dataset zip file.")
    args = parser.parse_args()
    
    generate_submission(args.dataset_zip)
