import numpy as np
import os
import zipfile
import json
import random
import shutil
import hashlib

def generate_depth_dataset(num_samples=10, dataset_name="dataset_6_raw_fields"):
    """
    Generates a dataset based on dataset_6 but with multiple raw depth/reflectivity maps.
    """
    dataset_dir = dataset_name

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    # Create directories
    hist_dir = os.path.join(dataset_dir, 'hist')
    pick_dir = os.path.join(dataset_dir, 'pick')
    config_dir = os.path.join(dataset_dir, 'config')
    wave_shape_dir = os.path.join(dataset_dir, 'wave_shape')
    tags_dir = os.path.join(dataset_dir, 'tags')
    
    # New raw fields
    raw_depth_dir = os.path.join(dataset_dir, 'raw_depth_gt')
    raw_refl_dir = os.path.join(dataset_dir, 'raw_reflectivity_gt')
    
    all_dirs = [hist_dir, pick_dir, config_dir, wave_shape_dir, tags_dir, raw_depth_dir, raw_refl_dir]
    for d in all_dirs:
        os.makedirs(d)

    # Create git.info file
    git_info = {
        "commit": hashlib.sha1(os.urandom(24)).hexdigest(),
        "branch": "main",
        "message": f"Generated multi-field raw dataset version of {dataset_name}"
    }
    with open(os.path.join(dataset_dir, 'git.info'), 'w') as f:
        json.dump(git_info, f, indent=4)

    print(f"Generating {num_samples} samples...")
    for i in range(1, num_samples + 1):
        sample_name = f"sample{i}"

        # 1. Histogram data (Dummy)
        blocks = 500
        bins = np.linspace(0, 100, blocks)
        counts = np.random.randint(0, 100, size=blocks)
        np.savez_compressed(os.path.join(hist_dir, f"{sample_name}.npz"), bins=bins, counts=counts)

        # 2. Ground truth peak
        with open(os.path.join(pick_dir, f"{sample_name}.txt"), 'w') as f:
            f.write(str(50.0))

        # 3. Wave Shape
        with open(os.path.join(wave_shape_dir, f"{sample_name}.txt"), 'w') as f:
            f.write('gaussian')

        # 4. Config file
        config = { "param": "value" }
        with open(os.path.join(config_dir, f"{sample_name}.json"), 'w') as f:
            json.dump(config, f, indent=4)

        # 5. Tags
        with open(os.path.join(tags_dir, f"{sample_name}.txt"), 'w') as f:
            f.write('multi_raw_test')

        # 6. DEPTH MAP (3x3)
        depth_map = np.random.rand(3, 3) * 100
        depth_filename = f"{sample_name}_3x3.npz"
        np.savez_compressed(os.path.join(raw_depth_dir, depth_filename), data=depth_map)

        # 7. REFLECTIVITY MAP (3x3)
        refl_map = np.random.rand(3, 3) * 255
        refl_filename = f"{sample_name}_3x3.npz"
        np.savez_compressed(os.path.join(raw_refl_dir, refl_filename), data=refl_map)

    # Zip the directory
    zip_filename = f"{dataset_name}.zip"
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(dataset_dir, '..'))
                zipf.write(os.path.join(root, file), arcname)
    
    # Clean up the directory
    shutil.rmtree(dataset_dir)
    print(f"Generated dataset archive: {zip_filename}")

if __name__ == "__main__":
    generate_depth_dataset(num_samples=5, dataset_name="dataset_6_raw_fields")
