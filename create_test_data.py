"""
Test script to verify custom fields functionality.

This script creates test data to verify:
1. Custom image fields are detected and stored
2. Custom scalar fields are detected and stored  
3. Custom metrics (metric_*) are detected and stored
4. Metric sorting works in leaderboard
"""

import os
import tempfile
import zipfile
import json

def create_test_dataset_zip():
    """Create a test dataset with custom fields"""
    import numpy as np
    
    temp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(temp_dir, 'test_dataset')
    os.makedirs(dataset_dir)
    
    # Create standard folders
    for folder in ['hist', 'pick', 'wave_shape', 'config']:
        os.makedirs(os.path.join(dataset_dir, folder))
    
    # Create custom image folder
    custom_viz_dir = os.path.join(dataset_dir, 'custom_visualization')
    os.makedirs(custom_viz_dir)
    
    # Create custom scalar folder
    custom_scalar_dir = os.path.join(dataset_dir, 'custom_density')
    os.makedirs(custom_scalar_dir)
    
    # Create sample files
    sample_names = ['sample1', 'sample2', 'sample3']
    
    for i, sample_name in enumerate(sample_names):
        # Standard fields - histogram (required!)
        bins = np.linspace(0, 20, 100)
        counts = np.random.poisson(50, 100) + 10  # Some random histogram data
        hist_file = os.path.join(dataset_dir, 'hist', f'{sample_name}.npz')
        np.savez(hist_file, bins=bins, counts=counts)
        
        # Ground truth peak
        with open(os.path.join(dataset_dir, 'pick', f'{sample_name}.txt'), 'w') as f:
            f.write(str(10.5 + i * 0.5))  # 10.5, 11.0, 11.5
        
        # Wave shape
        with open(os.path.join(dataset_dir, 'wave_shape', f'{sample_name}.txt'), 'w') as f:
            f.write('gaussian')
        
        # Config
        with open(os.path.join(dataset_dir, 'config', f'{sample_name}.json'), 'w') as f:
            json.dump({'param1': 100 + i * 10, 'param2': 200 + i * 20}, f)
        
        # Custom scalar field
        with open(os.path.join(custom_scalar_dir, f'{sample_name}.txt'), 'w') as f:
            f.write(str(0.75 + i * 0.05))  # 0.75, 0.80, 0.85
        
        # Note: For image field, you would create actual image files
        # For this test, we'll just create a placeholder text file
        with open(os.path.join(custom_viz_dir, f'{sample_name}.png'), 'w') as f:
            f.write('placeholder')
    
    # Create zip file
    zip_path = os.path.join(temp_dir, 'test_dataset.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
    
    print(f"Test dataset created at: {zip_path}")
    return zip_path

def create_test_submission_zip():
    """Create a test submission with custom metrics"""
    import numpy as np
    
    temp_dir = tempfile.mkdtemp()
    submission_dir = os.path.join(temp_dir, 'test_submission')
    os.makedirs(submission_dir)
    
    # Create standard folders
    for folder in ['metric_peak', 'hist_filtered']:
        os.makedirs(os.path.join(submission_dir, folder))
    
    # Create custom metric folder
    metric_accuracy_dir = os.path.join(submission_dir, 'metric_accuracy')
    os.makedirs(metric_accuracy_dir)
    
    metric_precision_dir = os.path.join(submission_dir, 'metric_precision')
    os.makedirs(metric_precision_dir)
    
    # Create sample files
    sample_names = ['sample1', 'sample2', 'sample3']
    
    for i, sample_name in enumerate(sample_names):
        # Standard fields - predicted peak
        with open(os.path.join(submission_dir, 'metric_peak', f'{sample_name}.txt'), 'w') as f:
            f.write(str(10.3 + i * 0.3))  # 10.3, 10.6, 10.9
        
        # Filtered histogram (required!)
        bins = np.linspace(0, 20, 100)
        counts = np.random.poisson(45, 100) + 5  # Slightly different from GT
        hist_file = os.path.join(submission_dir, 'hist_filtered', f'{sample_name}.npz')
        np.savez(hist_file, bins=bins, counts=counts)
        
        # Custom metrics
        with open(os.path.join(metric_accuracy_dir, f'{sample_name}.txt'), 'w') as f:
            f.write(str(0.9 + i * 0.01))  # 0.9, 0.91, 0.92
        
        with open(os.path.join(metric_precision_dir, f'{sample_name}.txt'), 'w') as f:
            f.write(str(0.85 + i * 0.02))  # 0.85, 0.87, 0.89
    
    # Create zip file
    zip_path = os.path.join(temp_dir, 'test_submission.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
    
    print(f"Test submission created at: {zip_path}")
    return zip_path

if __name__ == '__main__':
    print("Creating test data...")
    print()
    
    dataset_zip = create_test_dataset_zip()
    submission_zip = create_test_submission_zip()
    
    print()
    print("Test data created successfully!")
    print()
    print("To test:")
    print(f"1. Upload the dataset: {dataset_zip}")
    print(f"2. Create a leaderboard for the dataset")
    print(f"3. Upload the submission: {submission_zip}")
    print(f"4. Check the leaderboard for custom metrics (metric_accuracy, metric_precision)")
    print(f"5. Click on metric column headers to test sorting")
