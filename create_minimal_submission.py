"""
Test script to create a MINIMAL submission with only custom metrics.
This verifies that custom fields work without standard peaks or histograms.
"""

import os
import tempfile
import zipfile

def create_minimal_submission_zip():
    """Create a minimal test submission with ONLY custom metrics (no standard peaks, no hist_*)"""
    
    temp_dir = tempfile.mkdtemp()
    submission_dir = os.path.join(temp_dir, 'minimal_submission')
    os.makedirs(submission_dir)
    
    # Create ONLY custom metric folders - NO standard peaks or hist_*
    metric_accuracy_dir = os.path.join(submission_dir, 'metric_accuracy2')
    os.makedirs(metric_accuracy_dir)
    
    metric_precision_dir = os.path.join(submission_dir, 'metric_precision2')
    os.makedirs(metric_precision_dir)
    
    # Create sample files for custom metrics only
    # NOTE: These sample names must match your dataset sample names
    sample_names = ['sample1', 'sample2', 'sample3']
    
    for i, sample_name in enumerate(sample_names):
        # Custom metric: accuracy
        with open(os.path.join(metric_accuracy_dir, f'{sample_name}.txt'), 'w') as f:
            f.write(str(0.88 + i * 0.03))  # 0.88, 0.91, 0.94
        
        # Custom metric: precision
        with open(os.path.join(metric_precision_dir, f'{sample_name}.txt'), 'w') as f:
            f.write(str(0.82 + i * 0.04))  # 0.82, 0.86, 0.90
    
    # Create zip file
    zip_path = os.path.join(temp_dir, 'minimal_submission.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
    
    print(f"✅ Minimal test submission created at: {zip_path}")
    print(f"\nContents:")
    print(f"  minimal_submission/")
    print(f"    metric_accuracy/")
    for sample in sample_names:
        print(f"      {sample}.txt")
    print(f"    metric_precision/")
    for sample in sample_names:
        print(f"      {sample}.txt")
    
    return zip_path

if __name__ == '__main__':
    print("Creating MINIMAL submission (only custom metrics)...\n")
    
    zip_path = create_minimal_submission_zip()
    
    print(f"\n{'='*60}")
    print("To test:")
    print(f"1. Make sure you have a dataset with samples: sample1, sample2, sample3")
    print(f"2. Create a leaderboard for that dataset")
    print(f"3. Upload this submission: {zip_path}")
    print(f"4. Check if metric_accuracy and metric_precision appear in:")
    print(f"   - Leaderboard view (as sortable columns)")
    print(f"   - Comparison view (as metric columns)")
    print(f"5. Check server logs for custom field detection debug output")
    print(f"{'='*60}\n")
