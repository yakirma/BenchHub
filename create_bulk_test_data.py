
import zipfile
import os
import shutil
import tempfile
import base64

# 1x1 Red Pixel PNG
DUMMY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

def create_valid_png(path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(DUMMY_PNG_B64))

def create_sub_zip(submission_name, output_dir, samples=["sample1", "sample2"]):
    """
    Creates a submission zip with:
    - predictions.csv (required for single sub detection if at root, but strict requirement depends on heuristics)
    - git.info
    - metric_custom_a/ (Custom Metric)
    - custom_vis/ (Custom Visualization)
    - NO hist_* or standard peaks
    """
    zip_path = os.path.join(output_dir, f"{submission_name}.zip")
    
    # Create temp dir structure for the submission content
    temp_sub_dir = os.path.join(output_dir, f"{submission_name}_content")
    os.makedirs(temp_sub_dir, exist_ok=True)
    
    # 1. git.info
    with open(os.path.join(temp_sub_dir, "git.info"), "w") as f:
        git_info = {
            "commit": "abc1234", 
            "commit_sha": "abc1234", # Legacy support
            "branch": "feature/optimization",
            "repo_url": "http://github.com/example/repo", # Legacy support
            "message": "Improved peak detection algorithm"
        }
        json.dump(git_info, f)

    # 2. Custom Metric: metric_accuracy
    metric_dir = os.path.join(temp_sub_dir, "metric_accuracy")
    os.makedirs(metric_dir, exist_ok=True)
    for sample in samples:
        with open(os.path.join(metric_dir, f"{sample}.txt"), "w") as f:
            f.write("0.95") # Dummy accuracy

    # 3. Custom Visualization: heatmaps
    vis_dir = os.path.join(temp_sub_dir, "heatmaps")
    os.makedirs(vis_dir, exist_ok=True)
    for sample in samples:
        create_valid_png(os.path.join(vis_dir, f"{sample}.png"))

    # 4. predictions.csv
    with open(os.path.join(temp_sub_dir, "predictions.csv"), "w") as f:
        f.write("sample_id,prediction\n")
        for sample in samples:
            f.write(f"{sample},0.5\n")

    # 5. Tags (Method 1: tags.txt)
    if "alpha" in submission_name:
        with open(os.path.join(temp_sub_dir, "tags.txt"), "w") as f:
            f.write("production, v1.0, alpha-team")

    # 6. Tags (Method 2: tags/ folder)
    if "beta" in submission_name:
        tags_dir = os.path.join(temp_sub_dir, "tags")
        os.makedirs(tags_dir, exist_ok=True)
        # Create empty files as tags
        open(os.path.join(tags_dir, "experimental"), "w").close()
        open(os.path.join(tags_dir, "beta-team"), "w").close()

    # Zip it up
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for root, dirs, files in os.walk(temp_sub_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, temp_sub_dir)
                zf.write(abs_path, rel_path)
    
    # Cleanup temp content dir
    shutil.rmtree(temp_sub_dir)
    return zip_path

def create_bulk_zip():
    temp_dir = tempfile.mkdtemp()
    try:
        print("Generating submission 1...")
        sub1_path = create_sub_zip("submission_alpha", temp_dir)
        
        print("Generating submission 2...")
        sub2_path = create_sub_zip("submission_beta", temp_dir)
        
        bulk_zip_path = os.path.abspath("bulk_submissions_custom.zip")
        with zipfile.ZipFile(bulk_zip_path, 'w') as zf:
            zf.write(sub1_path, "submission_alpha.zip")
            zf.write(sub2_path, "submission_beta.zip")
            
        print(f"Created bulk zip with custom fields at: {bulk_zip_path}")
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    create_bulk_zip()
