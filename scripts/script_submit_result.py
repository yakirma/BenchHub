import requests
import zipfile
import os
import argparse
import sys

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:6060"  # Adjust port as needed
CACHE_DIR = "./dataset_cache"
# ---------------------

def ensure_dataset(project_name, leaderboard_name, force_download=False):
    """Checks LB info, downloads/caches dataset if needed."""
    
    # 1. Get Leaderboard Info to find Dataset ID
    info_url = f"{API_BASE_URL}/{project_name}/api/leaderboard/by_name/{leaderboard_name}/info"
    try:
        resp = requests.get(info_url)
        resp.raise_for_status()
        info = resp.json()
        leaderboard_id = info['id']
        dataset_id = info['dataset']['id']
        dataset_name = info['dataset']['name']
        print(f"Leaderboard '{leaderboard_name}' (ID: {leaderboard_id}) uses Dataset ID {dataset_id} ('{dataset_name}')")
    except Exception as e:
        print(f"Error fetching leaderboard info from {info_url}: {e}")
        sys.exit(1)

    # 2. Check Cache
    dataset_folder = os.path.join(CACHE_DIR, str(dataset_id))
    local_zip_path = os.path.join(CACHE_DIR, f"{dataset_id}.zip")
    
    # If folder exists and we don't force download, assume it's ready
    if os.path.exists(dataset_folder) and not force_download:
        print(f"Dataset found in cache: {dataset_folder}")
        return dataset_folder
    
    # 3. Download if needed
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(local_zip_path) or force_download:
        print(f"Downloading dataset {dataset_id}...")
        download_url = f"{API_BASE_URL}/{project_name}/api/dataset/{dataset_id}/download"
        try:
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(local_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)
            
    # 4. Unzip
    print(f"Unzipping to {dataset_folder}...")
    if os.path.exists(dataset_folder):
        # Clean up if needed, or simple overwrite
        pass 
    
    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_folder)
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        sys.exit(1)
        
    return dataset_folder

def submit_result(project_name, leaderboard_name, submission_zip_path=None, submission_name=None, force_download=False):
    """Submit a result to a leaderboard.
    
    Args:
        project_name: Name of the project
        leaderboard_name: Name of the leaderboard to submit to
        submission_zip_path: Path to submission ZIP file
        submission_name: Name for the submission (defaults to filename)
        force_download: Force redownload of dataset
    """
    # 1. Get leaderboard ID from name
    info_url = f"{API_BASE_URL}/{project_name}/api/leaderboard/by_name/{leaderboard_name}/info"
    try:
        resp = requests.get(info_url)
        resp.raise_for_status()
        info = resp.json()
        leaderboard_id = info['id']
        print(f"Found leaderboard '{leaderboard_name}' with ID: {leaderboard_id}")
    except Exception as e:
        print(f"Error fetching leaderboard info: {e}")
        sys.exit(1)
    
    # 2. Verify submission file
    if not submission_zip_path:
        print("Error: submission_zip_path is required")
        sys.exit(1)
    
    # Verify submission file exists
    if not os.path.exists(submission_zip_path):
        print(f"Error: Submission file not found: {submission_zip_path}")
        sys.exit(1)
    
    if not submission_zip_path.endswith('.zip'):
        print(f"Warning: File does not have .zip extension: {submission_zip_path}")
    
    # Use filename (without extension) as submission name if not provided
    if submission_name is None:
        submission_name = os.path.splitext(os.path.basename(submission_zip_path))[0]
    
    print(f"Project: {project_name}")
    print(f"Submitting: {submission_zip_path}")
    print(f"Submission name: {submission_name}")
    
    # 2. Upload Submission
    upload_url = f"{API_BASE_URL}/{project_name}/api/leaderboard/{leaderboard_id}/submission/upload"
    
    try:
        with open(submission_zip_path, 'rb') as f:
            files = {
                'submission_zip': (os.path.basename(submission_zip_path), f, 'application/zip')
            }
            data = {
                'submission_name': submission_name
            }
            
            print(f"Uploading submission to {upload_url}...")
            response = requests.post(upload_url, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            
            print("✓ Submission Upload Successful!")
            print(f"  Message: {result.get('message', 'Submission queued')}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Submission Upload Failed: {e}")
        if 'response' in locals() and response is not None:
            print(f"  Response Text: {response.text}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Submit result via API',
        epilog='Examples:\n'
               '  python script_submit_result.py --project "My Project" "My Leaderboard" submission.zip\n'
               '  python script_submit_result.py --project "My Project" "My Leaderboard" /path/to/submission.zip --name "My Submission"\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('leaderboard_name', type=str, help='Name of the leaderboard')
    parser.add_argument('submission_zip', type=str, help='Path to submission ZIP file')
    parser.add_argument('--project', type=str, required=True, help='Name of the project')
    parser.add_argument('--name', type=str, default=None, help='Custom name for submission (defaults to filename)')
    parser.add_argument('--download-dataset', action='store_true', 
                        help='Download and cache the dataset before submitting (optional, for reference)')
    
    args = parser.parse_args()
    
    # Optionally download dataset if requested
    if args.download_dataset:
        print("Downloading dataset for reference...")
        ensure_dataset(args.project, args.leaderboard_name, force_download=False)
        print()
    
    submit_result(args.project, args.leaderboard_name, args.submission_zip, args.name, force_download=False)
