import requests
import os
import sys
import argparse

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:6060"  # Adjust port as needed
# ---------------------

def upload_dataset(zip_file_path, dataset_name=None, project_name=None):
    """Upload a dataset ZIP file to the server.
    
    Args:
        zip_file_path: Path to the ZIP file to upload
        dataset_name: Optional name for the dataset (defaults to filename without extension)
        project_name: Optional name of the project to associate this dataset with
    """
    if not os.path.exists(zip_file_path):
        print(f"Error: File not found: {zip_file_path}")
        sys.exit(1)
    
    if not zip_file_path.endswith('.zip'):
        print(f"Warning: File does not have .zip extension: {zip_file_path}")
    
    # Use filename (without extension) as dataset name if not provided
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(zip_file_path))[0]
    
    print(f"Uploading dataset from: {zip_file_path}")
    print(f"Dataset name: {dataset_name}")
    if project_name:
        print(f"Target Project: {project_name}")
    
    upload_url = f"{API_BASE_URL}/api/dataset/upload"
    
    try:
        with open(zip_file_path, 'rb') as f:
            files = {
                'dataset_zip': (os.path.basename(zip_file_path), f, 'application/zip')
            }
            data = {
                'dataset_name': dataset_name
            }
            if project_name:
                data['project_name'] = project_name
            
            print(f"Uploading to {upload_url}...")
            response = requests.post(upload_url, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            
            print("✓ Upload Successful!")
            print(f"  Dataset ID: {result.get('id')}")
            print(f"  Dataset Name: {result.get('name')}")
            if result.get('project_id'):
                print(f"  Project ID: {result.get('project_id')}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Upload Failed: {e}")
        if 'response' in locals() and response is not None:
             # Try to print JSON error if available
            try:
                print(f"  Server Error: {response.json().get('error', response.text)}")
            except:
                print(f"  Response Text: {response.text}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a dataset ZIP file to the Bench-Hub server.")
    parser.add_argument("zip_file", help="Path to the dataset .zip file")
    parser.add_argument("--name", "-n", help="Custom name for the dataset (defaults to filename)", default=None)
    parser.add_argument("--project", "-p", help="Name of the project to associate with", default='dTOF')
    
    args = parser.parse_args()
    
    upload_dataset(args.zip_file, args.name, args.project)
