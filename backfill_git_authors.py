#!/usr/bin/env python3
"""
Backfill Git Author Information from Git Branches
This script uses the git branch name stored in the database to extract the author name
using the command: git log -1 --format='%an' origin/BRANCH_NAME
"""

import sqlite3
import os
import sys
import subprocess

# Import local configuration (optional, with fallback)
try:
    from local_config import GIT_REPO_PATH
except ImportError:
    GIT_REPO_PATH = None  # Fallback to None if config doesn't exist

def get_db_path():
    """Get the database path from the app configuration."""
    user_home = os.path.expanduser("~")
    dtof_data_dir = os.path.join(user_home, ".dtofbenchmarking")
    db_path = os.path.join(dtof_data_dir, 'database.db')
    return db_path

def get_author_from_branch(branch_name, repo_path=None):
    """
    Get the author name from a git branch using remote origin.
    Uses: git log -1 --format='%an' origin/BRANCH_NAME
    
    Args:
        branch_name: The git branch name
        repo_path: Optional path to the git repository. If None, uses GIT_REPO_PATH or current directory.
    
    Returns:
        Author name or None if not found
    """
    if not branch_name or branch_name == 'N/A' or branch_name == '':
        return None
    
    git_path = repo_path or GIT_REPO_PATH
    
    try:
        # Fetch from remote first to ensure we have latest data
        fetch_cmd = ['git', 'fetch', 'origin']
        if git_path:
            fetch_cmd = ['git', '-C', git_path, 'fetch', 'origin']
        
        # Run fetch (suppress output, don't fail if it errors)
        subprocess.run(fetch_cmd, capture_output=True, text=True, timeout=30)
        
        # Get author from remote branch
        cmd = ['git', 'log', '-1', '--format=%an', f'origin/{branch_name}']
        if git_path:
            cmd = ['git', '-C', git_path, 'log', '-1', '--format=%an', f'origin/{branch_name}']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            author = result.stdout.strip()
            if author:
                return author
        
        return None
    except Exception as e:
        print(f"    Error querying git: {e}")
        return None


def backfill_authors_from_git(repo_path=None):
    """Backfill git author information using git commands."""
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)
    
    print(f"Using database: {db_path}")
    effective_repo_path = repo_path or GIT_REPO_PATH
    if effective_repo_path:
        print(f"Using git repository: {effective_repo_path}")
    else:
        print(f"Using current directory as git repository")
    print()
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Process Datasets
        print("=" * 60)
        print("BACKFILLING DATASET AUTHORS FROM GIT BRANCHES")
        print("=" * 60)
        
        cursor.execute("SELECT id, name, git_branch, git_author FROM dataset WHERE git_branch IS NOT NULL AND git_branch != ''")
        datasets = cursor.fetchall()
        
        dataset_updated = 0
        dataset_already_set = 0
        dataset_not_found = 0
        
        for dataset_id, dataset_name, git_branch, current_author in datasets:
            if current_author:
                print(f"⏭️  Dataset '{dataset_name}': Already has author '{current_author}'")
                dataset_already_set += 1
                continue
            
            author = get_author_from_branch(git_branch, repo_path)
            
            if author:
                cursor.execute("UPDATE dataset SET git_author = ? WHERE id = ?", (author, dataset_id))
                print(f"✅ Dataset '{dataset_name}' (branch {git_branch}): {author}")
                dataset_updated += 1
            else:
                print(f"❌ Dataset '{dataset_name}' (branch {git_branch}): Could not find author")
                dataset_not_found += 1
        
        print(f"\nDataset Summary:")
        print(f"  Updated: {dataset_updated}")
        print(f"  Already set: {dataset_already_set}")
        print(f"  Not found in git: {dataset_not_found}")
        
        # Process Submissions
        print("\n" + "=" * 60)
        print("BACKFILLING SUBMISSION AUTHORS FROM GIT BRANCHES")
        print("=" * 60)
        
        cursor.execute("SELECT id, name, git_branch, git_author FROM submission WHERE git_branch IS NOT NULL AND git_branch != ''")
        submissions = cursor.fetchall()
        
        submission_updated = 0
        submission_already_set = 0
        submission_not_found = 0
        
        for submission_id, submission_name, git_branch, current_author in submissions:
            if current_author:
                print(f"⏭️  Submission '{submission_name}': Already has author '{current_author}'")
                submission_already_set += 1
                continue
            
            author = get_author_from_branch(git_branch, repo_path)
            
            if author:
                cursor.execute("UPDATE submission SET git_author = ? WHERE id = ?", (author, submission_id))
                print(f"✅ Submission '{submission_name}' (ID: {submission_id}, branch {git_branch}): {author}")
                submission_updated += 1
            else:
                print(f"❌ Submission '{submission_name}' (ID: {submission_id}, branch {git_branch}): Could not find author")
                submission_not_found += 1
        
        print(f"\nSubmission Summary:")
        print(f"  Updated: {submission_updated}")
        print(f"  Already set: {submission_already_set}")
        print(f"  Not found in git: {submission_not_found}")
        
        # Commit changes
        if dataset_updated > 0 or submission_updated > 0:
            conn.commit()
            print("\n" + "=" * 60)
            print("BACKFILL COMPLETE")
            print("=" * 60)
            print(f"✅ Successfully updated {dataset_updated} dataset(s) and {submission_updated} submission(s)")
            print("\nYou can now refresh your browser to see the author information!")
        else:
            print("\n" + "=" * 60)
            print("NO UPDATES NEEDED")
            print("=" * 60)
            if dataset_already_set > 0 or submission_already_set > 0:
                print("All records already have author information.")
            else:
                print("No git branches found in database or branches not accessible in git repository.")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    print("=" * 60)
    print("BenchHub Git Author Backfill Script (from Git Branches)")
    print("=" * 60)
    print()
    print("This script will use the git branch names stored in the database")
    print("to extract author information from the remote origin.")
    print()
    print("Command used: git log -1 --format='%an' origin/BRANCH_NAME")
    print()
    
    # Prioritize GIT_REPO_PATH from local_config.py
    repo_path = GIT_REPO_PATH
    
    if repo_path:
        if os.path.exists(repo_path):
            print(f"Using GIT_REPO_PATH from local_config.py: {repo_path}")
        else:
            print(f"WARNING: GIT_REPO_PATH in local_config.py does not exist: {repo_path}")
            print("Enter the path to your git repository")
            print("(press Enter to use current directory):")
            user_input = input("> ").strip()
            if user_input:
                repo_path = user_input
            else:
                repo_path = None
    else:
        # Ask for git repository path if not in config
        print("Enter the path to your git repository")
        print("(press Enter to use current directory):")
        repo_path = input("> ").strip()
        if not repo_path:
            repo_path = None
    
    if repo_path:
        if not os.path.exists(repo_path):
            print(f"ERROR: Path '{repo_path}' does not exist")
            sys.exit(1)
        elif not os.path.exists(os.path.join(repo_path, '.git')):
            print(f"WARNING: '{repo_path}' does not appear to be a git repository")
            response = input("Continue anyway? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("Aborted.")
                sys.exit(0)
    else:
        print("Using current directory")
    
    print()
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        print()
        # Pass repo_path. If it was GIT_REPO_PATH, the function handles it.
        backfill_authors_from_git(repo_path)
    else:
        print("Aborted.")
        sys.exit(0)
