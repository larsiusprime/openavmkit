import sys
import os

def setup_environment():
    """Sets up repository root path
    """
    # Add the repository root to PYTHONPATH
    repo_root = os.path.abspath("../..")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    print("Environment setup completed.")

