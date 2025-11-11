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


def check_for_different_locality(locality: str):
    """
    Checks the "LOCALITY" environmental variable to see if its set. Returns it if it is, otherwise returns the passed in slug.

    Attributes
    ----------
    locality : str
        The locality slug from the notebook (e.g., "us-nc-guilford").

    Returns
    -------
    locality : str
        The new locality slug (e.g., "us-nc-guilford").
    """

    new_locality = os.getenv("LOCALITY")

    if new_locality:
        print("Found new locality: " + new_locality)
        return new_locality
    else:
        return locality
