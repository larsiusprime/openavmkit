# This file lets us abstract remote storage services (Azure, AWS, etc.) into a common interface
import os
from datetime import datetime, timedelta, timezone
from typing import Literal

CloudType = Literal[
  "azure"
  # In the future we'll add more types here:
  # "aws",
  # "gcp",
  # "sftp
]

class CloudFile:
  def __init__(self, name: str, last_modified_utc: datetime, size: int):
    self.name = name
    self.last_modified_utc = last_modified_utc
    self.size = size


class CloudCredentials:
  def __init__(self):
    pass


class CloudService:
  def __init__(self, cloud_type: CloudType, credentials: CloudCredentials, verbose: bool = False):
    self.cloud_type = cloud_type
    self.credentials = credentials
    self.verbose = verbose
    pass


  def list_files(self, remote_path: str) -> list[CloudFile]:
    pass


  def download_file(self, remote_file: CloudFile, local_path: str):
    pass


  def upload_file(self, remote_path: str, local_path: str):
    pass


  def sync_files(self, local_folder: str, remote_folder: str, verbose: bool = False):
    # Build a dictionary of remote files: {relative_path: file}
    remote_files = {}
    if verbose:
      print("Querying remote folder...")
    for file in self.list_files(remote_folder):
      remote_files[file.name] = file

    # Build a dictionary of local files relative to the local folder.
    local_files = {}
    for root, dirs, files in os.walk(local_folder):
      for file in files:
        # Compute the relative path with respect to the given local folder.
        rel_path = os.path.relpath(os.path.join(root, file), local_folder)
        local_files[rel_path] = os.path.join(root, file)

    # Process files that exist remotely:
    for rel_path, file in remote_files.items():
      local_file_path = os.path.join(local_folder, rel_path)

      if not os.path.exists(local_file_path):
        # File exists in remote only: download it
        if verbose:
          print(f"Local file missing for remote file '{rel_path}'. Downloading...")
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        self.download_file(file, local_file_path)
      else:
        # Both sides exist: compare file size and last modified timestamp.
        local_size = os.path.getsize(local_file_path)
        remote_size = file.size

        local_mod_time_utc = _get_local_file_mod_time_utc(local_file_path)
        remote_mod_time_utc = file.last_modified_utc

        if verbose:
          print(f"\nConflict for '{rel_path}':")
          print(f"-->Local  - size: {local_size:12.0f} bytes, modified: {local_mod_time_utc}")
          print(f"-->Remote - size: {remote_size:12.0f} bytes, modified: {local_mod_time_utc}")

        TIME_TOLERANCE =  timedelta(seconds=10)

        # If both the size and modification time are nearly identical, assume they are in sync.
        if (local_size == remote_size and
            abs(remote_mod_time_utc - local_mod_time_utc) <= TIME_TOLERANCE):
          if verbose:
            print("  Files are in sync. No action needed.")
          continue

        # Decide which version is more current.
        if remote_mod_time_utc > local_mod_time_utc + TIME_TOLERANCE:
          if verbose:
            print("  Remote file is newer. Downloading remote version...")
          self.download_file(file, local_file_path)
        elif local_mod_time_utc > remote_mod_time_utc + TIME_TOLERANCE:
          if local_size != remote_size:
            if verbose:
              print("  Local file is newer. Uploading local version...")
            self.upload_file(file, local_file_path)
          else:
            if verbose:
              print("  No action needed.")
        else:
          # If the time difference is within the tolerance but sizes differ, I'm not sure how to resolve.
          pass

    # Process files that exist locally but not remotely.
    for rel_path, local_file_path in local_files.items():
      if rel_path not in remote_files:
        # File exists in local only: upload it.
        if verbose:
          print(f"Remote file missing for local file '{rel_path}'. Uploading...")

        self.upload_file(rel_path, local_file_path)


def _get_local_file_mod_time_utc(file_path):
  """Return the local file's last modified time as a UTC datetime."""
  timestamp = os.path.getmtime(file_path)
  return datetime.fromtimestamp(timestamp, tz=timezone.utc)