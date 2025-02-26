# This file lets us abstract remote storage services (Azure, AWS, etc.) into a common interface
import os
from datetime import datetime, timedelta, timezone
from typing import Literal

CloudType = Literal[
  "azure",
  "huggingface",
  "sftp"
  # In the future we'll add more types here:
  # "aws",
  # "gcp"
]

class CloudFile:
  def __init__(self, name: str, last_modified_utc: datetime, size: int):
    self.name = _fix_path_slashes(name)
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


  def download_file(self, remote_file: CloudFile, local_file_path: str):
    r = os.path.basename(remote_file.name)
    l = os.path.basename(local_file_path)
    if r != l:
      raise ValueError(f"Remote path '{r}' does not match local path '{l}'.")

  def upload_file(self, remote_file_path: str, local_file_path: str):
    r = os.path.basename(remote_file_path)
    l = os.path.basename(local_file_path)
    if r != l:
      raise ValueError(f"Remote path '{r}' does not match local path '{l}'.")


  def sync_files(self, locality: str, local_folder: str, remote_folder: str, dry_run: bool = False, verbose: bool = False):

    # Build a dictionary of remote files: {relative_path: file}
    remote_files = {}
    if verbose:
      print(f"Syncing files from local=\"{local_folder}\" to remote=\"{remote_folder}\"...")
    for file in self.list_files(remote_folder):
      remote_files[_fix_path_slashes(file.name)] = file

    # Build a dictionary of local files relative to the local folder.
    local_files = []
    remote_file_map = {}
    for root, dirs, files in os.walk(local_folder):
      for file in files:
        # Compute the relative path with respect to the given local folder.
        rel_path = _fix_path_slashes(os.path.relpath(os.path.join(root, file), local_folder))
        loc_bits = locality.split("-")
        loc_path = os.path.join("", *loc_bits)
        remote_file_path = os.path.join(loc_path, rel_path)
        local_file_path = os.path.join(root, file)
        remote_file_path = _fix_path_slashes(remote_file_path)
        local_file_path = _fix_path_slashes(local_file_path)
        entry = {
          "remote": remote_file_path,
          "local": local_file_path
        }
        local_files.append(entry)
        remote_file_map[remote_file_path] = entry
      for dir in dirs:
        rel_path = _fix_path_slashes(os.path.relpath(os.path.join(root, dir), local_folder))
        loc_bits = locality.split("-")
        loc_path = os.path.join("", *loc_bits)
        remote_file_path = os.path.join(loc_path, rel_path)
        remote_file_path = _fix_path_slashes(remote_file_path)
        entry = {
          "remote": remote_file_path,
          "local": os.path.join(root, dir)
        }
        local_files.append(entry)
        remote_file_map[remote_file_path] = entry

    for key in remote_file_map:
      print(key)

    # Process files that exist remotely:
    for rel_path, file in remote_files.items():

      local_file_exists = False
      local_file_path = None

      if rel_path in remote_file_map:
        local_file_path = remote_file_map[rel_path]["local"]
        local_file_exists = os.path.exists(local_file_path)
      else:
        # remove "remote_folder" from the beginning of the path
        local_file_path = rel_path[len(remote_folder)+1:]
        local_file_path = _fix_path_slashes(os.path.join(local_folder, local_file_path))

      if not local_file_exists:
        if local_file_path is not None:
          # Create the local directory if it doesn't exist.
          if not os.path.exists(os.path.dirname(local_file_path)):
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
          # If the file IS just a directory, then we're done.
          if os.path.isdir(local_file_path):
            continue
        # File exists in remote only: download it
        if verbose:
          print(f"Local file '{local_file_path}' missing for remote file '{rel_path}'. Downloading...")
        _print_download(file.name, local_file_path)
        if not dry_run:
          self.download_file(file, local_file_path)
      else:
        if os.path.isdir(local_file_path):
          continue

        # Both sides exist: compare file size and last modified timestamp.
        local_size = os.path.getsize(local_file_path)
        remote_size = file.size

        local_mod_time_utc = _get_local_file_mod_time_utc(local_file_path)
        remote_mod_time_utc = file.last_modified_utc

        TIME_TOLERANCE = timedelta(days=1)
        size_delta = abs(local_size - remote_size)
        time_delta = abs(remote_mod_time_utc - local_mod_time_utc)

        if verbose:
          print(f"\nConflict for '{rel_path}':")
          print(f"-->Local  - size: {local_size:10,.0f} bytes, modified: {local_mod_time_utc}")
          print(f"-->Remote - size: {remote_size:10,.0f} bytes, modified: {remote_mod_time_utc}")
          print(f"-->Size delta: {size_delta:10,.0f} bytes")
          print(f"-->Time delta: {time_delta}")

        # If both the size and modification time are nearly identical, assume they are in sync.
        if (size_delta == 0 and time_delta <= TIME_TOLERANCE):
          if verbose:
            print("  Files are in sync. No action needed.")
          continue

        # Decide which version is more current.
        if remote_mod_time_utc > local_mod_time_utc:
          if verbose:
            print("  Remote file is newer. Downloading remote version...")
          _print_download(file.name, local_file_path)
          if not dry_run:
            self.download_file(file, local_file_path)
        elif local_mod_time_utc > remote_mod_time_utc:
          if verbose:
            print("  Local file is newer. Uploading local version...")
          _print_upload(file.name, local_file_path)
          if not dry_run:
            self.upload_file(file.name, local_file_path)

    # Process files that exist locally but not remotely.
    for entry in local_files:
      remote_path = entry["remote"]
      local_file_path = entry["local"]
      if remote_path not in remote_files:
        # File exists in local only: upload it.
        if verbose:
          print(f"Remote file missing for local file '{local_file_path}'. Uploading...")
        _print_upload(remote_path, local_file_path)
        if not dry_run:
          self.upload_file(remote_path, local_file_path)


def _print_download(remote_file: str, local_file: str):
  print(f"Downloading '{local_file}' <-- '{remote_file}'...")

def _print_upload(remote_file: str, local_file: str):
  print(f"Uploading '{local_file}' --> '{remote_file}'...")

def _fix_path_slashes(path: str):
  return path.replace("\\", "/")


def _get_local_file_mod_time_utc(file_path):
  """Return the local file's last modified time as a UTC datetime."""
  timestamp = os.path.getmtime(file_path)
  return datetime.fromtimestamp(timestamp, tz=timezone.utc)