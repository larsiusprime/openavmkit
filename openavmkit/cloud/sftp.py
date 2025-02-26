import os
import time
from datetime import datetime, timezone

import paramiko
from openavmkit.cloud.base import CloudCredentials, CloudService, CloudFile, _fix_path_slashes


class SFTPCredentials(CloudCredentials):
  def __init__(
      self,
      hostname: str,
      port: int,
      username: str,
      password: str = None,
      key_filename: str = None
  ):
    super().__init__()
    self.hostname = hostname
    self.port = port
    self.username = username
    self.password = password
    self.key_filename = key_filename


class SFTPService(CloudService):
  def __init__(self, credentials: SFTPCredentials, base_path: str = ".", timeout: int = 30, retries: int = 3):
    super().__init__("sftp", credentials)
    self.base_path = base_path
    self.ssh_client = paramiko.SSHClient()
    self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect with retry logic
    for attempt in range(retries):
      try:
        if credentials.key_filename:
          self.ssh_client.connect(
            hostname=credentials.hostname,
            port=credentials.port,
            username=credentials.username,
            key_filename=credentials.key_filename,
            timeout=timeout
          )
        else:
          self.ssh_client.connect(
            hostname=credentials.hostname,
            port=credentials.port,
            username=credentials.username,
            password=credentials.password,
            timeout=timeout
          )
        self.sftp = self.ssh_client.open_sftp()
        break
      except Exception as e:
        if attempt == retries - 1:
          raise ConnectionError(f"Failed to connect to SFTP server after {retries} attempts: {str(e)}")
        if self.verbose:
          print(f"Connection attempt {attempt+1} failed, retrying... ({str(e)})")
        time.sleep(2)  # Wait before retry

  def list_files(self, remote_path: str) -> list[CloudFile]:
    full_remote_path = _fix_path_slashes(os.path.join(self.base_path, remote_path))
    files = []
    if self.verbose:
      print(f"Querying remote folder: {full_remote_path}")
    try:
      for attr in self.sftp.listdir_attr(full_remote_path):
        file_path = os.path.join(remote_path, attr.filename)
        last_modified_utc = datetime.fromtimestamp(attr.st_mtime, tz=timezone.utc)
        files.append(CloudFile(
          name=file_path,
          last_modified_utc=last_modified_utc,
          size=attr.st_size
        ))
    except IOError:
      try:
        # Not a directory, assume single file:
        attr = self.sftp.stat(full_remote_path)
        last_modified = datetime.fromtimestamp(attr.st_mtime, tz=timezone.utc)
        files.append(CloudFile(
          name=remote_path,
          last_modified_utc=last_modified,
          size=attr.st_size
        ))
      except IOError:
        if self.verbose:
          print(f"Remote path not found: {full_remote_path}")
    return files

  def download_file(self, remote_file: CloudFile, local_file_path: str):
    super().download_file(remote_file, local_file_path)
    remote_path = _fix_path_slashes(os.path.join(self.base_path, remote_file.name))

    # Create local directory if it doesn't exist
    local_dir = os.path.dirname(local_file_path)
    if local_dir and not os.path.exists(local_dir):
      os.makedirs(local_dir)

    if self.verbose:
      print(f"Downloading {remote_path} to {local_file_path}...")

    try:
      self.sftp.get(remote_path, local_file_path)

      # Verify download
      if os.path.exists(local_file_path):
        local_size = os.path.getsize(local_file_path)
        remote_size = remote_file.size
        if local_size != remote_size:
          raise IOError(f"File size mismatch: local={local_size}, remote={remote_size}")
        if self.verbose:
          print(f"Download complete: {local_file_path} ({local_size} bytes)")
      else:
        raise IOError(f"Downloaded file not found: {local_file_path}")
    except Exception as e:
      raise IOError(f"Download failed: {str(e)}")

  def upload_file(self, remote_file_path: str, local_file_path: str):
    super().upload_file(remote_file_path, local_file_path)

    if not os.path.exists(local_file_path):
      raise FileNotFoundError(f"Local file not found: {local_file_path}")

    full_remote_path = _fix_path_slashes(os.path.join(self.base_path, remote_file_path))
    remote_dir = _fix_path_slashes(os.path.dirname(full_remote_path))

    # Create remote directory structure
    self._mkdir_p(remote_dir)

    # Get local file size for verification
    local_size = os.path.getsize(local_file_path)

    if self.verbose:
      print(f"Preparing to upload {local_file_path} ({local_size} bytes) to {full_remote_path}...")

    # Use explicit file handling for better control
    try:
      with open(local_file_path, 'rb') as local_file:
        with self.sftp.file(full_remote_path, 'wb') as remote_file:
          # Read and write in chunks for large files
          chunk_size = 32768  # 32KB chunks
          data = local_file.read(chunk_size)
          total_written = 0

          while data:
            remote_file.write(data)
            total_written += len(data)
            data = local_file.read(chunk_size)

          # Ensure data is written to disk
          remote_file.flush()

      if self.verbose:
        print(f"Upload complete: {full_remote_path} ({total_written} bytes)")

      # Verify upload
      try:
        stat = self.sftp.stat(full_remote_path)
        if stat.st_size != local_size:
          print(f"Warning: File size mismatch after upload: local={local_size}, remote={stat.st_size}")
        elif self.verbose:
          print(f"Upload verified: {full_remote_path}")
      except IOError as e:
        print(f"Error verifying upload: {str(e)}")

    except Exception as e:
      raise IOError(f"Upload failed: {str(e)}")

  def _mkdir_p(self, remote_directory: str):
    # Recursively create remote directories if they do not exist
    if remote_directory == '/' or remote_directory == '':
      return

    dirs = remote_directory.split("/")
    path = ""

    for d in dirs:
      if d.strip() == "":
        continue

      if path != "":
        path += "/" + d
      else:
        path = d

      try:
        self.sftp.stat(path)
      except IOError:
        try:
          self.sftp.mkdir(path)
          if self.verbose:
            print(f"Created remote directory: {path}")
        except IOError as e:
          # Handle race condition if directory was created by another process
          try:
            self.sftp.stat(path)
          except IOError:
            raise IOError(f"Failed to create remote directory {path}: {str(e)}")

  def close(self):
    """Close SFTP and SSH connections"""
    try:
      if hasattr(self, 'sftp') and self.sftp:
        self.sftp.close()
      if hasattr(self, 'ssh_client') and self.ssh_client:
        self.ssh_client.close()
    except Exception as e:
      if self.verbose:
        print(f"Error during connection cleanup: {str(e)}")

  def __enter__(self):
    """Support for context manager"""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Clean up resources when used as context manager"""
    self.close()


def init_service_sftp(credentials: SFTPCredentials) -> SFTPService:
  base_path = os.getenv("SFTP_BASE_PATH", "./openavmkit")
  return SFTPService(credentials, base_path)


def get_creds_from_env_sftp() -> SFTPCredentials:
  hostname = os.getenv("SFTP_HOSTNAME")
  port = int(os.getenv("SFTP_PORT", "22"))
  username = os.getenv("SFTP_USERNAME")
  password = os.getenv("SFTP_PASSWORD")
  key_filename = os.getenv("SFTP_KEY_FILENAME")

  if not hostname or not username:
    raise ValueError("Missing 'SFTP_HOSTNAME' or 'SFTP_USERNAME' in environment.")

  return SFTPCredentials(hostname, port, username, password, key_filename)
