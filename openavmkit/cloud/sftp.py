import os
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
  def __init__(self, credentials: SFTPCredentials, base_path: str = "."):
    super().__init__("sftp", credentials)
    self.base_path = base_path
    self.ssh_client = paramiko.SSHClient()
    self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if credentials.key_filename:
      self.ssh_client.connect(
        hostname=credentials.hostname,
        port=credentials.port,
        username=credentials.username,
        key_filename=credentials.key_filename
      )
    else:
      self.ssh_client.connect(
        hostname=credentials.hostname,
        port=credentials.port,
        username=credentials.username,
        password=credentials.password
      )
    self.sftp = self.ssh_client.open_sftp()


  def list_files(self, remote_path: str) -> list[CloudFile]:
    full_remote_path = os.path.join(self.base_path, remote_path)
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
    remote_path = os.path.join(self.base_path, remote_file.name)
    self.sftp.get(remote_path, local_file_path)


  def upload_file(self, remote_file_path: str, local_file_path: str):
    super().upload_file(remote_file_path, local_file_path)
    full_remote_path = _fix_path_slashes(os.path.join(self.base_path, remote_file_path))
    remote_dir = _fix_path_slashes(os.path.dirname(full_remote_path))
    self._mkdir_p(remote_dir)
    if self.verbose:
      print(f"Uploading file {local_file_path} to {full_remote_path}...")
    self.sftp.put(local_file_path, full_remote_path)


  def _mkdir_p(self, remote_directory: str):
    # Recursively create remote directories if they do not exist
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
        self.sftp.mkdir(path)

  def close(self):
    self.sftp.close()
    self.ssh_client.close()


def init_service_sftp(credentials: SFTPCredentials) -> SFTPService:
  base_path = os.getenv("SFTP_BASE_PATH", "openavmkit")
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