import os

from openavmkit.cloud.base import CloudCredentials, CloudService, CloudType, CloudFile
from azure.storage.blob import BlobServiceClient

class AzureCredentials(CloudCredentials):

  def __init__(self, connection_string: str):
    super().__init__()
    self.connection_string = connection_string


class AzureService(CloudService):

  def __init__(self, credentials: AzureCredentials, container_name: str):
    super().__init__("azure", credentials)
    self.connection_string = credentials.connection_string
    self.blob_service_client = BlobServiceClient.from_connection_string(credentials.connection_string)
    self.container_client = self.blob_service_client.get_container_client(container_name)


  def list_files(self, remote_path: str) -> list[CloudFile]:
    blob_list = self.container_client.list_blobs(name_starts_with=remote_path)
    return [
      CloudFile(
        name=blob.name,
        last_modified_utc=blob.last_modified,
        size=blob.size
      ) for blob in blob_list
    ]


  def download_file(self, remote_file: CloudFile, local_file_path: str):
    super().download_file(remote_file, local_file_path)
    blob_client = self.container_client.get_blob_client(remote_file.name)
    with open(local_file_path, "wb") as f:
      download_stream = blob_client.download_blob()
      f.write(download_stream.readall())


  def upload_file(self, remote_file_path: str, local_file_path: str):
    super().upload_file(remote_file_path, local_file_path)
    blob_client = self.container_client.get_blob_client(remote_file_path)
    with open(local_file_path, "rb") as f:
      blob_client.upload_blob(f, overwrite=True)


def init_service_azure(credentials: AzureCredentials) -> AzureService:
  container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
  if container_name is None:
    raise ValueError("Missing 'AZURE_STORAGE_CONTAINER_NAME' in environment.")
  if isinstance(credentials, AzureCredentials):
    return AzureService(credentials, container_name)
  else:
    raise ValueError("Invalid credentials for Azure service.")


def get_creds_from_env_azure() -> AzureCredentials:
  connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
  if not connection_string:
    raise ValueError("Missing Azure connection string in environment.")
  return AzureCredentials(connection_string)