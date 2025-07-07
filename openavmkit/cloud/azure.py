import os

from openavmkit.cloud.base import (
    CloudCredentials,
    CloudService,
    CloudFile,
    CloudAccess,
)
from azure.storage.blob import BlobServiceClient


class AzureCredentials(CloudCredentials):
    """Authentication credentials for Azure"""

    def __init__(self, connection_string: str):
        """Initialize AzureCredentials object

        Parameters
        ----------
        connection_string : str
            Your Azure connection string
        """
        super().__init__()
        self.connection_string = connection_string


class AzureService(CloudService):
    """Azure-specific CloudService object.

    Attributes
    ----------
    connection_string : str
        Your Azure connection string
    blob_service_client : BlobServiceClient
        Azure Blob Service Client
    container_client : ContainerClient
        Azure Container Client
    """

    def __init__(
        self, credentials: AzureCredentials, container_name: str, access: CloudAccess
    ):
        """Initialize AzureService object

        Attributes
        ----------
        credentials : AzureCredentials
            Authentication credentials for Azure
        container_name : str
            The name of your Azure container
        access : CloudAccess
            What kind of access/permission ("read_only", "read_write")
        """
        super().__init__("azure", credentials, access)
        self.connection_string = credentials.connection_string
        self.blob_service_client = BlobServiceClient.from_connection_string(
            credentials.connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            container_name
        )

    def list_files(self, remote_path: str) -> list[CloudFile]:
        """List all the files at the given path on Azure

        Parameters
        ----------
        remote_path : str
            Path on Azure you want to query

        Returns
        -------
        list[CloudFile]
            A listing of all the files contained within the queried path on the remote Azure service
        """
        blob_list = self.container_client.list_blobs(name_starts_with=remote_path)
        return [
            CloudFile(
                name=blob.name, last_modified_utc=blob.last_modified, size=blob.size
            )
            for blob in blob_list
        ]

    def download_file(self, remote_file: CloudFile, local_file_path: str):
        """Download a remote file from the Azure service

        Parameters
        ----------
        remote_file : CloudFile
            The file to download
        local_file_path : str
            The path on your local computer you want to save the remote file to
        """
        super().download_file(remote_file, local_file_path)
        blob_client = self.container_client.get_blob_client(remote_file.name)
        with open(local_file_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())

    def upload_file(self, remote_file_path: str, local_file_path: str):
        """Upload a local file to the Azure service

        Parameters
        ----------
        remote_file_path : str
            The remote path on the Azure service you want to upload your local file to
        local_file_path : str
            The local path to the file on your local computer that you want to upload
        """
        super().upload_file(remote_file_path, local_file_path)
        blob_client = self.container_client.get_blob_client(remote_file_path)
        with open(local_file_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)


def init_service_azure(
    credentials: AzureCredentials, access: CloudAccess
) -> AzureService:
    """Initializes the Azure service

    Parameters
    ----------
    credentials : AzureCredentials
        The credentials to your Azure account
    access : CloudAccess
        What kind of access/permission ("read_only", "read_write")
    """
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    if container_name is None:
        raise ValueError("Missing 'AZURE_STORAGE_CONTAINER_NAME' in environment.")
    if isinstance(credentials, AzureCredentials):
        service = AzureService(credentials, container_name, access)
    else:
        raise ValueError("Invalid credentials for Azure service.")
    return service


def get_creds_from_env_azure() -> AzureCredentials:
    """Reads and returns Azure credentials from the environment settings

    Returns
    -------
    AzureCredentials
        The credentials for Azure stored in environment settings
    """
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("Missing Azure connection string in environment.")
    return AzureCredentials(connection_string)
