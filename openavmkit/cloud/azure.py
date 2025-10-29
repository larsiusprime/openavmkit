import os

from openavmkit.cloud.base import (
    CloudCredentials,
    CloudService,
    CloudFile,
    CloudAccess,
)
from azure.storage.blob import BlobServiceClient, ContainerClient


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
        self, 
        credentials: AzureCredentials, 
        container_name: str, 
        access: CloudAccess
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
        if credentials is not None:
            self.connection_string = credentials.connection_string
            self.blob_service_client = BlobServiceClient.from_connection_string(
                credentials.connection_string
            )
            self.container_client = self.blob_service_client.get_container_client(
                container_name
            )
        else:
            self.connection_string = None
            self.blob_service_client = None
            self.container_client = None

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


class AzureAnonymousService(AzureService):
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
        self, 
        container_url: str,
        access: CloudAccess
    ):
        """Initialize AzureService object

        Attributes
        ----------
        container_url : str
            The url of a publicly accessible Azure container
        access : CloudAccess
            What kind of access/permission ("read_only", "read_write")
        """
        
        super().__init__(credentials=None, container_name=None, access=access)
        
        if access != "read_only":
            raise ValueError("AzureAnonymousService only supports 'read_only' access")
        
        self.connection_string = None
        self.blob_service_client = None
        
        self.container_url = container_url.rstrip("/")
        self.container_client = ContainerClient.from_container_url(
            self.container_url,
            credential=None
        )

    def upload_file(self, remote_file_path: str, local_file_path: str):
        """Upload a local file to the Azure service

        Parameters
        ----------
        remote_file_path : str
            The remote path on the Azure service you want to upload your local file to
        local_file_path : str
            The local path to the file on your local computer that you want to upload
        """
        
        # No uploading in anonymous mode!
        
        return



def init_service_azure(
    credentials: AzureCredentials, 
    access: CloudAccess,
    cloud_settings: dict
) -> AzureService:
    """Initializes the Azure service

    Parameters
    ----------
    credentials : AzureCredentials
        The credentials to your Azure account
    access : CloudAccess
        What kind of access/permission ("read_only", "read_write")
    cloud_settings : dict
        Local project settings for cloud storage
    """
    
    container_url = cloud_settings.get("azure_storage_container_url")
    container_name = cloud_settings.get("azure_storage_container_name")
    
    if credentials is None and access == "read_only":
        # Anonymous mode
        if container_url is None:
            if (container_name is not None):
                raise ValueError("Missing 'azure_storage_container_url' in cloud settings. You have 'azure_storage_container_name', but your access is 'read_only'; you need _url for that, not _name")
            raise ValueError("Missing 'azure_storage_container_url' in cloud settings.")
        service = AzureAnonymousService(container_url, access)
    else:
        # Authenticated mode
        if isinstance(credentials, AzureCredentials):
            container_name = cloud_settings.get("azure_storage_container_name")
            if not container_name:
                if (container_url is not None):
                    raise ValueError("Missing 'azure_storage_container_name' in cloud settings. You have 'azure_storage_container_url', but your access is 'read_write'; you need _name for that, not _url")
                raise ValueError("Missing 'azure_storage_container_name' in cloud settings.")
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
