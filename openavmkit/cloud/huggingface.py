import os
from datetime import datetime, timezone
import requests
from huggingface_hub import hf_hub_url, upload_file as hf_upload_file
from huggingface_hub.hf_api import HfApi, RepoFolder
from huggingface_hub.errors import EntryNotFoundError
from openavmkit.cloud.base import CloudCredentials, CloudService, CloudFile, CloudAccess


class HuggingFaceCredentials(CloudCredentials):
    """Authentication credentials for HuggingFace"""

    def __init__(self, token: str):
        """
        Initialize credentials for HuggingFace

        Parameters
        ----------
        token : str
            Your HuggingFace token
        """
        super().__init__()
        self.token = token


class HuggingFaceService(CloudService):
    """HuggingFace-specific CloudService object.

    Attributes
    ----------
    repo_id : str
        Repository identifier
    revision : str
        Revision identifier
    token : str
        Access token
    api : huggingface_hub.hf_api.HfApi
        HuggingFace API object
    """

    def __init__(
        self,
        credentials: HuggingFaceCredentials,
        repo_id: str,
        access: CloudAccess,
        revision: str = "main",
    ):
        """Initialize HuggingFaceService Object

        Attributes
        ----------
        credentials : HuggingFaceCredentials
            Authentication credentials for HuggingFace
        repo_id : str
            Repository identifier
        access : CloudAccess
            What kind of access/permission ("read_only", "read_write")
        revision : str
            Revision identifier

        """
        super().__init__("huggingface", credentials, access)
        self.repo_id = repo_id
        self.revision = revision
        self.token = credentials.token
        self.api = HfApi()

    def list_files(self, remote_path: str) -> list[CloudFile]:
        """List all the files at the given path on HuggingFace

        Parameters
        ----------
        remote_path : str
            Path on HuggingFace you want to query

        Returns
        -------
        list[CloudFile]
            A listing of all the files contained within the queried path on the remote HuggingFace service
        """
        infos = self.api.list_repo_tree(
            repo_id=self.repo_id,
            revision=self.revision,
            token=self.token,
            path_in_repo=remote_path,
            repo_type="dataset",
            recursive=True,
            expand=True,
        )

        remote_empty = False
        files = []

        try:
            for info in infos:
                print(info)
                break
        except EntryNotFoundError:
            remote_empty = True

        if not remote_empty:
            for info in infos:

                if isinstance(info, RepoFolder):
                    continue

                if info.rfilename.startswith(remote_path):
                    last_modified_date: datetime = info.last_commit.date
                    last_modified_utc = last_modified_date.astimezone(timezone.utc)
                    files.append(
                        CloudFile(
                            name=info.rfilename,
                            last_modified_utc=last_modified_utc,
                            size=info.size,
                        )
                    )

        return files

    def download_file(self, remote_file: CloudFile, local_file_path: str):
        """Download a remote file from the HuggingFace service

        Parameters
        ----------
        remote_file : CloudFile
            The file to download
        local_file_path : str
            The path on your local computer you want to save the remote file to
        """
        super().download_file(remote_file, local_file_path)
        url = hf_hub_url(
            repo_id=self.repo_id,
            filename=remote_file.name,
            repo_type="dataset",
            revision=self.revision,
        )
        headers = {"authorization": f"Bearer {self.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        with open(local_file_path, "wb") as f:
            f.write(response.content)

    def upload_file(self, remote_file_path: str, local_file_path: str):
        """Upload a local file to the HuggingFace service

        Parameters
        ----------
        remote_file_path : str
            The remote path on the HuggingFace service you want to upload your local file to
        local_file_path : str
            The local path to the file on your local computer that you want to upload
        """
        super().upload_file(remote_file_path, local_file_path)
        hf_upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=remote_file_path,
            repo_id=self.repo_id,
            token=self.token,
            repo_type="dataset",
            commit_message="Upload via OpenAVMKit",
        )


def init_service_huggingface(
    credentials: HuggingFaceCredentials, 
    access: CloudAccess,
    cloud_settings: dict
):
    """
    Initializes the HuggingFace service

    Parameters
    ----------
    credentials : HuggingFaceCredentials
        The credentials to your HuggingFace account
    access : CloudAccess
        What kind of access/permission ("read_only", "read_write")
    cloud_settings : dict
        Local project settings for cloud storage
    """
    repo_id = cloud_settings.get("hf_repo_id")
    
    if repo_id is None:
        raise ValueError("Missing 'hf_repo_id' in cloud settings")
    
    revision = cloud_settings.get("hf_revision", "main")
    
    service = HuggingFaceService(credentials, repo_id, access, revision)
    return service


def get_creds_from_env_huggingface() -> HuggingFaceCredentials:
    """Reads and returns HuggingFace credentials from the environment settings

    Returns
    -------
    HuggingFaceCredentials
        The credentials for HuggingFace stored in environment settings
    """
    token = os.getenv("HF_TOKEN")
    return HuggingFaceCredentials(token)
