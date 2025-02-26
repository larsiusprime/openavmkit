import os
from datetime import datetime
import requests
from huggingface_hub import hf_hub_url, HfApi, upload_file as hf_upload_file
from openavmkit.cloud.base import CloudCredentials, CloudService, CloudFile

class HuggingFaceCredentials(CloudCredentials):

  def __init__(self, token: str):
    super().__init__()
    self.token = token


class HuggingFaceService(CloudService):

  def __init__(
      self,
      credentials: HuggingFaceCredentials,
      repo_id: str,
      revision: str = "main",
      repo_type: str = "dataset"
  ):
    super().__init__("huggingface", credentials)
    self.repo_id = repo_id
    self.revision = revision
    self.token = credentials.token
    self.repo_type = repo_type
    self.api = HfApi()


  def list_files(self, remote_path: str) -> list[CloudFile]:
    info = self.api.model_info(repo_id=self.repo_id, revision=self.revision, token=self.token)
    files = []
    for sibling in info.siblings:
      if sibling.rfilename.startswith(remote_path):
        last_modified = sibling.lastModified
        if isinstance(last_modified, str):
          last_modified = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
        files.append(CloudFile(
          name=sibling.rfilename,
          last_modified_utc=last_modified,
          size=sibling.size
        ))
    return files


  def download_file(self, remote_file: CloudFile, local_file_path: str):
    super().download_file(remote_file, local_file_path)
    url = hf_hub_url(
      repo_id=self.repo_id,
      filename=remote_file.name,
      repo_type=self.repo_type,
      revision=self.revision,
    )
    response = requests.get(url)
    response.raise_for_status()
    with open(local_file_path, "wb") as f:
      f.write(response.content)


  def upload_file(self, remote_file_path: str, local_file_path: str):
    super().upload_file(remote_file_path, local_file_path)
    hf_upload_file(
      path_or_fileobj=local_file_path,
      path_in_repo=remote_file_path,
      repo_id=self.repo_id,
      token=self.token,
      commit_message="Upload via OpenAVMKit"
    )


def init_service_huggingface(credentials: HuggingFaceCredentials):
  repo_id = os.getenv("HF_REPO_ID")
  if repo_id is None:
    raise ValueError("Missing 'HF_REPO_ID' in environment")
  revision = os.getenv("HF_REVISION", "main")
  repo_type = os.getenv("HF_REPO_TYPE", "dataset")
  return HuggingFaceService(credentials, repo_id, revision, repo_type)


def get_creds_from_env_huggingface() -> HuggingFaceCredentials:
  token = os.getenv("HF_TOKEN")
  if not token:
    raise ValueError("Missing 'HF_TOKEN' in environment")
  return HuggingFaceCredentials(token)