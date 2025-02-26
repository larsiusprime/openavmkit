import os
from openavmkit.cloud.azure import init_service_azure, get_creds_from_env_azure
from dotenv import load_dotenv

from openavmkit.cloud.base import CloudService, CloudType, CloudCredentials
from openavmkit.cloud.huggingface import get_creds_from_env_huggingface, init_service_huggingface
from openavmkit.cloud.sftp import get_creds_from_env_sftp, init_service_sftp


def init(verbose: bool) -> CloudService | None:
  load_dotenv()
  cloud_type = os.getenv("CLOUD_TYPE").lower()
  credentials = _get_creds_from_env()
  try:
    cloud_service = _init_service(cloud_type, credentials)
  except ValueError as e:
    return None
  cloud_service.verbose = verbose
  return cloud_service


def _get_creds_from_env() -> CloudCredentials:
  cloud_type = os.getenv("CLOUD_TYPE").lower()
  if cloud_type == "azure":
    return get_creds_from_env_azure()
  elif cloud_type == "huggingface":
    return get_creds_from_env_huggingface()
  elif cloud_type == "sftp":
    return get_creds_from_env_sftp()
  # Add more cloud types here as needed:
  # elif cloud_type == <SOMETHING ELSE>:
  #   return get_creds_from_something_else():
  else:
    raise ValueError(f"Unsupported cloud type: {cloud_type}")


def _init_service(cloud_type: CloudType, credentials: CloudCredentials) -> CloudService:
  if cloud_type == "azure":
    return init_service_azure(credentials)
  elif cloud_type == "huggingface":
    return init_service_huggingface(credentials)
  elif cloud_type == "sftp":
    return init_service_sftp(credentials)
  # Add more cloud types here as needed:
  # elif cloud_type == <SOMETHING ELSE>:
  #   return init_service_something_else(cloud_type, credentials)
  else:
    raise ValueError(f"Unsupported cloud type: {cloud_type}")
