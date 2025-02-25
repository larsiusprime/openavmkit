import os
from openavmkit.cloud.azure import init_service_azure, get_creds_from_env_azure
from dotenv import load_dotenv

from openavmkit.cloud.base import CloudService, CloudType, CloudCredentials


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
  # Add more cloud types here as needed:
  # elif cloud_type == <SOMETHING ELSE>:
  #   return get_creds_from_something_else():
  else:
    raise ValueError(f"Unsupported cloud type: {cloud_type}")


def _init_service(cloud_type: CloudType, credentials: CloudCredentials) -> CloudService:
  if cloud_type == "azure":
    return init_service_azure(credentials)
  # Add more cloud types here as needed:
  # elif cloud_type == <SOMETHING ELSE>:
  #   return init_service_something_else(cloud_type, credentials)
  else:
    raise ValueError(f"Unsupported cloud type: {cloud_type}")
