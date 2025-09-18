import os
from openavmkit.cloud.azure import init_service_azure, get_creds_from_env_azure
from dotenv import load_dotenv

from openavmkit.cloud.base import CloudService, CloudType, CloudAccess, CloudCredentials
from openavmkit.cloud.huggingface import (
    get_creds_from_env_huggingface,
    init_service_huggingface,
)
from openavmkit.cloud.sftp import get_creds_from_env_sftp, init_service_sftp


def init(
    verbose: bool, env_path: str = "", settings: dict = None
) -> CloudService | None:
    """Creates a CloudService object based on user settings.

    Attributes
    ----------
    verbose : bool
        Whether to print verbose output.
    env_path : str
        Path to ".env" file (which stores credentials). Default is ""
    settings : dict
        Settings dictionary (which should contain cloud settings)

    Returns
    -------
    Initialized CloudService object
    """
    load_dotenv(dotenv_path=env_path)

    if settings is not None:
        s_cloud = settings.get("cloud", {})
    else:
        s_cloud = {}
    
    enabled = s_cloud.get("enabled", True)
    if not enabled:
        print("Cloud service disabled, skipping...")
        return None

    cloud_type = os.getenv("CLOUD_TYPE")
    cloud_type = s_cloud.get("type", cloud_type)
    if cloud_type is not None:
        cloud_type = cloud_type.lower()

    cloud_access = _get_cloud_access(cloud_type)

    if s_cloud is not None:
        cloud_access = s_cloud.get("access", cloud_access)
        illegal_values = [
            "hf_token",
            "azure_storage_connection_string",
            "sftp_password",
            "sftp_username",
        ]
        for key in illegal_values:
            if key.lower() in s_cloud:
                raise ValueError(
                    f"Sensitive credentials '{key}' should never be stored in your settings file! They should ONLY be in your local .env file!"
                )

    if verbose:
        print(
            f"Initializing cloud service of type '{cloud_type}' with access '{cloud_access}'..."
        )
    if cloud_type is None:
        raise ValueError(
            "Missing 'CLOUD_TYPE' in environment. Have you created your .env file and properly filled it out?"
        )
    if cloud_access is None:
        raise ValueError(
            "Missing 'CLOUD_ACCESS' in environment. Have you created your .env file and properly filled it out?"
        )
    cloud_type = cloud_type.lower()
    cloud_access = cloud_access.lower()
    credentials = _get_creds_from_env(cloud_type)

    try:
        cloud_service = _init_service(cloud_type, cloud_access, credentials)
    except ValueError as e:
        return None
    cloud_service.verbose = verbose
    return cloud_service


#######################################
# PRIVATE
#######################################


def _get_cloud_access(cloud_type):
    key = ""
    if cloud_type == "azure":
        key = "AZURE_ACCESS"
    elif cloud_type == "huggingface":
        key = "HF_ACCESS"
    elif cloud_type == "sftp":
        key = "SFTP_ACCESS"
    if key != "":
        return os.getenv(key)
    else:
        raise ValueError(f"Unsupported cloud type: {cloud_type}")


def _get_creds_from_env(cloud_type: str) -> CloudCredentials:
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


def _init_service(
    cloud_type: CloudType, cloud_access: CloudAccess, credentials: CloudCredentials
) -> CloudService:
    print(f"_init_service('{cloud_type}', '{cloud_access}')")
    if cloud_type == "azure":
        return init_service_azure(credentials, cloud_access)
    elif cloud_type == "huggingface":
        return init_service_huggingface(credentials, cloud_access)
    elif cloud_type == "sftp":
        return init_service_sftp(credentials, cloud_access)
    # Add more cloud types here as needed:
    # elif cloud_type == <SOMETHING ELSE>:
    #   return init_service_something_else(cloud_type, credentials)
    else:
        raise ValueError(f"Unsupported cloud type: {cloud_type}")
