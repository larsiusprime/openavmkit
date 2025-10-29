import os
import json
from openavmkit.cloud.azure import init_service_azure, get_creds_from_env_azure

from openavmkit.cloud.base import CloudService, CloudType, CloudAccess, CloudCredentials
from openavmkit.cloud.huggingface import (
    get_creds_from_env_huggingface,
    init_service_huggingface,
)
from openavmkit.cloud.sftp import get_creds_from_env_sftp, init_service_sftp


def load_cloud_settings(
    filepath: str = "cloud.json"
):
    """
    Load cloud settings file from disk
    
    Returns
    -------
    dict
        The cloud settings object
    """
    
    cloud : dict | None = None

    try:
        with open(filepath, "r") as f:
            cloud = json.load(f)
    except FileNotFoundError:
        cwd = os.getcwd()
        full_path = os.path.join(cwd, filepath)
        exists = os.path.exists(full_path)
        msg = f"Could not find cloud settings file: {filepath}. Go to '{cwd}' and create a cloud.json file there! {full_path} exists? {exists}"
        raise FileNotFoundError(msg)

    return cloud


def init(
    verbose: bool, 
    cloud_settings: dict
) -> CloudService | None:
    """Creates a CloudService object based on user settings.

    Attributes
    ----------
    verbose : bool
        Whether to print verbose output.
    cloud_settings : dict
        Cloud settings dictionary

    Returns
    -------
    Initialized CloudService object
    """
        
    enabled = cloud_settings.get("enabled", True)
    if not enabled:
        print("Cloud service disabled, skipping...")
        return None

    if cloud_settings is None:
        print("No cloud settings found, cannot initialize cloud...")
        return None

    cloud_type = cloud_settings.get("type", None)
    if cloud_type is None:
        print("Missing 'type' in cloud settings, cannot initialize cloud...")
        return
    
    cloud_type = cloud_type.lower()
    cloud_access = _get_cloud_access(cloud_type)
    if cloud_access is not None:
        cloud_access = cloud_access.lower()

    if cloud_settings is not None:
        illegal_values = [
            "hf_token",
            "azure_storage_connection_string",
            "sftp_password",
            "sftp_username",
        ]
        for key in illegal_values:
            if key.lower() in cloud_settings:
                raise ValueError(
                    f"Sensitive credentials '{key}' should never be stored in your settings file! They should ONLY be in your local .env file!"
                )
        warn_values = [
            "azure_access"
            "hf_access",
            "sftp_access",
        ]
        for key in warn_values:
            if key.lower() in cloud_settings:
                warnings.warn(f"Field '{key}' should be stored in your .env file, not in your cloud.json. The version in cloud.json will be ignored.")
    
    anonymous_access_allowed = False
    
    if cloud_type == "azure":
        anonymous_access_allowed = True
        # Add other conditions for anonymous access being allowed here
    
    if anonymous_access_allowed:
        if cloud_access is None:
            cloud_access = "read_only"
            if verbose:
                print(f"Cloud access was None, defaulting to 'read_only'...")
        elif cloud_access == "read_write":
            anonymous_access_allowed = False
    
    if verbose:
        print(
            f"Initializing cloud service of type '{cloud_type}' with access '{cloud_access}'..."
        )
    if cloud_access is None:
        cloud_access_key = _get_cloud_access_key(cloud_type)
        raise ValueError(
            f"Missing '{cloud_access_key}' in environment. Have you created your .env file and properly filled it out?"
        )
    
    if anonymous_access_allowed:
        try:
            credentials = _get_creds_from_env(cloud_type)
        except ValueError as e:
            credentials = None
    else:
        credentials = _get_creds_from_env(cloud_type)

    try:
        cloud_service = _init_service(cloud_type, cloud_access, credentials, cloud_settings)
    except ValueError as e:
        print(f"Error initializing cloud! Error = {e}")
        return None
    cloud_service.verbose = verbose
    return cloud_service


#######################################
# PRIVATE
#######################################


def _get_cloud_access_key(cloud_type):
    key = ""
    if cloud_type == "azure":
        key = "AZURE_ACCESS"
    elif cloud_type == "huggingface":
        key = "HF_ACCESS"
    elif cloud_type == "sftp":
        key = "SFTP_ACCESS"
    else:
        raise ValueError(f"Unsupported cloud type: {cloud_type}")
    return key


def _get_cloud_access(cloud_type):
    key = _get_cloud_access_key(cloud_type)
    if key != "":
        value = os.getenv(key)
        return value
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
    cloud_type: CloudType,
    cloud_access: CloudAccess,
    credentials: CloudCredentials,
    cloud_settings: dict
) -> CloudService:
    print(f"_init_service('{cloud_type}', '{cloud_access}')")
    if cloud_type == "azure":
        return init_service_azure(credentials, cloud_access, cloud_settings)
    elif cloud_type == "huggingface":
        return init_service_huggingface(credentials, cloud_access, cloud_settings)
    elif cloud_type == "sftp":
        return init_service_sftp(credentials, cloud_access)
    # Add more cloud types here as needed:
    # elif cloud_type == <SOMETHING ELSE>:
    #   return init_service_something_else(cloud_type, credentials)
    else:
        raise ValueError(f"Unsupported cloud type: {cloud_type}")
