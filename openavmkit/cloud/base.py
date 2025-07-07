# This file lets us abstract remote storage services (Azure, AWS, etc.) into a common interface
import fnmatch
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

CloudType = Literal[
    "azure",
    "huggingface",
    "sftp",
    # In the future we'll add more types here:
    # "aws",
    # "gcp"
]

CloudAccess = Literal["read_only", "read_write"]


class CloudFile:
    """Base class that represents a file stored on a remote cloud storage service"""

    def __init__(self, name: str, last_modified_utc: datetime, size: int):
        """Initialize a CloudFile object

        Attributes
        ----------
        name : str
            The filename
        last_modified_utc : datetime
            The last modified date of the file, in UTC
        size : int
            The size of the file, in bytes
        """
        self.name = _fix_path_slashes(name)
        self.last_modified_utc = last_modified_utc
        self.size = size


class CloudCredentials:
    """Authentication credentials for a cloud storage service"""

    def __init__(self):
        pass


class CloudService:
    """Provides a unified interface for different cloud storage services

    Attributes
    ----------
    cloud_type : CloudType
        Which cloud service it is ("azure", "huggingface", "sftp", etc.)
    access : CloudAccess
        What kind of access/permission ("read_only", "read_write")
    credentials : CloudCredentials
        Authentication credentials
    verbose : bool, optional
        Whether to produce verbose output. Default is False.

    """

    def __init__(
        self,
        cloud_type: CloudType,
        credentials: CloudCredentials,
        access: CloudAccess,
        verbose: bool = False,
    ):
        """Initialize a CloudService object

        Attributes
        ----------
        cloud_type : CloudType
            Which cloud service it is ("azure", "huggingface", "sftp", etc.)
        access : CloudAccess
            What kind of access/permission ("read_only", "read_write")
        credentials : CloudCredentials
            Authentication credentials
        verbose : bool, optional
            Whether to produce verbose output. Default is False.
        """
        self.cloud_type = cloud_type
        self.access = access
        self.write = access == "read_write"
        self.credentials = credentials
        self.verbose = verbose
        pass

    def list_files(self, remote_path: str) -> list[CloudFile]:
        """List all the files at the given path on the cloud storage service

        Parameters
        ----------
        remote_path : str
            Path on the remote service you want to query

        Returns
        -------
        list[CloudFile]
            A listing of all the files contained within the queried path on the remote cloud storage service
        """
        pass

    def download_file(self, remote_file: CloudFile, local_file_path: str):
        """Download a remote file from the cloud storage service

        Parameters
        ----------
        remote_file : CloudFile
            The file to download
        local_file_path : str
            The path on your local computer you want to save the remote file to
        """
        r = os.path.basename(remote_file.name)
        l = os.path.basename(local_file_path)
        if r != l:
            raise ValueError(f"Remote path '{r}' does not match local path '{l}'.")

    def upload_file(self, remote_file_path: str, local_file_path: str):
        """Upload a local file to the cloud storage service

        Parameters
        ----------
        remote_file_path : str
            The remote path on the cloud storage service you want to upload your local file to
        local_file_path : str
            The local path to the file on your local computer that you want to upload
        """
        if not self.write:
            raise ValueError("Access denied. This service does not have write access.")
        r = os.path.basename(remote_file_path)
        l = os.path.basename(local_file_path)
        if r != l:
            raise ValueError(f"Remote path '{r}' does not match local path '{l}'.")

    def sync_files(
        self,
        locality: str,
        local_folder: str,
        remote_folder: str,
        dry_run: bool = False,
        verbose: bool = False,
        ignore_paths: list[str] = None,
    ):
        """Synchronize files between your local computer and the cloud storage service.

        Parameters
        ----------
        locality : str
            Unique identifier for a locality/project
        local_folder : str
            Path on your local computer
        remote_folder : str
            Path on the remote cloud storage service
        dry_run : bool
            Prints all the operations that would have run, but does not actually execute them
        verbose : bool, optional
            Whether to print verbose output. Default is False.
        ignore_paths : list[str], optional
            List of paths that should NOT be synchronized. Anything within a matching directory will be skipped over.
        """
        # Build a dictionary of remote files: {relative_path: file}
        remote_files = {}
        if verbose:
            print(
                f'Syncing files from local="{local_folder}" to remote="{remote_folder}"...'
            )
        for file in self.list_files(remote_folder):
            remote_files[_fix_path_slashes(file.name)] = file

        # Build a dictionary of local files relative to the local folder.
        local_files = []
        remote_file_map = {}
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                # Compute the relative path with respect to the given local folder.
                rel_path = _fix_path_slashes(
                    os.path.relpath(os.path.join(root, file), local_folder)
                )
                loc_bits = locality.split("-")
                loc_path = os.path.join("", *loc_bits)
                remote_file_path = os.path.join(loc_path, rel_path)
                local_file_path = os.path.join(root, file)
                remote_file_path = _fix_path_slashes(remote_file_path)
                local_file_path = _fix_path_slashes(local_file_path)

                # Check if this file should be ignored
                should_ignore = _should_ignore(
                    rel_path, remote_file_path, ignore_paths, verbose=verbose
                )

                if should_ignore:
                    continue

                entry = {"remote": remote_file_path, "local": local_file_path}
                local_files.append(entry)
                remote_file_map[remote_file_path] = entry
            for dir in dirs:
                rel_path = _fix_path_slashes(
                    os.path.relpath(os.path.join(root, dir), local_folder)
                )
                loc_bits = locality.split("-")
                loc_path = os.path.join("", *loc_bits)
                remote_file_path = os.path.join(loc_path, rel_path)
                remote_file_path = _fix_path_slashes(remote_file_path)

                # Check if this directory should be ignored
                should_ignore = False
                if ignore_paths:
                    for ignore_path in ignore_paths:
                        if ignore_path in rel_path or ignore_path in remote_file_path:
                            if verbose:
                                print(
                                    f"Ignoring directory '{rel_path}' because it matches ignore pattern '{ignore_path}'"
                                )
                            should_ignore = True
                            break

                if should_ignore:
                    continue

                entry = {"remote": remote_file_path, "local": os.path.join(root, dir)}
                local_files.append(entry)
                remote_file_map[remote_file_path] = entry

        for key in remote_file_map:
            print(key)

        # Process files that exist remotely:
        for rel_path, file in remote_files.items():
            # Check if this remote file should be ignored
            should_ignore = False
            if ignore_paths:
                for ignore_path in ignore_paths:
                    if ignore_path in rel_path:
                        if verbose:
                            print(
                                f"Ignoring remote file '{rel_path}' because it matches ignore pattern '{ignore_path}'"
                            )
                        should_ignore = True
                        break

            if should_ignore:
                continue

            local_file_exists = False

            if rel_path in remote_file_map:
                local_file_path = remote_file_map[rel_path]["local"]
                local_file_exists = os.path.exists(local_file_path)
            else:
                # Construct the local file path properly
                # First, check if the remote_folder is a prefix of the rel_path
                if rel_path.startswith(remote_folder):
                    # Remove the remote_folder prefix and any leading slash
                    path_without_prefix = rel_path[len(remote_folder) :].lstrip("/")
                    local_file_path = _fix_path_slashes(
                        os.path.join(local_folder, path_without_prefix)
                    )
                else:
                    # If remote_folder is not a prefix, just join the paths
                    local_file_path = _fix_path_slashes(
                        os.path.join(local_folder, rel_path)
                    )

            if not local_file_exists:
                if local_file_path is not None:
                    # Create the local directory if it doesn't exist.
                    if not os.path.exists(os.path.dirname(local_file_path)):
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    # If the file IS just a directory, then we're done.
                    if os.path.isdir(local_file_path):
                        continue

                    # Check if this file should be ignored before downloading
                    should_ignore = False
                    if ignore_paths:
                        for ignore_path in ignore_paths:
                            if (
                                ignore_path in rel_path
                                or ignore_path in local_file_path
                            ):
                                if verbose:
                                    print(
                                        f"Ignoring download of '{rel_path}' because it matches ignore pattern '{ignore_path}'"
                                    )
                                should_ignore = True
                                break

                    if should_ignore:
                        continue

                # File exists in remote only: download it
                if verbose:
                    print(
                        f"Local file '{local_file_path}' missing for remote file '{rel_path}'. Downloading..."
                    )
                _print_download(file.name, local_file_path)
                if not dry_run:
                    self.download_file(file, local_file_path)
            else:
                if os.path.isdir(local_file_path):
                    continue

                # Both sides exist: compare file size and last modified timestamp.
                local_size = os.path.getsize(local_file_path)
                remote_size = file.size

                local_mod_time_utc = _get_local_file_mod_time_utc(local_file_path)
                remote_mod_time_utc = file.last_modified_utc

                TIME_TOLERANCE = timedelta(days=1)
                size_delta = abs(local_size - remote_size)
                time_delta = abs(remote_mod_time_utc - local_mod_time_utc)

                # If both the size and modification time are nearly identical, assume they are in sync.
                if size_delta == 0 and time_delta <= TIME_TOLERANCE:
                    if verbose:
                        print(f"{rel_path}: Files are in sync. No action needed.")
                    continue

                if verbose:
                    print(f"\nConflict for '{rel_path}':")
                    print(
                        f"-->Local  - size: {local_size:10,.0f} bytes, modified: {local_mod_time_utc}"
                    )
                    print(
                        f"-->Remote - size: {remote_size:10,.0f} bytes, modified: {remote_mod_time_utc}"
                    )
                    print(f"-->Size delta: {size_delta:10,.0f} bytes")
                    print(f"-->Time delta: {time_delta}")

                # Decide which version is more current.
                if remote_mod_time_utc > local_mod_time_utc:
                    if verbose:
                        print("  Remote file is newer. Downloading remote version...")
                    _print_download(file.name, local_file_path)
                    if not dry_run:
                        self.download_file(file, local_file_path)
                elif self.write and (local_mod_time_utc > remote_mod_time_utc):
                    if verbose:
                        print("  Local file is newer. Uploading local version...")
                    _print_upload(file.name, local_file_path)
                    if not dry_run:
                        self.upload_file(file.name, local_file_path)

        # Process files that exist locally but not remotely.
        for entry in local_files:
            remote_path = entry["remote"]
            local_file_path = entry["local"]
            if os.path.isdir(local_file_path):
                continue

            if self.write and (remote_path not in remote_files):
                # File exists in local only: upload it.
                if verbose:
                    print(
                        f"Remote file missing for local file '{local_file_path}'. Uploading..."
                    )
                _print_upload(remote_path, local_file_path)
                if not dry_run:
                    self.upload_file(remote_path, local_file_path)


def _should_ignore(rel_path: str, remote_path: str, ignore_paths, *, verbose=False):
    rel = Path(rel_path).resolve()  # normalise once
    rem = Path(remote_path).resolve()

    for raw_pat in ignore_paths:
        pat = raw_pat.strip()  # allow trailing spaces / new‑lines

        # Treat it as a glob pattern first
        if fnmatch.fnmatch(rel.as_posix(), pat) or fnmatch.fnmatch(rem.as_posix(), pat):
            if verbose:
                print(f"Ignoring '{rel_path}' (matched glob '{pat}')")
            return True

        # Then treat it as a literal path / prefix
        if rel.is_relative_to(pat) or rem.is_relative_to(pat):
            if verbose:
                print(f"Ignoring '{rel_path}' (under '{pat}')")
            return True

    return False


def _print_download(remote_file: str, local_file: str):
    print(f"Downloading '{local_file}' <-- '{remote_file}'...")


def _print_upload(remote_file: str, local_file: str):
    print(f"Uploading '{local_file}' --> '{remote_file}'...")


def _fix_path_slashes(path: str):
    return path.replace("\\", "/")


def _get_local_file_mod_time_utc(file_path):
    """Return the local file's last modified time as a UTC datetime."""
    timestamp = os.path.getmtime(file_path)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)
