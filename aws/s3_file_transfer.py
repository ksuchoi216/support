from pathlib import Path
import boto3
import os
import unicodedata

from loguru import logger


def upload_to_s3(
    bucket: str,
    prefix: str,
    local_path: Path | str,
):
    # check local_path file exist
    if isinstance(local_path, str):
        local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # upload the file to s3
    s3_client = boto3.client("s3")

    # If prefix is a directory (ends with /), append the filename
    s3_key = prefix
    if prefix.endswith("/"):
        s3_key = f"{prefix}{local_path.name}"

    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"Successfully uploaded {local_path} to s3://{bucket}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to S3: {e}")
        raise


def _resolve_s3_key_normalization(s3_client, bucket: str, prefix: str) -> str:
    """
    Checks the exact prefix key first. If not found, checks NFD and NFC
    normalization forms (extremely common for macOS Korean filenames on S3).
    Returns the key that actually exists, or the original prefix if none do.
    """
    # 1. Try original
    try:
        s3_client.head_object(Bucket=bucket, Key=prefix)
        return prefix
    except Exception:
        pass

    # 2. Try NFD (Decomposed, standard on macOS)
    nfd_prefix = unicodedata.normalize("NFD", prefix)
    if nfd_prefix != prefix:
        try:
            s3_client.head_object(Bucket=bucket, Key=nfd_prefix)
            return nfd_prefix
        except Exception:
            pass

    # 3. Try NFC (Composed, standard on Windows/Linux)
    nfc_prefix = unicodedata.normalize("NFC", prefix)
    if nfc_prefix != prefix:
        try:
            s3_client.head_object(Bucket=bucket, Key=nfc_prefix)
            return nfc_prefix
        except Exception:
            pass

    return prefix


def download_from_s3(
    bucket: str,
    prefix: str,
    local_path: Path | str,
):
    if isinstance(local_path, str):
        local_path = Path(local_path)

    # if directory does not exist, create it
    local_dir = Path(local_path).parent
    if not local_dir.exists():
        local_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")

    # Resolve unicode normalization (NFC/NFD)
    resolved_prefix = _resolve_s3_key_normalization(s3, bucket, prefix)

    try:
        s3.download_file(Bucket=bucket, Key=resolved_prefix, Filename=str(local_path))
        logger.info(
            "Downloaded from s3://%s/%s to %s", bucket, resolved_prefix, local_path
        )
        return local_path
    except Exception as e:
        raise ValueError(
            f"Failed to download from s3://{bucket}/{resolved_prefix} to {local_path}: {str(e)}"
        )


def check_file_in_s3(bucket, prefix):
    s3_client = boto3.client("s3")
    # Resolve unicode normalization (NFC/NFD)
    resolved_prefix = _resolve_s3_key_normalization(s3_client, bucket, prefix)
    try:
        s3_client.head_object(Bucket=bucket, Key=resolved_prefix)
        return True
    except Exception:
        return False
