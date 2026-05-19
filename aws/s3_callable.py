from functools import wraps
from inspect import signature
from pathlib import Path

from core.aws.s3_file_transfer import (
    check_file_in_s3,
    download_from_s3,
    upload_to_s3,
)
from core.aws.s3_url import S3Location
from typing import Callable


def communicate_s3(
    s3_location: S3Location, artifact_filename: str, function: Callable, *args, **kwargs
):
    artifact_local_path = s3_location.artifact_local_path(artifact_filename)
    artifact_s3_path = s3_location.artifact_s3_key(artifact_filename)
    bucket = s3_location.bucket

    if check_file_in_s3(bucket, artifact_s3_path):
        return download_from_s3(bucket, artifact_s3_path, artifact_local_path)

    result = function(*args, **kwargs)
    upload_to_s3(
        bucket,
        artifact_s3_path,
        Path(result) if result else artifact_local_path,
    )
    return result


def s3_decorator(artifact_filename: str):
    def decorator(function: Callable):
        @wraps(function)
        def wrapper(*args, **kwargs):
            bound_args = signature(function).bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            s3_location = S3Location(
                s3_url=bound_args.arguments["s3_url"],
                artifact_foldername=bound_args.arguments.get(
                    "artifact_foldername", "artifacts"
                ),
                download_dir=bound_args.arguments.get("download_dir", "./temp"),
            )
            return communicate_s3(
                s3_location, artifact_filename, function, *args, **kwargs
            )

        return wrapper

    return decorator
