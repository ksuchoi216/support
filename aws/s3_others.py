import json
import os
import re
from typing import Dict, List, TypedDict

import boto3


from loguru import logger
from core.support import load_file


# processing file path string ============================================
def refine_prefix(prefix, bucket):
    if prefix.startswith("s3://"):
        prefix = prefix[5:]
    else:
        prefix = prefix

    # if prefix contains bucket name, remove it by using regex
    if prefix.startswith(bucket):
        pattern = re.compile(rf"^{re.escape(bucket)}/?")
        prefix = pattern.sub("", prefix)

    return prefix


# def separate_bucket_and_prefix_from_s3_url(s3_url: str):
#     if s3_url.startswith("s3://"):
#         s3_url = s3_url[5:]

#     splits = s3_url.split("/")
#     bucket = splits[0]
#     prefix = "/".join(splits[1:])
#     return bucket, prefix


def extract_bucket_from_s3_url(s3_url: str) -> str:
    # print(s3_url)
    if s3_url.startswith("s3://"):
        bucket = s3_url.split("/")[2]
        return bucket
    else:
        raise ValueError(f"Invalid S3 PDF path: {s3_url}")


def extract_doc_prefix_from_s3_url(s3_url: str, bucket: str = None) -> tuple[str, str]:
    if bucket is None:
        bucket = extract_bucket_from_s3_url(s3_url)
    prefix = refine_prefix(s3_url, bucket)

    if prefix.endswith(".pdf"):
        prefix = prefix[:-4]
        extension = ".pdf"
    elif prefix.endswith(".html"):
        prefix = prefix[:-5]
        extension = ".html"
    elif prefix.endswith(".txt"):
        prefix = prefix[:-4]
        extension = ".txt"
    else:
        raise ValueError(f"Invalid file type: {s3_url}")

    return prefix, extension


def extract_prefix_from_s3_url(s3_url: str) -> str:
    if s3_url.endswith(".pdf"):
        prefix = s3_url[:-4]
    else:
        prefix = s3_url
    return prefix


# s3 upload, download ====================================================


def upload_to_s3(
    data: Dict | List | str,
    prefix: str,
    bucket: str,
    encoding: str = "utf-8",
):
    s3 = boto3.client("s3")

    if isinstance(data, (dict, list)):
        body = json.dumps(data, ensure_ascii=False, indent=4).encode(encoding)
        content_type = "application/json"
    elif isinstance(data, str):
        body = data.encode(encoding)
        content_type = "text/plain"
    else:
        raise TypeError(f"Data must be dict, list, or str, not {type(data).__name__}")

    # 업로드 실행
    s3.put_object(Bucket=bucket, Key=prefix, Body=body, ContentType=content_type)
    logger.info("Uploaded to s3://%s/%s (%s)", bucket, prefix, content_type)


def upload_folder_to_s3(local_dir, bucket, prefix):
    if bucket.startswith("s3://"):
        bucket = bucket[5:]

    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # 로컬 기준 경로를 S3 상대경로로 변환
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(prefix, relative_path).replace("\\", "/")

            s3.upload_file(local_path, bucket, s3_key)
            logger.info("Uploaded: %s → s3://%s/%s", local_path, bucket, s3_key)


def download_from_s3(
    prefix: str,
    bucket: str,
    local_path: str,
) -> str:
    # if directory does not exist, create it
    local_dir = os.path.dirname(local_path)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    s3 = boto3.client("s3")
    try:
        s3.download_file(Bucket=bucket, Key=prefix, Filename=str(local_path))
        logger.info("Downloaded from s3://%s/%s to %s", bucket, prefix, local_path)
        return str(local_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download from s3://{bucket}/{prefix} to {local_path}: {str(e)}"
        ) from e


def download_artifact_if_exists(
    has_flag: bool,
    folder: str,
    filename: str,
    temp_dir: str,
    doc_prefix: str,
    bucket: str,
) -> str:
    if has_flag:
        local_path = f"{temp_dir}/{folder}/{filename}"
        s3_prefix = f"{doc_prefix}/{folder}/{filename}"
        download_from_s3(local_path=local_path, prefix=s3_prefix, bucket=bucket)
        return load_file(local_path)
    return ""


def download_folder_from_s3(bucket, prefix, local_dir):
    """
    S3의 특정 폴더(prefix)에 있는 모든 파일을 로컬로 다운로드합니다.

    Args:
        bucket (str): S3 버킷 이름 (예: 'oneit-student-transcript-file-dev')
        prefix (str): 다운로드할 S3 경로(prefix) (예: 'test/vdb/')
        local_dir (str): 로컬에 저장할 기본 폴더 (기본값: './tmp')
    """

    if bucket.startswith("s3://"):
        bucket = bucket[5:]

    s3 = boto3.client("s3")

    # prefix 아래 객체 목록 가져오기
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        logger.info("No files found in s3://%s/%s", bucket, prefix)
        return

    for obj in response["Contents"]:
        s3_key = obj["Key"]
        # prefix 이후의 상대경로 계산
        relative_path = os.path.relpath(s3_key, prefix)
        local_path = os.path.join(local_dir, relative_path)
        # 로컬 폴더가 없다면 생성
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # 파일 다운로드
        s3.download_file(bucket, s3_key, local_path)
        logger.info("Downloaded: s3://%s/%s → %s", bucket, s3_key, local_path)


# s3 checking ====================================================


def check_file_existance_in_s3_folder(folder_prefix: str, bucket: str) -> bool:
    """
    S3에서 특정 '폴더' (prefix)에 파일이 존재하는지 확인.

    Args:
        bucket_name (str): S3 버킷 이름
        prefix (str): 폴더 경로 (예: "test/" or "data/input/")
    Returns:
        bool: 해당 경로에 비어있지 않은 파일이 하나라도 있으면 True, 없으면 False
    """

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
        for obj in page.get("Contents", []):
            if obj.get("Size", 0) > 0:
                return True
    return False


def check_certain_file_in_s3(file_prefix, bucket):
    s3_client = boto3.client("s3")
    try:
        s3_client.head_object(Bucket=bucket, Key=file_prefix)
        logger.info(f"File exists in S3: s3://{bucket}/{file_prefix}")
    except:
        raise FileExistsError(f"File does not exist in S3: s3://{bucket}/{file_prefix}")


class S3ArtifactStatus(TypedDict):
    has_extracted_text: bool
    has_images64: bool
    has_organizated_text: bool
    has_local_doc_path: bool
    local_doc_path: str
    doc_extension: str


def check_artifact_status_in_s3(
    config, doc_prefix, doc_extension, bucket, is_3rd_grade
):
    logger.info("Checking files in %s/%s/...", bucket, doc_prefix)
    has_organizated_text = check_file_existance_in_s3_folder(
        folder_prefix=f"{doc_prefix}/{config.s3.organization_folder}", bucket=bucket
    )
    has_extracted_text = check_file_existance_in_s3_folder(
        folder_prefix=f"{doc_prefix}/{config.s3.extraction_folder}",
        bucket=bucket,
    )

    has_image64_text = check_file_existance_in_s3_folder(
        folder_prefix=f"{doc_prefix}/{config.s3.images64_folder}", bucket=bucket
    )
    artifact_status = S3ArtifactStatus(
        has_extracted_text=has_extracted_text,
        has_images64=has_image64_text,
        has_organizated_text=has_organizated_text,
        has_local_doc_path=False,
        local_doc_path="",
        doc_extension=doc_extension,
    )
    logger.info("Has organizated text: %s", has_organizated_text)
    logger.info("Has images64 text: %s", has_image64_text)
    logger.info("Has extracted text: %s", has_extracted_text)
    if has_organizated_text:
        download_from_s3(
            local_path=f"{config.s3.temp_dir}/{config.s3.organization_folder}/{config.s3.organized_text_filename}",
            prefix=f"{doc_prefix}/{config.s3.organization_folder}/{config.s3.organized_text_filename}",
            bucket=bucket,
        )
    if has_image64_text and not has_organizated_text and is_3rd_grade:
        download_from_s3(
            local_path=f"{config.s3.temp_dir}/{config.s3.images64_folder}/{config.s3.images64_filename}",
            prefix=f"{doc_prefix}/{config.s3.images64_folder}/{config.s3.images64_filename}",
            bucket=bucket,
        )
    if has_extracted_text and not has_organizated_text:
        download_from_s3(
            local_path=f"{config.s3.temp_dir}/{config.s3.extraction_folder}/{config.s3.extracted_text_filename}",
            prefix=f"{doc_prefix}/{config.s3.extraction_folder}/{config.s3.extracted_text_filename}",
            bucket=bucket,
        )
    if not has_extracted_text and not has_image64_text:
        logger.info(
            f"{doc_prefix}.pdf is downloaded at {config.s3.temp_dir}/{config.s3.local_pdf_filename}"
        )
        check_certain_file_in_s3(
            file_prefix=f"{doc_prefix}{doc_extension}", bucket=bucket
        )
        local_doc_path = download_from_s3(
            local_path=f"{config.s3.temp_dir}/{config.s3.local_pdf_filename}",
            prefix=f"{doc_prefix}{doc_extension}",
            bucket=bucket,
        )
        if not os.path.exists(local_doc_path):
            raise FileNotFoundError(
                f"Downloaded PDF file not found at local: {local_doc_path} for {doc_prefix}.pdf"
            )
        logger.info("PDF file is valid: %s", local_doc_path)
        artifact_status["local_doc_path"] = local_doc_path
        artifact_status["has_local_doc_path"] = True

    return artifact_status


# combined ===================================================


def create_merged_text(
    files_and_3rd_grade,
    s3_bucket_name,
    save_local_dir="./tmp/merger",
    relative_organized_text_path="organization/organized_text.txt",
):
    organized_texts = []
    bucket = s3_bucket_name
    is_3rd_grades = []
    for tup in files_and_3rd_grade:
        s3_url = tup[0]
        is_3rd_grade = tup[1]

        logger.info("Processing S3 URL: %s", s3_url)
        logger.info("Is 3rd grade: %s", is_3rd_grade)

        if bucket is None:
            bucket = extract_bucket_from_s3_url(s3_url)

        prefix = extract_prefix_from_s3_url(s3_url)
        prefix = f"{prefix}/{relative_organized_text_path}"
        prefix = refine_prefix(prefix, bucket)

        check_certain_file_in_s3(file_prefix=prefix, bucket=bucket)
        local_path = f"{save_local_dir}/{prefix}"
        logger.info("Prefix: %s, Bucket: %s", prefix, bucket)

        logger.info("Downloading from s3://%s/%s to %s", bucket, prefix, local_path)
        org_path = download_from_s3(
            local_path=local_path,
            prefix=prefix,
            bucket=bucket,
        )
        logger.info(
            "Downloaded from s3://%s/%s to %s",
            bucket,
            prefix,
            local_path,
        )

        organized_text = load_file(org_path)

        # if organized_text is empty, raise error
        if not organized_text.strip():
            raise ValueError(
                f"The organized text from s3://{bucket}/{prefix} is empty."
            )

        organized_texts.append(organized_text)
        is_3rd_grades.append(is_3rd_grade)

    # sort by 3rd grade last
    organized_texts = [text for _, text in sorted(zip(is_3rd_grades, organized_texts))]
    merged_text = "\n\n".join(organized_texts)
    return merged_text
