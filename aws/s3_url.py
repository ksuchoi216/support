import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from core.aws.s3_file_transfer import download_from_s3


# processing file path string ============================================
def extract_bucket_from_s3_url(s3_url: str) -> str:
    parsed = urlparse(s3_url)
    if parsed.scheme == "s3":
        return parsed.netloc
    raise ValueError(f"Invalid S3 URL: {s3_url}")


def refine_prefix(s3_url: str, bucket: str) -> str:
    parsed = urlparse(s3_url)
    if parsed.scheme == "s3":
        return parsed.path.lstrip("/")
    # Fallback logic for non-s3:// strings
    prefix = s3_url.replace("s3://", "")
    return re.sub(rf"^{re.escape(bucket)}/?", "", prefix)


@dataclass
class S3Location:
    """
    Resolves S3 and local paths for a document and its artifacts.

    Example:
        s3_url = "s3://my-bucket/915/학생 생기부.pdf"
        loc = S3Location(s3_url)
    """

    s3_url: str
    artifact_foldername: str = "artifacts"
    download_dir: str = "./temp"

    @property
    def bucket(self) -> str:
        """S3 bucket name.  e.g. 'my-bucket'"""
        return extract_bucket_from_s3_url(self.s3_url)

    @property
    def raw_doc_prefix(self) -> str:
        """S3 key for the raw document.  e.g. '915/학생 생기부.pdf'"""
        return refine_prefix(self.s3_url, self.bucket)

    @property
    def artifact_prefix(self) -> str:
        """S3 prefix for artifacts.  e.g. '915/학생 생기부.pdf/artifacts'"""
        return f"{self.raw_doc_prefix}/{self.artifact_foldername}"

    @property
    def local_artifact_dir(self) -> Path:
        """Local directory for artifacts.  e.g. './temp/artifacts'"""
        return Path(self.download_dir) / self.artifact_foldername

    @property
    def filename(self) -> str:
        """Filename of the raw document.  e.g. '학생 생기부.pdf'"""
        return Path(self.raw_doc_prefix).name

    @property
    def extension(self) -> str:
        """File extension of the raw document.  e.g. '.pdf'"""
        for ext in [".pdf", ".html", ".txt"]:
            if self.raw_doc_prefix.lower().endswith(ext):
                return ext
        raise ValueError(f"No supported extension found in: {self.s3_url}")

    @property
    def prefix(self) -> str:
        """S3 key without the extension.  e.g. '915/학생 생기부'"""
        return self.raw_doc_prefix.removesuffix(self.extension)

    @property
    def local_raw_doc_path(self) -> Path:
        """Local path for the downloaded raw document.  e.g. './temp/artifacts/학생 생기부.pdf'"""
        return Path(self.download_dir) / self.artifact_foldername / self.filename

    def download_raw_doc(self) -> Path:
        """Download the raw document from S3 to local_raw_doc_path. Returns the local path."""
        return download_from_s3(
            bucket=self.bucket,
            prefix=self.raw_doc_prefix,
            local_path=self.local_raw_doc_path,
        )
