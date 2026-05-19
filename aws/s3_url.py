import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from core.aws.s3_file_transfer import download_from_s3


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
    Low-level path helper for one source S3 object.

    This class does not know parser concepts such as ArtifactName or DocumentType.
    It only parses the source S3 URL and builds paths from plain filenames.

    Example:
        s3_url = "s3://my-bucket/915/학생 생기부.pdf"
        loc = S3Location(s3_url)
    """

    s3_url: str
    artifact_foldername: str = "artifacts"
    download_dir: str = "./temp"
    bucket: str = field(init=False)
    raw_doc_s3_key: str = field(init=False)
    raw_doc_filename: str = field(init=False)

    def __post_init__(self) -> None:
        self.bucket = extract_bucket_from_s3_url(self.s3_url)
        self.raw_doc_s3_key = refine_prefix(self.s3_url, self.bucket)
        self.raw_doc_filename = Path(self.raw_doc_s3_key).name

    @property
    def local_raw_doc_path(self) -> Path:
        """Default local path for the downloaded source document."""
        return self.artifact_local_path(self.raw_doc_filename)

    def download_raw_doc(self, local_path: Path | str | None = None) -> Path:
        """Download the raw document from S3. Returns the local path."""
        target_path = (
            Path(local_path) if local_path is not None else self.local_raw_doc_path
        )
        return download_from_s3(
            bucket=self.bucket,
            prefix=self.raw_doc_s3_key,
            local_path=target_path,
        )

    def artifact_local_path(self, filename: str) -> Path:
        """Local path for a plain artifact filename."""
        return Path(self.download_dir) / self.artifact_foldername / filename

    def artifact_s3_key(self, filename: str) -> str:
        """S3 key for a plain artifact filename under the source document."""
        return f"{self.raw_doc_s3_key}/{self.artifact_foldername}/{filename}"
