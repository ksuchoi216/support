from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar


ArtifactT = TypeVar("ArtifactT")


def _create_s3_location(
    s3_url: str,
    artifact_foldername: str,
    download_dir: str,
):
    from core.aws.s3_url import S3Location

    return S3Location(
        s3_url=s3_url,
        artifact_foldername=artifact_foldername,
        download_dir=download_dir,
    )


def _check_file_in_s3(bucket: str, prefix: str) -> bool:
    from core.aws.s3_file_transfer import check_file_in_s3

    return check_file_in_s3(bucket, prefix)


def _download_from_s3(bucket: str, prefix: str, local_path: Path) -> Path:
    from core.aws.s3_file_transfer import download_from_s3

    return download_from_s3(bucket, prefix, local_path)


class ArtifactManager(Generic[ArtifactT]):
    """Resolve artifact processing order and dependency downloads."""

    def __init__(
        self,
        artifacts: Sequence[ArtifactT],
        needed_artifacts_for_artifact: Mapping[ArtifactT, Sequence[ArtifactT]],
        artifact_exists: Callable[[ArtifactT], bool],
        download_artifact_if_needed: Callable[[ArtifactT], None],
    ) -> None:
        self.artifacts = list(artifacts)
        self.needed_artifacts_for_artifact = needed_artifacts_for_artifact
        self.artifact_exists = artifact_exists
        self.download_artifact_if_needed = download_artifact_if_needed

    def initialize_artifact_list(self) -> list[ArtifactT]:
        """Return artifacts from the first missing artifact onward."""
        first_missing_index = None

        for index, artifact in enumerate(self.artifacts):
            if not self.artifact_exists(artifact):
                first_missing_index = index
                break

        if first_missing_index is None:
            return []

        first_missing_artifact = self.artifacts[first_missing_index]
        self.download_needed_artifacts(first_missing_artifact)

        return self.artifacts[first_missing_index:]

    def download_needed_artifacts(self, artifact: ArtifactT) -> None:
        for needed_artifact in self.needed_artifacts_for_artifact.get(artifact, []):
            self.download_artifact_if_needed(needed_artifact)


class S3ArtifactManager(Generic[ArtifactT]):
    """
    Reusable artifact manager backed by one source S3 object.

    Domain-specific documents should provide artifact enums/order/dependencies.
    This class owns S3 paths, local paths, raw-document handling, and artifact
    processing list initialization.
    """

    def __init__(
        self,
        id: int,
        s3_url: str,
        artifact_file_name: Any | None = None,
        artifacts: Sequence[ArtifactT] | None = None,
        needed_artifacts_for_artifact: (
            Mapping[ArtifactT, Sequence[ArtifactT]] | None
        ) = None,
        raw_artifact: ArtifactT | None = None,
        artifact_foldername: str = "artifacts",
        download_dir: str = "./temp",
    ) -> None:
        self.id = id
        self.artifact_file_name = artifact_file_name
        if raw_artifact is None and artifact_file_name is None:
            raise ValueError("artifact_file_name or raw_artifact is required")

        self.raw_artifact = raw_artifact
        if self.raw_artifact is None:
            self.raw_artifact = getattr(artifact_file_name, "RAW_DOCUMENT", None)
        self._s3 = _create_s3_location(
            s3_url=s3_url,
            artifact_foldername=artifact_foldername,
            download_dir=download_dir,
        )
        self.all_artifacts: list[ArtifactT] = []
        self.artifacts_to_be_processed: list[ArtifactT] = []
        self.artifact_list: list[ArtifactT] = self.artifacts_to_be_processed

        if artifacts is not None and needed_artifacts_for_artifact is not None:
            self.configure_artifacts(
                artifacts=artifacts,
                needed_artifacts_for_artifact=needed_artifacts_for_artifact,
            )

    def configure_artifacts(
        self,
        artifacts: Sequence[ArtifactT],
        needed_artifacts_for_artifact: Mapping[ArtifactT, Sequence[ArtifactT]],
    ) -> list[ArtifactT]:
        self.all_artifacts = [
            artifact for artifact in artifacts if artifact != self.raw_artifact
        ]
        self._artifact_manager = ArtifactManager(
            artifacts=self.all_artifacts,
            needed_artifacts_for_artifact=needed_artifacts_for_artifact,
            artifact_exists=self._artifact_exists_in_s3,
            download_artifact_if_needed=self._download_artifact_if_needed,
        )
        self.artifacts_to_be_processed = (
            self._artifact_manager.initialize_artifact_list()
        )
        self.artifact_list = self.artifacts_to_be_processed
        return self.artifacts_to_be_processed

    def download_needed_artifacts(self, artifact: ArtifactT) -> None:
        self._artifact_manager.download_needed_artifacts(artifact)

    def download_raw_document(self) -> Path:
        if self.raw_artifact is None:
            raise ValueError("raw_artifact is not configured")

        return self._s3.download_raw_doc(
            local_path=self.artifact_local_path(self.raw_artifact)
        )

    def _artifact_exists_in_s3(self, artifact: ArtifactT) -> bool:
        return _check_file_in_s3(self.bucket, self.artifact_s3_key(artifact))

    def _download_artifact_if_needed(self, artifact: ArtifactT) -> None:
        if not self._artifact_exists_in_s3(artifact):
            return

        local_path = self.artifact_local_path(artifact)
        if local_path.exists():
            return

        if artifact == self.raw_artifact:
            self.download_raw_document()
            return

        _download_from_s3(
            bucket=self.bucket,
            prefix=self.artifact_s3_key(artifact),
            local_path=local_path,
        )

    @property
    def bucket(self) -> str:
        return self._s3.bucket

    @property
    def local_artifact_dir(self) -> Path:
        return Path(self._s3.download_dir) / self._s3.artifact_foldername / str(self.id)

    def artifact_filename(self, artifact: ArtifactT) -> str:
        if artifact == self.raw_artifact:
            return self._s3.raw_doc_filename
        return str(getattr(artifact, "value", artifact))

    def artifact_local_path(self, artifact: ArtifactT) -> Path:
        return self.local_artifact_dir / self.artifact_filename(artifact)

    def artifact_s3_key(self, artifact: ArtifactT) -> str:
        if artifact == self.raw_artifact:
            return self._s3.raw_doc_s3_key
        return self._s3.artifact_s3_key(self.artifact_filename(artifact))
