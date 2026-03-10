from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from app.config import ServiceSettings


@dataclass(frozen=True)
class StoredObject:
    backend: str
    bucket: str | None
    object_key: str
    size_bytes: int


class ObjectStorage(Protocol):
    backend_name: str

    def upload_file(
        self,
        local_path: Path,
        object_key: str,
        content_type: str,
    ) -> StoredObject:
        ...

    def build_access_url(self, bucket: str | None, object_key: str) -> str:
        ...


class LocalObjectStorage:
    backend_name = "local"

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def upload_file(
        self,
        local_path: Path,
        object_key: str,
        content_type: str,
    ) -> StoredObject:
        destination = self._root / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, destination)
        return StoredObject(
            backend=self.backend_name,
            bucket=None,
            object_key=object_key.replace("\\", "/"),
            size_bytes=destination.stat().st_size,
        )

    def build_access_url(self, bucket: str | None, object_key: str) -> str:
        return f"local://{object_key}"


class MinioObjectStorage:
    backend_name = "minio"

    def __init__(self, settings: ServiceSettings) -> None:
        from minio import Minio

        self._settings = settings
        self._bucket = settings.minio_bucket
        self._client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if self._client.bucket_exists(self._bucket):
            return
        if not self._settings.minio_auto_create_bucket:
            raise ValueError(
                f"Bucket MinIO inexistente: {self._bucket}. "
                "Activa YOLO_WS_MINIO_AUTO_CREATE_BUCKET o crea el bucket manualmente."
            )
        self._client.make_bucket(self._bucket)

    def _rewrite_public_endpoint(self, url: str) -> str:
        if not self._settings.minio_public_endpoint:
            return url
        public_parts = urlsplit(self._settings.minio_public_endpoint)
        url_parts = urlsplit(url)
        scheme = public_parts.scheme or url_parts.scheme
        netloc = public_parts.netloc or public_parts.path or url_parts.netloc
        return urlunsplit(
            (scheme, netloc, url_parts.path, url_parts.query, url_parts.fragment)
        )

    def upload_file(
        self,
        local_path: Path,
        object_key: str,
        content_type: str,
    ) -> StoredObject:
        normalized_key = object_key.replace("\\", "/")
        self._client.fput_object(
            bucket_name=self._bucket,
            object_name=normalized_key,
            file_path=str(local_path),
            content_type=content_type,
        )
        return StoredObject(
            backend=self.backend_name,
            bucket=self._bucket,
            object_key=normalized_key,
            size_bytes=local_path.stat().st_size,
        )

    def build_access_url(self, bucket: str | None, object_key: str) -> str:
        from minio.error import S3Error

        bucket_name = bucket or self._bucket
        try:
            url = self._client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_key,
                expires=timedelta(seconds=self._settings.presigned_url_expiry_seconds),
            )
            return self._rewrite_public_endpoint(url)
        except S3Error as exc:
            raise RuntimeError(
                f"No se pudo generar presigned URL para {bucket_name}/{object_key}"
            ) from exc


def create_storage(settings: ServiceSettings) -> ObjectStorage:
    if settings.storage_backend == "minio":
        return MinioObjectStorage(settings)
    return LocalObjectStorage(settings.resolve_local_storage_root())
