from __future__ import annotations

import aioboto3
from storysmith.settings import Settings


class S3Storage:
    """StoragePort backed by S3 (aioboto3)."""

    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.s3_bucket
        self._region = settings.aws_region
        self._session = aioboto3.Session()

    async def put(self, *, key: str, data: bytes, content_type: str) -> str:
        async with self._session.client("s3", region_name=self._region) as s3:
            await s3.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)
        return f"s3://{self._bucket}/{key}"

    async def get(self, *, uri: str) -> bytes:
        bucket, key = self._parse_uri(uri)
        async with self._session.client("s3", region_name=self._region) as s3:
            obj = await s3.get_object(Bucket=bucket, Key=key)
            body = await obj["Body"].read()
        return bytes(body)

    async def presign(self, *, uri: str, expires_s: int = 3600) -> str:
        bucket, key = self._parse_uri(uri)
        async with self._session.client("s3", region_name=self._region) as s3:
            url = await s3.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_s
            )
        return str(url)

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"S3Storage cannot resolve non-s3 uri: {uri!r}")
        bucket, _, key = uri.removeprefix("s3://").partition("/")
        return bucket, key
