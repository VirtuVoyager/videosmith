from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from storysmith.errors import StorySmithError
from storysmith.settings import Settings

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_KIDS_CATEGORY_ID = "24"  # Entertainment; §7 requires madeForKids=true


class YouTubePublish:
    """PublishPort via the YouTube Data API v3.

    Auth is a refresh token produced once by scripts/youtube_auth.py's
    installed-app OAuth flow (§7) -- this adapter only ever reads/refreshes
    that token, it never runs the interactive consent flow itself.
    """

    def __init__(self, settings: Settings) -> None:
        self._token_path = Path(settings.youtube_token_path)

    def _credentials(self) -> Credentials:
        if not self._token_path.exists():
            raise StorySmithError(
                f"no YouTube refresh token at {self._token_path} -- run "
                "`uv run python scripts/youtube_auth.py` once first"
            )
        creds = Credentials.from_authorized_user_info(
            json.loads(self._token_path.read_text()), SCOPES
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._token_path.write_text(creds.to_json())
        return creds

    def _upload_sync(
        self, *, video: bytes, thumbnail: bytes, title: str, description: str, tags: list[str]
    ) -> str:
        # google-api-python-client is a blocking SDK (httplib2) -- run it off
        # the event loop rather than making the rest of the codebase pretend
        # it's async (§0.2 wants async I/O; asyncio.to_thread reconciles that
        # with a sync-only third-party client).
        youtube = build("youtube", "v3", credentials=self._credentials())
        body: dict[str, Any] = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": _KIDS_CATEGORY_ID,
            },
            "status": {
                "privacyStatus": "private",
                "selfDeclaredMadeForKids": True,
            },
        }
        media = MediaIoBaseUpload(io.BytesIO(video), mimetype="video/mp4", resumable=False)
        response = (
            youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
        )
        video_id = response["id"]

        thumb_media = MediaIoBaseUpload(io.BytesIO(thumbnail), mimetype="image/jpeg")
        youtube.thumbnails().set(videoId=video_id, media_body=thumb_media).execute()

        return f"https://www.youtube.com/watch?v={video_id}"

    async def upload(
        self, *, video: bytes, thumbnail: bytes, title: str, description: str, tags: list[str]
    ) -> str:
        return await asyncio.to_thread(
            self._upload_sync,
            video=video,
            thumbnail=thumbnail,
            title=title,
            description=description,
            tags=tags,
        )
