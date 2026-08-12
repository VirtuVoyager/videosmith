"""One-time interactive OAuth setup for YouTube publishing (§7).

Run this once locally after downloading an OAuth client secrets JSON (type
"Desktop app") from the Google Cloud Console for a project with the YouTube
Data API v3 enabled:

    uv run python scripts/youtube_auth.py

It opens a browser for consent, then writes a refresh token to
SS_YOUTUBE_TOKEN_PATH (default ./secrets/yt_token.json). publish_youtube.py
reads that file on every upload and refreshes it in place -- this script
never needs to run again unless the token file is deleted or access is
revoked.
"""

from __future__ import annotations

from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from storysmith.settings import Settings
from storysmith_adapters.publish_youtube import SCOPES


def main() -> None:
    settings = Settings()
    secrets_path = Path(settings.youtube_client_secrets_path)
    token_path = Path(settings.youtube_token_path)

    if not secrets_path.exists():
        raise SystemExit(
            f"OAuth client secrets not found at {secrets_path} "
            "(SS_YOUTUBE_CLIENT_SECRETS_PATH) -- download a 'Desktop app' "
            "OAuth client JSON from the Google Cloud Console first"
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
    credentials = flow.run_local_server(port=0)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json())
    print(f"Saved refresh token to {token_path}")


if __name__ == "__main__":
    main()
