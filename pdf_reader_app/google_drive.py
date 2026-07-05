import base64
import hashlib
import json
import os
import re
import secrets
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import certifi
except Exception:
    certifi = None


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD_FILES_URL = "https://www.googleapis.com/upload/drive/v3/files"
DRIVE_SCOPE = " ".join(
    (
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/drive.file",
    )
)
BUILT_IN_GOOGLE_CLIENT_ID = "38125019135-0670baqakamdae39plpvm06jpgoq54o0.apps.googleusercontent.com"
BUILT_IN_GOOGLE_CLIENT_SECRET = ""
DRIVE_FOLDER_NAME = "PDF Reader"
DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
GOOGLE_PDF_MIME_TYPES = {
    "application/pdf",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.spreadsheet",
}


@dataclass
class CloudDocument:
    id: str
    name: str
    mime_type: str
    modified_time: str = ""
    size: str = ""
    thumbnail_link: str = ""
    web_view_link: str = ""

    @property
    def can_export_as_pdf(self) -> bool:
        return self.mime_type.startswith("application/vnd.google-apps.")


class GoogleDriveError(RuntimeError):
    pass


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    server: "_OAuthCallbackServer"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        self.server.oauth_result = {key: values[0] for key, values in params.items() if values}
        body = b"Google Drive login is complete. You can return to PDF Reader."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


class _OAuthCallbackServer(HTTPServer):
    oauth_result: Dict[str, str]


class GoogleDriveClient:
    def __init__(self, token_path: Path, cache_dir: Path, credential_path: Optional[Path] = None) -> None:
        self.token_path = token_path
        self.cache_dir = cache_dir
        self.credential_path = credential_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._token: Dict[str, object] = self._load_token()
        self._ssl_context = self._create_ssl_context()

    @property
    def is_logged_in(self) -> bool:
        return bool(self._token.get("refresh_token") or self._token.get("access_token"))

    def sign_out(self) -> None:
        self._token = {}
        try:
            self.token_path.unlink()
        except FileNotFoundError:
            pass

    def authenticate(self, open_url, process_events=None) -> None:
        client_id, client_secret = self._load_credentials()
        verifier = self._code_verifier()
        challenge = self._code_challenge(verifier)
        state = secrets.token_urlsafe(24)

        callback_server = _OAuthCallbackServer(("127.0.0.1", 0), _OAuthCallbackHandler)
        callback_server.oauth_result = {}
        host, port = callback_server.server_address
        redirect_uri = f"http://{host}:{port}"

        server_thread = threading.Thread(target=callback_server.handle_request, daemon=True)
        server_thread.start()

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": DRIVE_SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        open_url(f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}")

        deadline = time.time() + 15
        while server_thread.is_alive() and time.time() < deadline:
            if process_events is not None:
                process_events()
            server_thread.join(0.1)
        callback_server.server_close()

        result = callback_server.oauth_result
        if not result:
            raise GoogleDriveError("Google login timed out.")
        if result.get("state") != state:
            raise GoogleDriveError("Google login returned an invalid state.")
        if result.get("error"):
            raise GoogleDriveError(f"Google login failed: {result['error']}")
        code = result.get("code")
        if not code:
            raise GoogleDriveError("Google login did not return an authorization code.")

        payload = {
            "client_id": client_id,
            "code": code,
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
        if client_secret:
            payload["client_secret"] = client_secret
        token = self._post_token(payload)
        self._store_token(token)

    def list_documents(self, process_events=None) -> List[CloudDocument]:
        folder_id = self.ensure_documents_folder_id()
        if not folder_id:
            return []

        mime_query = " or ".join(f"mimeType='{mime_type}'" for mime_type in sorted(GOOGLE_PDF_MIME_TYPES))
        query = f"trashed=false and '{folder_id}' in parents and ({mime_query})"
        fields = (
            "nextPageToken,"
            "files(id,name,mimeType,modifiedTime,size,thumbnailLink,webViewLink)"
        )
        documents: List[CloudDocument] = []
        page_token = ""

        while True:
            params = {
                "q": query,
                "fields": fields,
                "orderBy": "modifiedTime desc",
                "pageSize": "50",
            }
            if page_token:
                params["pageToken"] = page_token
            data = self._api_json(f"{DRIVE_FILES_URL}?{urllib.parse.urlencode(params)}")
            if process_events is not None:
                process_events()
            for item in data.get("files", []):
                if not isinstance(item, dict):
                    continue
                doc_id = str(item.get("id") or "")
                name = str(item.get("name") or "")
                mime_type = str(item.get("mimeType") or "")
                if not doc_id or not name or mime_type not in GOOGLE_PDF_MIME_TYPES:
                    continue
                documents.append(
                    CloudDocument(
                        id=doc_id,
                        name=name,
                        mime_type=mime_type,
                        modified_time=str(item.get("modifiedTime") or ""),
                        size=str(item.get("size") or ""),
                        thumbnail_link=str(item.get("thumbnailLink") or ""),
                        web_view_link=str(item.get("webViewLink") or ""),
                    )
                )
            page_token = str(data.get("nextPageToken") or "")
            if not page_token:
                return documents

    def ensure_documents_folder_id(self) -> str:
        folder_id = self.find_documents_folder_id()
        if folder_id:
            return folder_id
        return self.create_documents_folder()

    def find_documents_folder_id(self) -> str:
        query = (
            "trashed=false "
            f"and mimeType='{DRIVE_FOLDER_MIME_TYPE}' "
            f"and name='{self._drive_query_string(DRIVE_FOLDER_NAME)}'"
        )
        params = {
            "q": query,
            "fields": "files(id,name,modifiedTime)",
            "orderBy": "modifiedTime desc",
            "pageSize": "1",
        }
        data = self._api_json(f"{DRIVE_FILES_URL}?{urllib.parse.urlencode(params)}")
        files = data.get("files", [])
        if isinstance(files, list) and files:
            item = files[0]
            if isinstance(item, dict):
                return str(item.get("id") or "")
        return ""

    def create_documents_folder(self) -> str:
        metadata = {
            "name": DRIVE_FOLDER_NAME,
            "mimeType": DRIVE_FOLDER_MIME_TYPE,
        }
        data = self._api_json_request(
            DRIVE_FILES_URL + "?" + urllib.parse.urlencode({"fields": "id,name"}),
            method="POST",
            data=json.dumps(metadata).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        folder_id = str(data.get("id") or "")
        if not folder_id:
            raise GoogleDriveError(f"Could not create the {DRIVE_FOLDER_NAME} Drive folder.")
        return folder_id

    def upload_pdf(self, path: Path) -> CloudDocument:
        if not path.exists() or path.suffix.lower() != ".pdf":
            raise GoogleDriveError(f"Not a PDF file: {path}")

        folder_id = self.ensure_documents_folder_id()
        metadata = {
            "name": path.name,
            "mimeType": "application/pdf",
            "parents": [folder_id],
        }
        boundary = f"pdf-reader-{secrets.token_hex(16)}"
        file_bytes = path.read_bytes()
        body = b"".join(
            (
                f"--{boundary}\r\n".encode("utf-8"),
                b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
                json.dumps(metadata).encode("utf-8"),
                b"\r\n",
                f"--{boundary}\r\n".encode("utf-8"),
                b"Content-Type: application/pdf\r\n\r\n",
                file_bytes,
                b"\r\n",
                f"--{boundary}--\r\n".encode("utf-8"),
            )
        )
        fields = "id,name,mimeType,modifiedTime,size,thumbnailLink,webViewLink"
        data = self._api_json_request(
            DRIVE_UPLOAD_FILES_URL + "?" + urllib.parse.urlencode({"uploadType": "multipart", "fields": fields}),
            method="POST",
            data=body,
            headers={"Content-Type": f"multipart/related; boundary={boundary}"},
        )
        return CloudDocument(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or path.name),
            mime_type=str(data.get("mimeType") or "application/pdf"),
            modified_time=str(data.get("modifiedTime") or ""),
            size=str(data.get("size") or ""),
            thumbnail_link=str(data.get("thumbnailLink") or ""),
            web_view_link=str(data.get("webViewLink") or ""),
        )

    def thumbnail_bytes(self, document: CloudDocument) -> bytes:
        if not document.thumbnail_link:
            return b""
        return self._api_bytes(document.thumbnail_link)

    def download_document(self, document: CloudDocument) -> Path:
        out_path = self._cache_path(document)
        meta_path = out_path.with_suffix(out_path.suffix + ".json")
        cached_path = self.cached_document_path(document)
        if cached_path is not None:
            return cached_path

        if document.can_export_as_pdf:
            params = urllib.parse.urlencode({"mimeType": "application/pdf"})
            url = f"{DRIVE_FILES_URL}/{urllib.parse.quote(document.id)}/export?{params}"
        else:
            url = f"{DRIVE_FILES_URL}/{urllib.parse.quote(document.id)}?alt=media"

        data = self._api_bytes(url)
        if not data:
            raise GoogleDriveError(f"Google Drive returned an empty file for {document.name}.")
        out_path.write_bytes(data)
        meta_path.write_text(
            json.dumps(
                {
                    "id": document.id,
                    "name": document.name,
                    "mime_type": document.mime_type,
                    "modified_time": document.modified_time,
                    "size": document.size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out_path

    def cached_document_path(self, document: CloudDocument) -> Optional[Path]:
        out_path = self._cache_path(document)
        meta_path = out_path.with_suffix(out_path.suffix + ".json")
        if out_path.exists() and self._cached_metadata_matches(meta_path, document):
            return out_path
        return None

    def _api_json(self, url: str) -> dict:
        data = self._api_bytes(url)
        try:
            return json.loads(data.decode("utf-8"))
        except Exception as exc:
            raise GoogleDriveError("Google Drive returned invalid JSON.") from exc

    def _api_json_request(
        self,
        url: str,
        method: str,
        data: bytes,
        headers: Optional[Dict[str, str]] = None,
    ) -> dict:
        response = self._api_bytes_request(url, method=method, data=data, headers=headers)
        try:
            return json.loads(response.decode("utf-8"))
        except Exception as exc:
            raise GoogleDriveError("Google Drive returned invalid JSON.") from exc

    def _api_bytes(self, url: str) -> bytes:
        return self._api_bytes_request(url, method="GET")

    def _api_bytes_request(
        self,
        url: str,
        method: str = "GET",
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bytes:
        token = self._access_token()
        request_headers = {"Authorization": f"Bearer {token}"}
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=45, context=self._ssl_context) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in (401, 403) and "insufficient" in detail.lower():
                raise GoogleDriveError(
                    "Google Drive needs updated permissions. Sign out of Drive in the app, then sign in again."
                ) from exc
            raise GoogleDriveError(f"Google Drive request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise GoogleDriveError(f"Could not reach Google Drive: {exc.reason}") from exc

    def _access_token(self) -> str:
        access_token = str(self._token.get("access_token") or "")
        expires_at = float(self._token.get("expires_at") or 0)
        if access_token and expires_at > time.time() + 60:
            return access_token

        refresh_token = str(self._token.get("refresh_token") or "")
        if not refresh_token:
            raise GoogleDriveError("Please sign in to Google Drive.")

        client_id, client_secret = self._load_credentials()
        payload = {
            "client_id": client_id,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        if client_secret:
            payload["client_secret"] = client_secret
        token = self._post_token(payload)
        if "refresh_token" not in token:
            token["refresh_token"] = refresh_token
        self._store_token(token)
        return str(self._token.get("access_token") or "")

    def _post_token(self, payload: Dict[str, str]) -> dict:
        encoded = urllib.parse.urlencode(payload).encode("utf-8")
        request = urllib.request.Request(
            GOOGLE_TOKEN_URL,
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45, context=self._ssl_context) as response:
                token = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GoogleDriveError(f"Google OAuth token exchange failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise GoogleDriveError(f"Could not reach Google OAuth: {exc.reason}") from exc

        if "access_token" not in token:
            raise GoogleDriveError("Google OAuth did not return an access token.")
        return token

    @staticmethod
    def _create_ssl_context() -> ssl.SSLContext:
        if certifi is not None:
            try:
                return ssl.create_default_context(cafile=certifi.where())
            except Exception:
                pass
        return ssl.create_default_context()

    def _store_token(self, token: dict) -> None:
        expires_in = float(token.get("expires_in") or 3600)
        stored = dict(token)
        stored["expires_at"] = time.time() + expires_in
        self._token = stored
        self.token_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    def _load_token(self) -> Dict[str, object]:
        if not self.token_path.exists():
            return {}
        try:
            data = json.loads(self.token_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_credentials(self) -> Tuple[str, str]:
        env_client_id = os.environ.get("PDF_READER_GOOGLE_CLIENT_ID", "").strip()
        env_client_secret = os.environ.get("PDF_READER_GOOGLE_CLIENT_SECRET", "").strip()
        if env_client_id:
            return env_client_id, env_client_secret

        if self.credential_path and self.credential_path.exists():
            try:
                data = json.loads(self.credential_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise GoogleDriveError(f"Could not read Google OAuth credentials: {exc}") from exc
            installed = data.get("installed") if isinstance(data, dict) else None
            if isinstance(installed, dict):
                client_id = str(installed.get("client_id") or "").strip()
                client_secret = str(installed.get("client_secret") or "").strip()
                if client_id:
                    return client_id, client_secret
            client_id = str(data.get("client_id") or "").strip() if isinstance(data, dict) else ""
            client_secret = str(data.get("client_secret") or "").strip() if isinstance(data, dict) else ""
            if client_id:
                return client_id, client_secret

        if BUILT_IN_GOOGLE_CLIENT_ID:
            return BUILT_IN_GOOGLE_CLIENT_ID, BUILT_IN_GOOGLE_CLIENT_SECRET

        raise GoogleDriveError(
            "Google Drive OAuth is not configured. Set PDF_READER_GOOGLE_CLIENT_ID "
            "or add google_oauth_client.json next to the app."
        )

    @staticmethod
    def _code_verifier() -> str:
        return secrets.token_urlsafe(64)[:128]

    @staticmethod
    def _code_challenge(verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def _cache_path(self, document: CloudDocument) -> Path:
        stem = Path(document.name).stem if document.name.lower().endswith(".pdf") else document.name
        safe_stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" .") or "document"
        return self.cache_dir / f"{safe_stem}__{document.id}.pdf"

    @staticmethod
    def _cached_metadata_matches(meta_path: Path, document: CloudDocument) -> bool:
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            data.get("id") == document.id
            and data.get("modified_time") == document.modified_time
            and str(data.get("size") or "") == document.size
        )

    @staticmethod
    def _drive_query_string(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")
