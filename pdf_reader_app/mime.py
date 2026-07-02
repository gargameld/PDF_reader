from pathlib import Path
from typing import List

from .constants import SUPPORTED_DOCUMENT_BUILDER_EXTENSIONS

def pdf_paths_from_mime_data(mime_data) -> List[str]:
    if not mime_data.hasUrls():
        return []

    paths: List[str] = []
    seen = set()
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        path = url.toLocalFile()
        if not path or path in seen:
            continue
        file_path = Path(path)
        if file_path.is_file() and file_path.suffix.lower() == ".pdf":
            paths.append(path)
            seen.add(path)
    return paths


def builder_paths_from_mime_data(mime_data) -> List[str]:
    if not mime_data.hasUrls():
        return []

    paths: List[str] = []
    seen = set()
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        path = url.toLocalFile()
        if not path or path in seen:
            continue
        file_path = Path(path)
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_DOCUMENT_BUILDER_EXTENSIONS:
            paths.append(path)
            seen.add(path)
    return paths

