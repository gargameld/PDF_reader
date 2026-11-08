import hashlib
import json
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import fitz

try:
    import certifi
except Exception:
    certifi = None


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODELS = (
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-5.6-luna",
    "gpt-5.4",
    "gpt-5.6",
)
OPENAI_API_URL = "https://api.openai.com/v1"
AI_SETTINGS_FILENAME = "ai_settings.json"
AI_INDEX_DIRNAME = "ai_indexes"
AI_INDEX_VERSION = 1
AI_CHAT_HISTORY_VERSION = 1
MAX_CHUNK_CHARS = 2600
MIN_CHUNK_CHARS = 500
EMBEDDING_BATCH_SIZE = 12
EMBEDDING_BATCH_TOKEN_BUDGET = 12000
OPENAI_RETRY_ATTEMPTS = 4
EMBEDDING_TPM_BUDGET = 24000
EMBEDDING_TPM_WINDOW_SECONDS = 61.0
SEMANTIC_RESULT_LIMIT = 8
KEYWORD_RESULT_LIMIT = 18
SEMANTIC_CANDIDATE_LIMIT = 80
KEYWORD_CANDIDATE_LIMIT = 80
SECTION_CANDIDATE_LIMIT = 8
NEIGHBOR_CONTEXT_LIMIT = 4
ENUMERATION_MIN_LIMIT = 30
ENUMERATION_PATTERN_CANDIDATE_LIMIT = 120
DEFAULT_RETRIEVAL_LIMIT = 18

MODEL_PRICES_PER_1M = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "gpt-5.6-sol": {"input": 5.0, "output": 30.0},
    "gpt-5.6": {"input": 5.0, "output": 30.0},
    "gpt-5.6-terra": {"input": 2.5, "output": 15.0},
    "gpt-5.6-luna": {"input": 1.0, "output": 6.0},
    "gpt-5.4": {"input": 2.5, "output": 15.0},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}


@dataclass
class AIChunk:
    chunk_id: str
    page: int
    y: float
    text: str
    x: float = 0.0
    rect: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class AIReference:
    page: int
    y: float
    text: str
    x: float = 0.0
    rect: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class RetrievedChunk:
    label: str
    chunk: AIChunk
    score: float
    source: str = "semantic"


@dataclass
class TOCSectionMatch:
    level: int
    title: str
    start_page: int
    end_page: int
    score: float


@dataclass
class AIUsageTotals:
    total_usd: float = 0.0
    embedding_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0


class OpenAIRateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


_SESSION_USAGE = AIUsageTotals()
_EMBEDDING_TOKEN_EVENTS: List[Tuple[float, int]] = []


def load_ai_settings(base_dir: Path) -> dict:
    settings_path = base_dir / AI_SETTINGS_FILENAME
    if not settings_path.exists():
        return {}
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_ai_settings(base_dir: Path, settings: dict) -> None:
    settings_path = base_dir / AI_SETTINGS_FILENAME
    clean = {
        "api_key": str(settings.get("api_key") or "").strip(),
        "chat_model": str(settings.get("chat_model") or "").strip(),
        "embedding_model": str(settings.get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip()
        or DEFAULT_EMBEDDING_MODEL,
        "retrieval_limit": int(settings.get("retrieval_limit") or DEFAULT_RETRIEVAL_LIMIT),
    }
    settings_path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def openai_api_key(base_dir: Path) -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    return str(load_ai_settings(base_dir).get("api_key") or "").strip()


def _openai_json(api_key: str, endpoint: str, payload: dict, timeout: int = 90) -> dict:
    if not api_key:
        raise RuntimeError("Enter and save an OpenAI API key first.")

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OPENAI_API_URL}{endpoint}",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error_type = ""
        error_code = ""
        try:
            parsed = json.loads(detail)
            error = parsed.get("error", {}) if isinstance(parsed, dict) else {}
            message = error.get("message") or detail
            error_type = str(error.get("type") or "")
            error_code = str(error.get("code") or "")
        except Exception:
            message = detail
        if exc.code == 401 or error_type == "invalid_request_error" and error_code == "invalid_api_key":
            raise RuntimeError(
                "The OpenAI API key is invalid or was not accepted. "
                "Create a new key at https://platform.openai.com/api-keys, paste it into AI Chat, "
                "click Save Key, and try again."
            ) from exc
        if exc.code == 429 and error_code == "insufficient_quota":
            raise RuntimeError(
                "The OpenAI API key was accepted, but this account or project has no available API quota. "
                "Check billing, credits, usage limits, or the selected project at https://platform.openai.com/settings/organization/billing. "
                "After billing is active, try Scan Document again."
            ) from exc
        if exc.code == 429:
            raise OpenAIRateLimitError(
                "OpenAI rate limit reached. The API key is valid, but the project is sending requests too quickly "
                f"or has hit a temporary limit. Details: {message}",
                retry_after=_retry_after_seconds(message),
            ) from exc
        if exc.code == 403:
            raise RuntimeError(
                "The OpenAI API key was accepted, but this project does not have access to the requested model "
                f"or operation. Details: {message}"
            ) from exc
        raise RuntimeError(f"OpenAI API error: {message}") from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
            raise RuntimeError(
                "Could not verify the OpenAI API TLS certificate. This is a certificate/network trust problem, "
                "not a wrong API key. Make sure the app dependencies are installed with certifi, check any VPN/proxy, "
                "and try again."
            ) from exc
        raise RuntimeError(f"Could not reach OpenAI API: {reason}") from exc


def _ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def session_usage_totals() -> AIUsageTotals:
    return AIUsageTotals(
        total_usd=_SESSION_USAGE.total_usd,
        embedding_tokens=_SESSION_USAGE.embedding_tokens,
        input_tokens=_SESSION_USAGE.input_tokens,
        output_tokens=_SESSION_USAGE.output_tokens,
        request_count=_SESSION_USAGE.request_count,
    )


def _record_usage(model: str, endpoint: str, usage: object) -> None:
    if not isinstance(usage, dict):
        return
    prices = MODEL_PRICES_PER_1M.get(model, {})
    input_price = float(prices.get("input") or 0.0)
    output_price = float(prices.get("output") or 0.0)

    if endpoint == "/embeddings":
        tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        _SESSION_USAGE.embedding_tokens += tokens
        _SESSION_USAGE.input_tokens += tokens
        _SESSION_USAGE.total_usd += tokens * input_price / 1_000_000
        _SESSION_USAGE.request_count += 1
        return

    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    _SESSION_USAGE.input_tokens += input_tokens
    _SESSION_USAGE.output_tokens += output_tokens
    _SESSION_USAGE.total_usd += (
        input_tokens * input_price / 1_000_000
        + output_tokens * output_price / 1_000_000
    )
    _SESSION_USAGE.request_count += 1


def _retry_after_seconds(message: str) -> Optional[float]:
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if not match:
        return None
    try:
        return max(0.0, float(match.group(1)))
    except ValueError:
        return None


def _estimated_tokens(text: str) -> int:
    # Good enough for TPM pacing without adding a tokenizer dependency.
    return max(1, math.ceil(len(text) / 4))


def _embedding_batches(chunks: List[AIChunk]) -> List[List[AIChunk]]:
    batches: List[List[AIChunk]] = []
    current: List[AIChunk] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = _estimated_tokens(chunk.text)
        if (
            current
            and (
                len(current) >= EMBEDDING_BATCH_SIZE
                or current_tokens + chunk_tokens > EMBEDDING_BATCH_TOKEN_BUDGET
            )
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += chunk_tokens

    if current:
        batches.append(current)
    return batches


def _embedding_batch_tokens(texts: List[str]) -> int:
    return sum(_estimated_tokens(text) for text in texts)


def _pace_embedding_tokens(tokens: int, progress: Optional[Callable[[str], None]] = None) -> None:
    now = time.monotonic()
    cutoff = now - EMBEDDING_TPM_WINDOW_SECONDS
    recent_events = [
        (timestamp, token_count)
        for timestamp, token_count in _EMBEDDING_TOKEN_EVENTS
        if timestamp >= cutoff
    ]
    del _EMBEDDING_TOKEN_EVENTS[:]
    _EMBEDDING_TOKEN_EVENTS.extend(recent_events)

    used = sum(token_count for _timestamp, token_count in _EMBEDDING_TOKEN_EVENTS)
    if used + tokens > EMBEDDING_TPM_BUDGET and _EMBEDDING_TOKEN_EVENTS:
        oldest = min(timestamp for timestamp, _token_count in _EMBEDDING_TOKEN_EVENTS)
        delay = max(1.0, oldest + EMBEDDING_TPM_WINDOW_SECONDS - now + 1.0)
        if progress:
            progress(
                f"Embedding token limit guard: waiting {delay:.0f}s "
                f"({used:,}/{EMBEDDING_TPM_BUDGET:,} tokens used this minute)..."
            )
        time.sleep(delay)
        _pace_embedding_tokens(tokens, progress=progress)
        return

    _EMBEDDING_TOKEN_EVENTS.append((time.monotonic(), tokens))


def create_embeddings(api_key: str, model: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    payload = {"model": model, "input": texts}
    data = _openai_json(api_key, "/embeddings", payload, timeout=120)
    _record_usage(model, "/embeddings", data.get("usage"))
    rows = sorted(data.get("data", []), key=lambda row: int(row.get("index", 0)))
    embeddings = [row.get("embedding") for row in rows]
    if len(embeddings) != len(texts) or any(not isinstance(item, list) for item in embeddings):
        raise RuntimeError("OpenAI returned an unexpected embeddings response.")
    return embeddings


def create_embeddings_with_retry(
    api_key: str,
    model: str,
    texts: List[str],
    progress: Optional[Callable[[str], None]] = None,
) -> List[List[float]]:
    _pace_embedding_tokens(_embedding_batch_tokens(texts), progress=progress)
    for attempt in range(1, OPENAI_RETRY_ATTEMPTS + 1):
        try:
            return create_embeddings(api_key, model, texts)
        except OpenAIRateLimitError as exc:
            if attempt >= OPENAI_RETRY_ATTEMPTS:
                raise
            delay = exc.retry_after if exc.retry_after is not None else min(60.0, 8.0 * attempt)
            delay = max(1.0, delay + 2.0)
            if progress:
                progress(f"Embedding rate limit reached. Waiting {delay:.0f}s before retry {attempt + 1}/{OPENAI_RETRY_ATTEMPTS}...")
            time.sleep(delay)
            _pace_embedding_tokens(_embedding_batch_tokens(texts), progress=progress)
    raise RuntimeError("Could not create embeddings after retrying rate limits.")


def create_answer(
    api_key: str,
    model: str,
    question: str,
    retrieved: List[RetrievedChunk],
    page_ranges: Optional[List[Tuple[int, int]]] = None,
) -> str:
    excerpts = []
    for item in retrieved:
        chunk = item.chunk
        source_note = f" ({item.source})" if item.source else ""
        excerpts.append(
            f"[{item.label}] Page {chunk.page}, vertical position {chunk.y:.0f}{source_note}\n"
            f"{chunk.text.strip()}"
        )

    system_prompt = (
        "You answer questions about a PDF using only the provided excerpts. "
        "If the excerpts do not contain enough information, say that clearly. "
        "Preserve mathematical notation, formulas, Greek letters, operators, and symbols exactly when they appear. "
        "When writing math, use standard LaTeX notation such as \\( ... \\), \\[ ... \\], fractions, sums, "
        "integrals, vectors, matrices, and subscripts/superscripts as needed. "
        "For simple tuples or encodings, prefer readable symbols like ⟨φ, k⟩ instead of raw \\langle ... \\rangle text. "
        "Do not wrap ordinary language/problem names like 3SAT, CLIQUE, VERTEX-COVER, HAMPATH, or VC in LaTeX commands; write them as plain text. "
        "Put important equations on their own lines inside \\[ ... \\] so the app can display them clearly. "
        "Cite every document-based claim with one or more excerpt citations in the exact form [[C1]], [[C2]], etc. "
        "Only cite labels that appear in the excerpts. "
        "Some excerpts are exact keyword matches from the document; treat them as important even if they are short or definition-like. "
        "If an excerpt is marked as a PDF bookmark/section match, treat it as table-of-contents structure for locating the relevant section. "
        "For list/enumeration questions, collect every named language/problem found in the excerpts before writing the final answer. "
        "For location questions like 'where is X defined?', answer with the page and nearby section when a definition-like excerpt is available."
    )
    page_scope = format_page_ranges(page_ranges)
    scope_text = f"Only answer from PDF pages {page_scope}.\n\n" if page_scope else ""
    user_prompt = (
        f"{scope_text}"
        f"Question:\n{question.strip()}\n\n"
        "Relevant PDF excerpts:\n\n"
        + "\n\n---\n\n".join(excerpts)
    )
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    data = _openai_json(api_key, "/responses", payload, timeout=180)
    _record_usage(model, "/responses", data.get("usage"))
    return _extract_response_text(data).strip()


def _extract_response_text(data: dict) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    parts: List[str] = []
    for output in data.get("output", []) or []:
        for content in output.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "\n".join(parts)
    raise RuntimeError("OpenAI returned an unexpected answer response.")


def document_fingerprint(path: str, embedding_model: str) -> dict:
    file_path = Path(path)
    stat = file_path.stat()
    pages = 0
    first_page_rect = ""
    try:
        doc = fitz.open(path)
        try:
            pages = len(doc)
            if pages:
                rect = doc.load_page(0).rect
                first_page_rect = f"{rect.width:.2f}x{rect.height:.2f}"
        finally:
            doc.close()
    except Exception:
        pass
    return {
        "index_version": AI_INDEX_VERSION,
        "path": str(file_path.resolve()),
        "name": file_path.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "pages": pages,
        "first_page_rect": first_page_rect,
        "embedding_model": embedding_model,
    }


def index_cache_path(base_dir: Path, pdf_path: str, embedding_model: str) -> Path:
    fingerprint = document_fingerprint(pdf_path, embedding_model)
    key = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return base_dir / AI_INDEX_DIRNAME / f"{digest}.json"


def chat_history_cache_path(base_dir: Path, pdf_path: str) -> Path:
    try:
        key = str(Path(pdf_path).resolve())
    except Exception:
        key = pdf_path
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return base_dir / AI_INDEX_DIRNAME / f"{digest}.chat.json"


def load_chat_history(base_dir: Path, pdf_path: str) -> List[dict]:
    path = chat_history_cache_path(base_dir, pdf_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict) or data.get("version") != AI_CHAT_HISTORY_VERSION:
        return []
    entries = data.get("entries")
    return entries if isinstance(entries, list) else []


def save_chat_history(base_dir: Path, pdf_path: str, entries: List[dict]) -> None:
    path = chat_history_cache_path(base_dir, pdf_path)
    path.parent.mkdir(exist_ok=True)
    clean_entries = [entry for entry in entries if isinstance(entry, dict)][-200:]
    payload = {
        "version": AI_CHAT_HISTORY_VERSION,
        "document": str(Path(pdf_path).resolve()) if Path(pdf_path).exists() else pdf_path,
        "updated_at": int(time.time()),
        "entries": clean_entries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_chat_history(base_dir: Path, pdf_path: str, entry: dict) -> None:
    entries = load_chat_history(base_dir, pdf_path)
    entries.append(entry)
    save_chat_history(base_dir, pdf_path, entries)


def clear_chat_history(base_dir: Path, pdf_path: str) -> None:
    path = chat_history_cache_path(base_dir, pdf_path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def has_current_index(base_dir: Path, pdf_path: str, embedding_model: str) -> bool:
    return index_cache_path(base_dir, pdf_path, embedding_model).exists()


def build_document_index(
    base_dir: Path,
    pdf_path: str,
    api_key: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    page_ranges: Optional[List[Tuple[int, int]]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    cache_path = index_cache_path(base_dir, pdf_path, embedding_model)
    cache_path.parent.mkdir(exist_ok=True)
    fingerprint = document_fingerprint(pdf_path, embedding_model)

    if progress:
        if page_ranges:
            progress(f"Extracting text chunks from {format_page_ranges(page_ranges)}...")
        else:
            progress("Extracting text chunks...")
    chunks = extract_chunks(pdf_path, page_ranges=page_ranges)
    if not chunks:
        raise RuntimeError("No extractable text was found. This PDF may need OCR first.")

    embeddings: List[List[float]] = []
    batches = _embedding_batches(chunks)
    processed = 0
    for batch_index, batch in enumerate(batches, start=1):
        if progress:
            progress(
                f"Embedding chunks {processed + 1}-{processed + len(batch)} / {len(chunks)} "
                f"(batch {batch_index}/{len(batches)})..."
            )
        embeddings.extend(
            create_embeddings_with_retry(
                api_key,
                embedding_model,
                [chunk.text for chunk in batch],
                progress=progress,
            )
        )
        processed += len(batch)
        if progress:
            progress(f"Embedded {processed} / {len(chunks)} chunk(s).")

    new_rows = [
        {
            "id": chunk.chunk_id,
            "page": chunk.page,
            "x": chunk.x,
            "y": chunk.y,
            "rect": list(chunk.rect),
            "text": chunk.text,
            "embedding": embedding,
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    if page_ranges and cache_path.exists():
        try:
            existing_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            existing_payload = {}
        existing_rows = existing_payload.get("chunks", []) if isinstance(existing_payload, dict) else []
        scanned_pages = _pages_from_ranges(page_ranges, int(fingerprint.get("pages") or 0))
        merged_rows = [
            row
            for row in existing_rows
            if isinstance(row, dict) and int(row.get("page") or 0) not in scanned_pages
        ] + new_rows
    else:
        merged_rows = new_rows

    merged_rows.sort(key=lambda row: (int(row.get("page") or 0), str(row.get("id") or "")))
    indexed_pages = {
        int(row.get("page") or 0)
        for row in merged_rows
        if isinstance(row, dict) and int(row.get("page") or 0) > 0
    }

    payload = {
        "fingerprint": fingerprint,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "indexed_pages": sorted(indexed_pages),
        "chunks": merged_rows,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {
        "path": str(cache_path),
        "chunk_count": len(merged_rows),
        "added_chunk_count": len(chunks),
        "page_count": fingerprint.get("pages", 0),
        "text_page_count": len(indexed_pages),
        "scanned_text_page_count": len({chunk.page for chunk in chunks}),
        "page_ranges": format_page_ranges(page_ranges) if page_ranges else "",
    }


def load_document_index(base_dir: Path, pdf_path: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> dict:
    cache_path = index_cache_path(base_dir, pdf_path, embedding_model)
    if not cache_path.exists():
        raise RuntimeError("Scan this document for AI chunks first.")
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not load AI index: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
        raise RuntimeError("The AI index file is invalid. Scan the document again.")
    return data


def extract_chunks(pdf_path: str, page_ranges: Optional[List[Tuple[int, int]]] = None) -> List[AIChunk]:
    doc = fitz.open(pdf_path)
    chunks: List[AIChunk] = []
    try:
        page_numbers = _page_numbers_for_doc(len(doc), page_ranges)
        for page_number in page_numbers:
            page = doc.load_page(page_number - 1)
            blocks = _page_text_blocks(page)
            chunks.extend(_chunks_for_page(page_number, blocks))
    finally:
        doc.close()
    return chunks


def _page_numbers_for_doc(total_pages: int, page_ranges: Optional[List[Tuple[int, int]]]) -> List[int]:
    if not page_ranges:
        return list(range(1, total_pages + 1))
    return sorted(_pages_from_ranges(page_ranges, total_pages))


def _pages_from_ranges(page_ranges: List[Tuple[int, int]], total_pages: int) -> set:
    pages = set()
    for start, end in page_ranges:
        lo = max(1, min(int(start), int(end)))
        hi = min(total_pages, max(int(start), int(end)))
        for page in range(lo, hi + 1):
            pages.add(page)
    return pages


def format_page_ranges(page_ranges: Optional[List[Tuple[int, int]]]) -> str:
    if not page_ranges:
        return ""
    parts = []
    for start, end in page_ranges:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
    return ", ".join(parts)


def _page_text_blocks(page: fitz.Page) -> List[Tuple[float, float, Tuple[float, float, float, float], str]]:
    blocks: List[Tuple[float, float, Tuple[float, float, float, float], str]] = []
    try:
        raw_blocks = page.get_text("blocks", sort=True)
    except Exception:
        raw_blocks = []
    for block in raw_blocks:
        if len(block) < 5:
            continue
        text = _clean_text(str(block[4] or ""))
        if text:
            rect = (float(block[0]), float(block[1]), float(block[2]), float(block[3]))
            blocks.append((rect[0], rect[1], rect, text))

    if blocks:
        return blocks

    fallback = _clean_text(page.get_text("text", sort=True) or "")
    if not fallback:
        return []
    rect = page.rect
    return [(0.0, 0.0, (0.0, 0.0, float(rect.width), float(rect.height)), fallback)]


def _chunks_for_page(
    page_number: int,
    blocks: List[Tuple[float, float, Tuple[float, float, float, float], str]],
) -> List[AIChunk]:
    chunks: List[AIChunk] = []
    current_parts: List[str] = []
    current_x = 0.0
    current_y = 0.0
    current_rect: Optional[Tuple[float, float, float, float]] = None
    current_len = 0

    for x, y, rect, text in blocks:
        next_len = current_len + len(text) + 2
        if current_parts and current_len >= MIN_CHUNK_CHARS and next_len > MAX_CHUNK_CHARS:
            chunks.append(_make_chunk(page_number, len(chunks), current_x, current_y, current_rect, current_parts))
            current_parts = []
            current_len = 0
            current_rect = None
        if not current_parts:
            current_x = x
            current_y = y
            current_rect = rect
        else:
            current_rect = _union_rect(current_rect, rect)
        current_parts.append(text)
        current_len += len(text) + 2

    if current_parts:
        chunks.append(_make_chunk(page_number, len(chunks), current_x, current_y, current_rect, current_parts))
    return chunks


def _union_rect(
    first: Optional[Tuple[float, float, float, float]],
    second: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    if first is None:
        return second
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _make_chunk(
    page_number: int,
    page_chunk_index: int,
    x: float,
    y: float,
    rect: Optional[Tuple[float, float, float, float]],
    parts: List[str],
) -> AIChunk:
    clean_rect = rect or (max(0.0, float(x)), max(0.0, float(y)), max(0.0, float(x)), max(0.0, float(y)))
    return AIChunk(
        chunk_id=f"p{page_number}-{page_chunk_index + 1}",
        page=page_number,
        x=max(0.0, float(clean_rect[0])),
        y=max(0.0, float(y)),
        rect=tuple(float(value) for value in clean_rect),
        text="\n\n".join(parts).strip(),
    )


def _clean_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\x00", "").splitlines()]
    out: List[str] = []
    blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not blank and out:
                out.append("")
            blank = True
            continue
        out.append(stripped)
        blank = False
    return "\n".join(out).strip()


def retrieve_chunks(
    base_dir: Path,
    pdf_path: str,
    api_key: str,
    question: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
    page_ranges: Optional[List[Tuple[int, int]]] = None,
) -> List[RetrievedChunk]:
    enumeration_context = _is_enumeration_question(question)
    if _needs_definition_context(question):
        limit = max(limit, 4)
    if enumeration_context:
        limit = max(limit, ENUMERATION_MIN_LIMIT)
    data = load_document_index(base_dir, pdf_path, embedding_model)
    rows = [row for row in data.get("chunks", []) if isinstance(row, dict)]
    total_pages = int((data.get("fingerprint") or {}).get("pages") or 0)
    if page_ranges:
        allowed_pages = _pages_from_ranges(page_ranges, total_pages)
        rows = [row for row in rows if int(row.get("page") or 0) in allowed_pages]
        if not rows:
            raise RuntimeError(
                f"No scanned AI chunks were found in pages {format_page_ranges(page_ranges)}. "
                "Scan that page range first, or clear the question page range."
            )
    section_matches = _toc_section_matches(pdf_path, question, total_pages, page_ranges)
    section_scored = _toc_section_scored_rows(section_matches)
    question_embedding = create_embeddings_with_retry(api_key, embedding_model, [question.strip()])[0]

    scored: List[Tuple[float, dict, str]] = []
    for row in rows:
        embedding = row.get("embedding")
        if not isinstance(embedding, list):
            continue
        page = int(row.get("page") or 0)
        semantic_score = _cosine_similarity(question_embedding, embedding)
        scored.append(
            (
                (semantic_score * 10.0) + _toc_section_page_boost(page, section_matches),
                row,
                "semantic",
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    keyword_scored = _keyword_scored_rows(question, rows)
    keyword_scored = _boost_rows_in_toc_sections(keyword_scored, section_matches)
    pattern_scored = _enumeration_pattern_scored_rows(question, rows) if enumeration_context else []
    pattern_scored = _boost_rows_in_toc_sections(pattern_scored, section_matches)

    if _needs_definition_context(question) or enumeration_context:
        source_groups = (
            (section_scored[:SECTION_CANDIDATE_LIMIT], 1.25),
            (pattern_scored[:ENUMERATION_PATTERN_CANDIDATE_LIMIT], 1.20),
            (keyword_scored[:KEYWORD_CANDIDATE_LIMIT], 1.10),
            (scored[:SEMANTIC_CANDIDATE_LIMIT], 0.75),
        )
    else:
        source_groups = (
            (section_scored[:SECTION_CANDIDATE_LIMIT], 1.00),
            (scored[:SEMANTIC_CANDIDATE_LIMIT], 1.00),
            (keyword_scored[:KEYWORD_CANDIDATE_LIMIT], 0.90),
        )

    selected = _merge_scored_sources(source_groups)
    selected = _add_neighbor_context(selected, rows, limit, generous=enumeration_context)

    retrieved: List[RetrievedChunk] = []
    for index, (score, row, source) in enumerate(selected[:limit], start=1):
        retrieved.append(
            RetrievedChunk(
                label=f"C{index}",
                chunk=AIChunk(
                    chunk_id=str(row.get("id") or f"chunk-{index}"),
                    page=int(row.get("page") or 1),
                    x=float(row.get("x") or 0.0),
                    y=float(row.get("y") or 0.0),
                    rect=_row_rect(row),
                    text=str(row.get("text") or ""),
                ),
                score=score,
                source=source,
            )
        )
    return retrieved


def _merge_scored_sources(
    source_groups: Tuple[Tuple[List[Tuple[float, dict, str]], float], ...],
) -> List[Tuple[float, dict, str]]:
    merged: Dict[str, Tuple[float, dict, List[str]]] = {}
    for rows, weight in source_groups:
        for score, row, source in rows:
            row_id = str(row.get("id") or "")
            if not row_id:
                continue
            weighted_score = float(score) * float(weight)
            current = merged.get(row_id)
            if current is None:
                merged[row_id] = (weighted_score, row, [source])
                continue
            current_score, current_row, sources = current
            if source not in sources:
                sources.append(source)
            # A row found by multiple cheap methods is more trustworthy than a
            # row found by only one method, but avoid letting duplicates explode.
            merged[row_id] = (current_score + min(weighted_score, 12.0), current_row, sources)

    out = [(score, row, " + ".join(sources)) for score, row, sources in merged.values()]
    out.sort(key=lambda item: (item[0], -int(item[1].get("page") or 0)), reverse=True)
    return out


def _add_neighbor_context(
    selected: List[Tuple[float, dict, str]],
    rows: List[dict],
    limit: int,
    generous: bool = False,
) -> List[Tuple[float, dict, str]]:
    if limit <= 2 or not selected:
        return selected[:limit]

    neighbor_budget = min(NEIGHBOR_CONTEXT_LIMIT * (2 if generous else 1), max(1, limit // 3 if generous else limit // 4))
    primary_limit = max(1, limit - neighbor_budget)
    row_by_id = {str(row.get("id") or ""): row for row in rows}
    with_neighbors: List[Tuple[float, dict, str]] = []
    seen_ids = set()
    neighbors_added = 0

    for score, row, source in selected:
        row_id = str(row.get("id") or "")
        if row_id in seen_ids:
            continue
        with_neighbors.append((score, row, source))
        seen_ids.add(row_id)

        if len(with_neighbors) >= primary_limit:
            continue
        if source == "bookmark section match" or row_id.startswith("toc-"):
            continue
        for neighbor_id in _neighbor_chunk_ids(row_id):
            if neighbors_added >= neighbor_budget:
                break
            if neighbor_id in seen_ids:
                continue
            neighbor = row_by_id.get(neighbor_id)
            if neighbor is None:
                continue
            with_neighbors.append((max(0.0, score - 2.0), neighbor, "neighbor context"))
            seen_ids.add(neighbor_id)
            neighbors_added += 1
        if len(with_neighbors) >= limit:
            break

    for score, row, source in selected:
        if len(with_neighbors) >= limit:
            break
        row_id = str(row.get("id") or "")
        if row_id in seen_ids:
            continue
        with_neighbors.append((score, row, source))
        seen_ids.add(row_id)
    return with_neighbors[:limit]


def _neighbor_chunk_ids(chunk_id: str) -> List[str]:
    match = re.fullmatch(r"p(\d+)-(\d+)", chunk_id)
    if not match:
        return []
    page = int(match.group(1))
    index = int(match.group(2))
    out = []
    if index > 1:
        out.append(f"p{page}-{index - 1}")
    out.append(f"p{page}-{index + 1}")
    out.append(f"p{page + 1}-1")
    if page > 1:
        out.append(f"p{page - 1}-1")
    return out


def _row_rect(row: dict) -> Tuple[float, float, float, float]:
    raw = row.get("rect")
    if isinstance(raw, list) and len(raw) >= 4:
        try:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        except Exception:
            pass
    x = float(row.get("x") or 0.0)
    y = float(row.get("y") or 0.0)
    return (x, y, x, y)


def _toc_section_matches(
    pdf_path: str,
    question: str,
    total_pages: int,
    page_ranges: Optional[List[Tuple[int, int]]] = None,
) -> List[TOCSectionMatch]:
    if total_pages <= 0:
        return []
    try:
        doc = fitz.open(pdf_path)
        try:
            toc = doc.get_toc(simple=True)
        finally:
            doc.close()
    except Exception:
        return []

    normalized: List[Tuple[int, str, int]] = []
    for entry in toc:
        if len(entry) < 3:
            continue
        try:
            level = max(1, int(entry[0]))
            title = str(entry[1]).strip()
            page = max(1, min(int(entry[2]), total_pages))
        except Exception:
            continue
        if title:
            normalized.append((level, title, page))

    if not normalized:
        return []

    allowed_pages = _pages_from_ranges(page_ranges, total_pages) if page_ranges else None
    matches: List[TOCSectionMatch] = []
    for index, (level, title, start_page) in enumerate(normalized):
        end_page = total_pages
        for next_level, _next_title, next_page in normalized[index + 1 :]:
            if next_level <= level:
                end_page = max(start_page, next_page - 1)
                break
        if allowed_pages and not any(page in allowed_pages for page in range(start_page, end_page + 1)):
            continue
        score = _toc_title_score(question, title)
        if score > 0:
            matches.append(TOCSectionMatch(level, title, start_page, end_page, score))

    matches.sort(key=lambda item: (item.score, item.level, -item.start_page), reverse=True)
    return matches


def _toc_title_score(question: str, title: str) -> float:
    terms = _keyword_terms(question)
    if not terms:
        return 0.0

    normalized_title = _normalize_keyword_text(title)
    normalized_question = _normalize_keyword_text(question)
    score = 0.0
    matched = 0
    for term in terms:
        is_acronym = any(ch.isalpha() for ch in term) and term.upper() == term
        if _has_keyword_term(normalized_title, term):
            matched += 1
            score += 12.0 if is_acronym else 5.0

    if matched == 0:
        return 0.0

    if matched == len(terms):
        score += 5.0 + matched
    if _needs_definition_context(question):
        score += 8.0
    if re.search(r"\bclass\b", normalized_question, re.IGNORECASE) and re.search(r"\bclass\b", normalized_title, re.IGNORECASE):
        score += 6.0
    if _asks_for_plain_np_definition(question) and re.search(
        r"\b(?:NPC|NP\s+complete|NP\s+completeness|NP\s+hard)\b",
        normalized_title,
        re.IGNORECASE,
    ):
        score -= 10.0
    return max(0.0, score)


def _toc_section_scored_rows(matches: List[TOCSectionMatch]) -> List[Tuple[float, dict, str]]:
    rows: List[Tuple[float, dict, str]] = []
    for index, match in enumerate(matches):
        page_range = (
            str(match.start_page)
            if match.start_page == match.end_page
            else f"{match.start_page}-{match.end_page}"
        )
        rows.append(
            (
                match.score,
                {
                    "id": f"toc-{index}-{match.start_page}-{match.end_page}",
                    "page": match.start_page,
                    "x": 0.0,
                    "y": 0.0,
                    "rect": [0.0, 0.0, 0.0, 0.0],
                    "text": (
                        f"PDF bookmark/section match: {match.title}\n"
                        f"This section starts on page {match.start_page} and spans pages {page_range}."
                    ),
                },
                "bookmark section match",
            )
        )
    return rows


def _toc_section_page_boost(page: int, matches: List[TOCSectionMatch]) -> float:
    boost = 0.0
    for match in matches[:4]:
        if match.start_page <= page <= match.end_page:
            boost = max(boost, min(7.0, match.score / 4.0))
    return boost


def _boost_rows_in_toc_sections(
    scored_rows: List[Tuple[float, dict, str]],
    matches: List[TOCSectionMatch],
) -> List[Tuple[float, dict, str]]:
    if not matches:
        return scored_rows
    boosted: List[Tuple[float, dict, str]] = []
    for score, row, source in scored_rows:
        page = int(row.get("page") or 0)
        section_bonus = 0.0
        for match in matches[:4]:
            if match.start_page <= page <= match.end_page:
                section_bonus = max(section_bonus, min(8.0, match.score / 3.0))
        if section_bonus:
            boosted.append((score + section_bonus, row, f"{source} + bookmark section"))
        else:
            boosted.append((score, row, source))
    boosted.sort(key=lambda item: (item[0], -int(item[1].get("page") or 0)), reverse=True)
    return boosted


def _enumeration_pattern_scored_rows(question: str, rows: List[dict]) -> List[Tuple[float, dict, str]]:
    terms = _keyword_terms(question)
    aliases = _enumeration_alias_terms(question)
    scored: List[Tuple[float, dict, str]] = []
    for row in rows:
        text = str(row.get("text") or "")
        normalized = _normalize_keyword_text(text)
        lower = normalized.lower()
        score = 0.0

        for alias in aliases:
            if _has_keyword_phrase(normalized, alias) or _has_keyword_term(normalized, alias):
                score += 6.0

        matched_terms = 0
        for term in terms:
            if _has_keyword_term(normalized, term):
                matched_terms += 1
        if terms:
            score += 4.0 * (matched_terms / max(1, len(terms)))

        # High-recall textbook-language patterns. These are intentionally broad
        # but only active for enumeration/list questions.
        if re.search(r"\{[^{}\n]{2,220}\}", text):
            score += 7.0
        if re.search(r"\bL\s*\(\s*G\s*\d*\s*\)|\bG\s*\d+\b", text):
            score += 5.0
        if re.search(r"\b(?:language|languages)\s+(?:of|generated|recognized|accepted|described)\b", lower):
            score += 5.0
        if re.search(r"\b(?:grammar|cfg|context\s+free|contextfree|cfl|pushdown|pda)\b", lower):
            score += 6.0
        if re.search(r"\b(?:regular\s+languages?|closure|closed\s+under|union|complement|reverse|substring)\b", lower):
            score += 3.0
        if re.search(r"\b(?:example|exercise|corollary|theorem|lemma)\b", lower):
            score += 1.5
        if re.search(r"\b(?:is|are|is\s+also|shown\s+to\s+be|prove(?:d)?)\s+(?:a\s+)?(?:context\s+free|regular|np\s+complete|decidable)\b", lower):
            score += 5.0
        if re.search(r"\b(?:is|are)\s+(?:not\s+)?(?:context\s+free|regular|decidable|recognizable)\b", lower):
            score += 3.0

        # Avoid flooding a CFL list with unrelated NP-completeness pages unless
        # the question explicitly asks about NP/NPC.
        normalized_question = _normalize_keyword_text(question).lower()
        if "context" in normalized_question and re.search(r"\b(?:np\s+complete|npc|np\s+hard)\b", lower):
            score -= 6.0

        if score > 0:
            scored.append((score, row, "pattern match"))

    scored.sort(key=lambda item: (item[0], -int(item[1].get("page") or 0)), reverse=True)
    return scored


def _enumeration_alias_terms(question: str) -> List[str]:
    normalized = _normalize_keyword_text(question).lower()
    aliases: List[str] = []
    if any(marker in normalized for marker in ("context free", "context-free", "cfg", "cfl", "grammar", "pushdown", "pda")):
        aliases.extend(
            [
                "context free",
                "context-free",
                "CFL",
                "CFG",
                "grammar",
                "generated by",
                "language of",
                "pushdown automaton",
                "PDA",
                "regular language",
                "closure",
                "union",
            ]
        )
    if any(marker in normalized for marker in ("np complete", "np-complete", "npc", "np completeness", "np-completeness")):
        aliases.extend(["NP complete", "NP-complete", "NPC", "reduction", "VERTEX COVER", "3SAT", "SAT"])
    if "regular" in normalized:
        aliases.extend(["regular", "DFA", "NFA", "finite automaton", "regular expression"])
    return aliases


def _keyword_scored_rows(question: str, rows: List[dict]) -> List[Tuple[float, dict, str]]:
    terms = _keyword_terms(question)
    if not terms:
        return []

    definition_context = _needs_definition_context(question)
    asks_plain_np = _asks_for_plain_np_definition(question)
    phrases = _keyword_phrases(question)
    scored: List[Tuple[float, dict, str]] = []
    for row in rows:
        text = str(row.get("text") or "")
        normalized = _normalize_keyword_text(text)
        normalized_lower = normalized.lower()
        score = 0.0
        matched_terms = set()
        for term in terms:
            term_lower = term.lower()
            is_acronym = any(ch.isalpha() for ch in term) and term.upper() == term
            has_term = _has_keyword_term(normalized, term)
            if has_term:
                matched_terms.add(term_lower)
                # Acronyms and uncommon terms should punch through embedding misses.
                score += 3.0 if is_acronym else 1.0
                if definition_context:
                    if re.search(rf"\b{re.escape(term)}\b\s*(?:=|is|are|contains|consists|denote|denotes)", normalized, re.IGNORECASE):
                        score += 8.0
                    if is_acronym and re.search(rf"\bclass\s+{re.escape(term)}\b|\b{re.escape(term)}\s+class\b", normalized, re.IGNORECASE):
                        score += 9.0
                    if is_acronym and re.search(rf"\b{re.escape(term)}\b.*\bclass\b|\bclass\b.*\b{re.escape(term)}\b", normalized, re.IGNORECASE):
                        score += 3.0
                    if re.search(r"\b(?:defined|definition|define|denote|denotes|let)\b", normalized, re.IGNORECASE):
                        score += 4.0
                if term_lower in {"np complete", "np completeness", "npc"}:
                    score += 4.0
            if re.search(r"\b(?:is|are|show|prove|proved|shown)\s+NP\s+complete\b", normalized, re.IGNORECASE):
                score += 4.0
            if re.search(r"\b(?:VERTEX\s+COVER|VERTEXCOVER|VC)\b", normalized, re.IGNORECASE):
                score += 5.0
            compact_term = term_lower.replace(" ", "")
            if len(compact_term) >= 3 and not is_acronym and compact_term in normalized_lower.replace(" ", ""):
                score += 0.75
        for phrase in phrases:
            if _has_keyword_phrase(normalized, phrase):
                score += 6.0
        if matched_terms:
            coverage = len(matched_terms) / max(1, len({term.lower() for term in terms}))
            score += 5.0 * coverage
            if coverage >= 0.999:
                score += 4.0
            elif len(terms) >= 3 and coverage <= 0.34 and not definition_context:
                score *= 0.75
        if asks_plain_np:
            if re.search(r"\b(?:NPC|NP\s+complete|NP\s+completeness|NP\s+hard)\b", normalized, re.IGNORECASE):
                score -= 7.0
            if re.search(r"\b(?:the\s+)?class\s+NP\b|\bNP\b\s+(?:is|contains|consists|denote|denotes)", normalized, re.IGNORECASE):
                score += 12.0
        if score > 0:
            source = "definition match" if definition_context else "keyword match"
            scored.append((score, row, source))

    scored.sort(key=lambda item: (item[0], -int(item[1].get("page") or 0)), reverse=True)
    return scored


def _keyword_phrases(question: str) -> List[str]:
    raw_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]*", question)
    content_terms = []
    stopwords = _keyword_stopwords()
    for raw in raw_terms:
        normalized = _normalize_keyword_text(raw)
        if not normalized or normalized.lower() in stopwords:
            continue
        content_terms.append(normalized)

    phrases: List[str] = []
    for size in (4, 3, 2):
        for index in range(0, max(0, len(content_terms) - size + 1)):
            phrase = " ".join(content_terms[index : index + size])
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return phrases[:12]


def _has_keyword_phrase(text: str, phrase: str) -> bool:
    normalized_text = _normalize_keyword_text(text)
    normalized_phrase = _normalize_keyword_text(phrase)
    if not normalized_phrase:
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(normalized_phrase)}(?![A-Za-z0-9])",
            normalized_text,
            re.IGNORECASE,
        )
    )


def _has_keyword_term(text: str, term: str) -> bool:
    term = _normalize_keyword_text(term)
    if not term:
        return False
    is_acronym = any(ch.isalpha() for ch in term) and term.upper() == term
    if is_acronym:
        return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text))
    if " " in term:
        return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.IGNORECASE))
    return term.lower() in text.lower()


def _asks_for_plain_np_definition(question: str) -> bool:
    if not _needs_definition_context(question):
        return False
    normalized = _normalize_keyword_text(question)
    if not re.search(r"(?<![A-Za-z0-9])NP(?![A-Za-z0-9])", normalized):
        return False
    return not re.search(r"\b(?:NPC|NP\s+complete|NP\s+completeness|NP\s+hard)\b", normalized, re.IGNORECASE)


def _keyword_terms(question: str) -> List[str]:
    raw_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]*", question)
    stopwords = _keyword_stopwords()
    terms: List[str] = []
    normalized_question = _normalize_keyword_text(question).lower()
    if any(marker in normalized_question for marker in ("npc", "np complete", "np completeness", "np complete")):
        terms.extend(["NPC", "NP complete", "NP completeness", "complete", "VERTEX COVER", "VC"])
    for alias in _enumeration_alias_terms(question):
        normalized_alias = _normalize_keyword_text(alias)
        if normalized_alias and normalized_alias not in terms:
            terms.append(normalized_alias)
    for raw in raw_terms:
        if raw.lower() in stopwords:
            continue
        candidates = [raw]
        if raw.lower().endswith("s") and len(raw) > 3:
            candidates.append(raw[:-1])
        if raw.lower().endswith("lang"):
            candidates.append(raw + "uage")
        if raw.lower().endswith("langs"):
            candidates.append(raw[:-1] + "uage")
            candidates.append(raw[:-1] + "uages")
        for candidate in candidates:
            normalized = _normalize_keyword_text(candidate)
            if normalized and normalized not in terms:
                terms.append(normalized)
    return terms


def _keyword_stopwords() -> set:
    return {
        "a",
        "about",
        "after",
        "also",
        "an",
        "and",
        "are",
        "as",
        "book",
        "by",
        "class",
        "define",
        "defined",
        "definition",
        "does",
        "for",
        "from",
        "have",
        "in",
        "is",
        "lang",
        "language",
        "languages",
        "languege",
        "langueges",
        "mention",
        "mentioned",
        "mentions",
        "problem",
        "problems",
        "of",
        "on",
        "or",
        "show",
        "that",
        "the",
        "there",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
    }


def _is_enumeration_question(question: str) -> bool:
    normalized = question.lower()
    return any(
        marker in normalized
        for marker in (
            "what are",
            "which",
            "list",
            "mentioned",
            "all",
            "languages",
            "problems",
        )
    )


def _needs_definition_context(question: str) -> bool:
    normalized = question.lower()
    return any(
        marker in normalized
        for marker in (
            "where",
            "defined",
            "definition",
            "define",
            "introduced",
        )
    )


def _normalize_keyword_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("-", " ")).strip()


def answer_question(
    base_dir: Path,
    pdf_path: str,
    api_key: str,
    question: str,
    chat_model: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    page_ranges: Optional[List[Tuple[int, int]]] = None,
    retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> Tuple[str, Dict[str, AIReference]]:
    retrieval_limit = max(1, min(60, int(retrieval_limit)))
    retrieved = retrieve_chunks(
        base_dir,
        pdf_path,
        api_key,
        question,
        embedding_model,
        limit=retrieval_limit,
        page_ranges=page_ranges,
    )
    if not retrieved:
        raise RuntimeError("No relevant chunks were found. Scan the document again.")
    answer = create_answer(api_key, chat_model, question, retrieved, page_ranges=page_ranges)
    references = {
        item.label: AIReference(
            page=item.chunk.page,
            x=item.chunk.x,
            y=item.chunk.y,
            rect=item.chunk.rect,
            text=item.chunk.text,
        )
        for item in retrieved
    }
    return answer, references


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        norm_a += fx * fx
        norm_b += fy * fy
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
