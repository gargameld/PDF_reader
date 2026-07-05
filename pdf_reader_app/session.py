import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz

from .constants import SESSION_MARKER
from .models import ATxtData, SessionPersistence

def _parse_session_data_from_lines(lines: List[str]) -> Optional[SessionPersistence]:
    json_text = "\n".join(line for line in lines if line.strip()).strip()
    if not json_text:
        return None

    try:
        data = json.loads(json_text)
        return SessionPersistence(
            active_index=int(data.get("active_index", 0)),
            documents=list(data.get("documents", [])),
            recent_documents=list(data.get("recent_documents", [])),
            equation_macros={
                str(key): str(value)
                for key, value in dict(data.get("equation_macros", {}) or {}).items()
                if str(key)
            },
            shortcut_overrides={
                str(key): str(value)
                for key, value in dict(data.get("shortcut_overrides", {}) or {}).items()
                if str(key)
            },
        )
    except Exception as exc:
        raise ValueError(f"Invalid session-state JSON in a.txt: {exc}") from exc


def parse_a_txt(path: Path) -> ATxtData:
    """Read a.txt without assuming the whole file is chapter-start data.

    The equation-search accelerator is a simple block at the *start* of a.txt:
        1 - 11
        2 - 27
        3 - 131

    Anything after that initial block belongs to other features / notes / saved
    state.  Older versions of this reader accidentally treated every later
    non-empty line as a chapter-start line, which made the parser fail and left
    equation search starting from page 1.  This parser stops the chapter scan at
    the first non chapter-start line after the initial block and preserves the
    rest of the file.
    """
    if not path.exists():
        return ATxtData()

    lines = path.read_text(encoding="utf-8").splitlines()
    chapter_starts: Dict[int, int] = {}
    preserved_lines: List[str] = []
    session_data: Optional[SessionPersistence] = None

    header_done = False
    saw_chapter_line = False
    preserved_start = len(lines)

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()

        if line == SESSION_MARKER:
            preserved_start = index
            header_done = True
            break

        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", raw_line)
        if m and not header_done:
            chapter = int(m.group(1))
            page = int(m.group(2))
            if chapter < 1 or page < 1:
                raise ValueError(f"Chapter and page must be >= 1 at line {index + 1}: {raw_line!r}")
            chapter_starts[chapter] = page
            saw_chapter_line = True
            continue

        # Allow harmless blanks/comments before the leading chapter block.
        # After the first chapter line, the first non-chapter line -- even a blank
        # or comment -- is considered the start of the preserved tail. This avoids
        # silently deleting notes or other data that live after the chapter block.
        if not saw_chapter_line and (not line or line.startswith("#")):
            continue

        # First unrelated line: everything from here belongs to the preserved tail.
        preserved_start = index
        header_done = True
        break

    if header_done:
        tail = lines[preserved_start:]
    else:
        tail = []

    if SESSION_MARKER in tail:
        marker_index = tail.index(SESSION_MARKER)
        preserved_lines = tail[:marker_index]
        session_data = _parse_session_data_from_lines(tail[marker_index + 1:])
    else:
        preserved_lines = tail

    return ATxtData(
        chapter_starts=dict(sorted(chapter_starts.items())),
        session_data=session_data,
        preserved_lines=preserved_lines,
    )


def write_a_txt(
    path: Path,
    chapter_starts: Dict[int, int],
    session_data: Optional[SessionPersistence],
    preserved_lines: Optional[List[str]] = None,
) -> None:
    """Save reader state while keeping unrelated a.txt content intact."""
    lines: List[str] = []

    for chapter, page in sorted(chapter_starts.items()):
        lines.append(f"{chapter} - {page}")

    if preserved_lines:
        lines.extend(preserved_lines)

    if session_data is not None:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(SESSION_MARKER)
        lines.append(
            json.dumps(
                {
                    "active_index": session_data.active_index,
                    "documents": session_data.documents,
                    "recent_documents": session_data.recent_documents,
                    "equation_macros": session_data.equation_macros,
                    "shortcut_overrides": session_data.shortcut_overrides,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_default_data(a_txt_path: Path) -> Tuple[Optional[Dict[int, int]], Optional[SessionPersistence], List[str]]:
    if not a_txt_path.exists():
        return None, None, []
    a_txt_data = parse_a_txt(a_txt_path)
    return a_txt_data.chapter_starts or None, a_txt_data.session_data, a_txt_data.preserved_lines




def load_chapter_starts_from_candidate_paths(candidate_paths: List[Path]) -> Tuple[Optional[Dict[int, int]], Optional[str]]:
    """Load chapter start pages from the first usable a.txt candidate.

    Expected leading format:
        1 - 14
        2 - 71
        3 - 114

    The left side is the chapter number and the right side is the 1-based PDF
    page where that chapter starts. The search code uses these values directly
    as PDF page numbers.
    """
    seen = set()
    for candidate in candidate_paths:
        try:
            candidate = candidate.resolve()
        except Exception:
            pass
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        try:
            data = parse_a_txt(candidate)
        except Exception as exc:
            print(f"[PDF Reader] Could not parse chapter starts from {candidate}: {exc}", flush=True)
            continue
        if data.chapter_starts:
            return dict(data.chapter_starts), str(candidate)
    return None, None

def get_search_page_range(
    eq_id: str,
    chapter_starts: Optional[Dict[int, int]],
    total_pages: int,
) -> Tuple[int, int]:
    if not chapter_starts:
        return 1, total_pages

    parts = eq_id.split(".")
    if not parts:
        return 1, total_pages

    try:
        chapter = int(parts[0])
    except ValueError:
        return 1, total_pages

    if chapter not in chapter_starts:
        return 1, total_pages

    start_page = chapter_starts[chapter]
    next_chapters = [ch for ch in chapter_starts if ch > chapter]
    if next_chapters:
        next_chapter = min(next_chapters)
        end_page = chapter_starts[next_chapter] - 1
    else:
        end_page = total_pages

    start_page = max(1, min(start_page, total_pages))
    end_page = max(start_page, min(end_page, total_pages))
    return start_page, end_page


def _numeric_tuple_from_text(text: str) -> Optional[Tuple[int, ...]]:
    """Extract a chapter/section number from a PDF bookmark title.

    Supports TOCs such as:
        Part I The Tools of Astronomy
          1 The Celestial Sphere
            1.1 The Greek Tradition

    Older code only recognized multi-part numbers such as 1.1.  That meant
    chapter-level bookmarks like "1 The Celestial Sphere" were ignored, so an
    equation such as (1.23) could fall back to a slow whole-document search when
    no a.txt chapter map existed.
    """
    raw = text.strip()

    # Ignore non-numeric grouping bookmarks such as "Part I ...".  A chapter or
    # section bookmark should start with Arabic digits.
    if not raw or not raw[0].isdigit():
        return None

    # Common PDF bookmark forms: "1 Title", "1.1 Title", "1.1.2 Title",
    # and occasionally "1 - 1 Title" / "1 – 1 Title".  We intentionally anchor
    # at the beginning to avoid treating years or figure numbers inside the title
    # as navigation numbers.
    m = re.match(r"^\s*(\d+(?:\s*(?:\.|-|–|—)\s*\d+)*)\b", raw)
    if not m:
        return None

    parts = [int(part) for part in re.findall(r"\d+", m.group(1))]
    if not parts:
        return None
    return tuple(parts)


def extract_section_starts_from_toc(doc: fitz.Document) -> Dict[Tuple[int, ...], int]:
    """Build {section_number_tuple: page} from the PDF bookmarks/table of contents."""
    section_starts: Dict[Tuple[int, ...], int] = {}
    try:
        toc = doc.get_toc(simple=True)
    except Exception:
        return section_starts

    for entry in toc:
        if len(entry) < 3:
            continue
        title = str(entry[1])
        try:
            page = max(1, int(entry[2]))
        except Exception:
            continue
        section = _numeric_tuple_from_text(title)
        if section is None:
            continue
        # Keep the earliest page if the same section appears more than once.
        if section not in section_starts or page < section_starts[section]:
            section_starts[section] = page

    return dict(sorted(section_starts.items()))
