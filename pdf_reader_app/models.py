from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import fitz

from .constants import EQUATION_LOOKUP_ROUGH_INDEX, EQUATION_LOOKUP_SCAN, EQUATION_MODE_SECTION

@dataclass
class ViewState:
    page: int
    scroll_x: int
    scroll_y: int
    zoom: float


@dataclass
class UserBookmark:
    title: str
    page: int


@dataclass
class SelectionLine:
    page: int
    y: float


@dataclass
class SearchHit:
    page: int
    rect: Tuple[float, float, float, float]


@dataclass
class PdfComment:
    page: int
    rect: Tuple[float, float, float, float]
    text: str
    color: str = "#2EAD4A"


@dataclass
class EquationLocation:
    page: int
    y: float


@dataclass
class EquationCandidate:
    eq_id: str
    raw_text: str
    location: EquationLocation


@dataclass
class FigureDestination:
    page: int
    y: float = 0.0


@dataclass
class SessionPersistence:
    active_index: int = 0
    documents: List[dict] = field(default_factory=list)
    recent_documents: List[object] = field(default_factory=list)
    equation_macros: Dict[str, str] = field(default_factory=dict)


@dataclass
class ATxtData:
    chapter_starts: Dict[int, int] = field(default_factory=dict)
    session_data: Optional[SessionPersistence] = None
    preserved_lines: List[str] = field(default_factory=list)


@dataclass
class DocumentSession:
    path: str
    doc: fitz.Document
    current_page: int = 1
    history_back: List[ViewState] = field(default_factory=list)
    history_forward: List[ViewState] = field(default_factory=list)
    saved_view_state: Optional[ViewState] = None
    chapter_starts: Optional[Dict[int, int]] = None
    chapter_starts_source: Optional[str] = None
    section_starts: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    equation_cache: Dict[str, int] = field(default_factory=dict)
    figure_cache: Dict[str, FigureDestination] = field(default_factory=dict)
    user_bookmarks: List[UserBookmark] = field(default_factory=list)
    selection_lines: List[SelectionLine] = field(default_factory=list)
    comments: List[PdfComment] = field(default_factory=list)
    last_equation_return_state: Optional[ViewState] = None
    equation_numbering_mode: str = EQUATION_MODE_SECTION
    rendered_pages: List[Any] = field(default_factory=list)
    text_search_query: str = ""
    text_search_hits: List[SearchHit] = field(default_factory=list)
    text_search_index: int = -1
    text_search_origin_state: Optional[ViewState] = None
    text_search_scan_page: int = 1
    text_search_complete: bool = True
    equation_index: Dict[str, EquationLocation] = field(default_factory=dict)
    equation_rough_index: Dict[str, EquationLocation] = field(default_factory=dict)
    equation_pattern_index: Dict[str, EquationLocation] = field(default_factory=dict)
    equation_index_patterns: Dict[int, str] = field(default_factory=dict)
    equation_format: Dict[str, object] = field(default_factory=dict)
    manual_equation_samples: List[EquationCandidate] = field(default_factory=list)
    equation_index_signature: Optional[dict] = None
    equation_lookup_mode: str = EQUATION_LOOKUP_SCAN
    rendered_page_numbers: set = field(default_factory=set)
    page_display_sizes: List[Tuple[int, int]] = field(default_factory=list)
