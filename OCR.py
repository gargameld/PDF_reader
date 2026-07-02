import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPainter, QPixmap, QPen, QColor, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

EQUATION_REF_RE = re.compile(r"\(?\d+(?:[.\s,\-]+\d+)+\)?")
SESSION_MARKER = "[SESSION_STATE]"


@dataclass
class ViewState:
    page: int
    scroll_x: int
    scroll_y: int
    zoom: float


@dataclass
class SessionPersistence:
    active_index: int = 0
    documents: List[dict] = field(default_factory=list)


@dataclass
class DocumentSession:
    path: str
    doc: fitz.Document
    current_page: int = 1
    history_back: List[ViewState] = field(default_factory=list)
    history_forward: List[ViewState] = field(default_factory=list)
    saved_view_state: Optional[ViewState] = None
    chapter_starts: Optional[Dict[int, int]] = None


def parse_a_txt(path: Path) -> Tuple[Dict[int, int], Optional[SessionPersistence]]:
    chapter_starts: Dict[int, int] = {}
    session_data: Optional[SessionPersistence] = None

    if not path.exists():
        return chapter_starts, None

    lines = path.read_text(encoding="utf-8").splitlines()

    in_session = False
    json_lines: List[str] = []

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not in_session:
            if not line or line.startswith("#"):
                continue
            if line == SESSION_MARKER:
                in_session = True
                continue

            m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", line)
            if not m:
                raise ValueError(f"Invalid chapter-start format at line {lineno}: {raw_line!r}")

            chapter = int(m.group(1))
            page = int(m.group(2))
            if chapter < 1 or page < 1:
                raise ValueError(f"Chapter and page must be >= 1 at line {lineno}: {raw_line!r}")

            chapter_starts[chapter] = page
        else:
            if raw_line.strip():
                json_lines.append(raw_line)

    if json_lines:
        try:
            data = json.loads("\n".join(json_lines))
            session_data = SessionPersistence(
                active_index=int(data.get("active_index", 0)),
                documents=list(data.get("documents", [])),
            )
        except Exception as exc:
            raise ValueError(f"Invalid session-state JSON in a.txt: {exc}") from exc

    return dict(sorted(chapter_starts.items())), session_data


def write_a_txt(
    path: Path,
    chapter_starts: Dict[int, int],
    session_data: Optional[SessionPersistence],
) -> None:
    lines: List[str] = []

    for chapter, page in sorted(chapter_starts.items()):
        lines.append(f"{chapter} - {page}")

    if session_data is not None:
        lines.append("")
        lines.append(SESSION_MARKER)
        lines.append(
            json.dumps(
                {
                    "active_index": session_data.active_index,
                    "documents": session_data.documents,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_default_data(a_txt_path: Path) -> Tuple[Optional[Dict[int, int]], Optional[SessionPersistence]]:
    if not a_txt_path.exists():
        return None, None
    chapter_starts, session_data = parse_a_txt(a_txt_path)
    return chapter_starts or None, session_data


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


class PdfView(QGraphicsView):
    view_changed = Signal()
    clicked = Signal(float, float)
    page_down_requested = Signal()
    page_up_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._base_pixmap: Optional[QPixmap] = None
        self._current_zoom: float = 1.0
        self._debug_rect: Optional[QGraphicsRectItem] = None
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.lightGray)
        self.setFocusPolicy(Qt.StrongFocus)

        self.horizontalScrollBar().valueChanged.connect(lambda _: self.view_changed.emit())
        self.verticalScrollBar().valueChanged.connect(lambda _: self.view_changed.emit())

    def set_page_image(self, pixmap: QPixmap) -> None:
        self.scene().clear()
        self._base_pixmap = pixmap
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.resetTransform()
        self._current_zoom = 1.0
        self.centerOn(self._pixmap_item)
        self._debug_rect = None
        self.view_changed.emit()

    def clear_page(self) -> None:
        self.scene().clear()
        self._pixmap_item = None
        self._base_pixmap = None
        self._current_zoom = 1.0
        self._debug_rect = None
        self.view_changed.emit()

    def show_debug_box(self, x: float, y: float, w: float, h: float) -> None:
        if self._debug_rect is not None:
            self.scene().removeItem(self._debug_rect)
        rect = QRectF(x - w / 2, y - h / 2, w, h)
        self._debug_rect = self.scene().addRect(rect, QPen(QColor("red"), 2))

    def zoom_factor(self) -> float:
        return self._current_zoom

    def set_zoom_factor(self, factor: float) -> None:
        if self._base_pixmap is None:
            return
        factor = max(0.2, min(6.0, factor))
        current = self._current_zoom
        if abs(current - factor) < 1e-9:
            return
        scale_ratio = factor / current
        self.scale(scale_ratio, scale_ratio)
        self._current_zoom = factor
        self.view_changed.emit()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            step = 1.15 if delta > 0 else 1 / 1.15
            self.set_zoom_factor(self._current_zoom * step)
            event.accept()
            return
        super().wheelEvent(event)
        self.view_changed.emit()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.clicked.emit(scene_pos.x(), scene_pos.y())
        super().mouseReleaseEvent(event)
        self.view_changed.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Down:
            self.page_down_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key_Up:
            self.page_up_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        super().keyReleaseEvent(event)
        self.view_changed.emit()


class PdfReaderWindow(QMainWindow):
    CLICK_BOX_W = 20
    CLICK_BOX_H = 1
    CLICK_RADIUS_PDF = 4.0
    LINE_Y_TOL = 3.0
    MAX_HISTORY = 200

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PDF Reader with Clickable Equation References")
        self.resize(1200, 850)

        self.a_txt_path = Path(__file__).resolve().parent / "a.txt"

        self.sessions: List[DocumentSession] = []
        self.current_index: int = -1
        self._suspend_history = False
        self._switching_docs = False

        self.default_chapter_starts: Optional[Dict[int, int]] = None
        self.startup_session_data: Optional[SessionPersistence] = None

        try:
            self.default_chapter_starts, self.startup_session_data = load_default_data(self.a_txt_path)
        except Exception as exc:
            QMessageBox.warning(self, "a.txt", f"Could not load a.txt:\n{exc}")
            self.default_chapter_starts = None
            self.startup_session_data = None

        self.view = PdfView()
        self.view.view_changed.connect(self._on_view_changed)
        self.view.clicked.connect(self.jump_if_reference_clicked)
        self.view.page_down_requested.connect(self.next_page)
        self.view.page_up_requested.connect(self.prev_page)

        self.doc_tabs = QTabBar()
        self.doc_tabs.setExpanding(False)
        self.doc_tabs.setUsesScrollButtons(True)
        self.doc_tabs.setTabsClosable(True)
        self.doc_tabs.currentChanged.connect(self._tab_changed)
        self.doc_tabs.tabCloseRequested.connect(self.close_document_at_index)

        self.page_input = QSpinBox()
        self.page_input.setMinimum(1)
        self.page_input.setMaximum(1)
        self.page_input.setKeyboardTracking(False)
        self.page_input.valueChanged.connect(self._page_spin_changed)

        self.eq_input = QLineEdit()
        self.eq_input.setPlaceholderText("Equation number, e.g. 5.4.13")
        self.eq_input.returnPressed.connect(self.jump_to_equation)

        self.status_label = QLabel("No PDF loaded")
        self.zoom_label = QLabel("100%")

        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_pdf)

        open_many_action = QAction("Open Multiple", self)
        open_many_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_many_action.triggered.connect(self.open_multiple_pdfs)

        close_doc_action = QAction("Close Document", self)
        close_doc_action.setShortcut(QKeySequence.Close)
        close_doc_action.triggered.connect(self.close_current_document)

        back_action = QAction("Back", self)
        back_action.setShortcut(QKeySequence("Alt+Left"))
        back_action.triggered.connect(self.go_back)

        forward_action = QAction("Forward", self)
        forward_action.setShortcut(QKeySequence("Alt+Right"))
        forward_action.triggered.connect(self.go_forward)

        prev_action = QAction("Previous Page", self)
        prev_action.setShortcut(QKeySequence.MoveToPreviousPage)
        prev_action.triggered.connect(self.prev_page)

        next_action = QAction("Next Page", self)
        next_action.setShortcut(QKeySequence.MoveToNextPage)
        next_action.triggered.connect(self.next_page)

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(lambda: self.view.set_zoom_factor(self.view.zoom_factor() * 1.2))

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(lambda: self.view.set_zoom_factor(self.view.zoom_factor() / 1.2))

        for action in [
            open_action,
            open_many_action,
            close_doc_action,
            back_action,
            forward_action,
            prev_action,
            next_action,
            zoom_in_action,
            zoom_out_action,
        ]:
            self.addAction(action)

        toolbar = QToolBar("Main")
        toolbar.addAction(open_action)
        toolbar.addAction(open_many_action)
        toolbar.addAction(close_doc_action)
        toolbar.addSeparator()
        toolbar.addAction(back_action)
        toolbar.addAction(forward_action)
        toolbar.addSeparator()
        toolbar.addAction(prev_action)
        toolbar.addAction(next_action)
        toolbar.addWidget(QLabel("  Page: "))
        toolbar.addWidget(self.page_input)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Eq: "))
        toolbar.addWidget(self.eq_input)
        jump_button = QPushButton("Jump")
        jump_button.clicked.connect(self.jump_to_equation)
        toolbar.addWidget(jump_button)
        toolbar.addSeparator()
        toolbar.addAction(zoom_in_action)
        toolbar.addAction(zoom_out_action)
        self.addToolBar(toolbar)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.doc_tabs)
        layout.addWidget(self.view)

        bottom = QHBoxLayout()
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        bottom.addWidget(self.zoom_label)
        layout.addLayout(bottom)

        self.setCentralWidget(central)

        self._restore_startup_session()
        self.view.setFocus()

    # ----------------------------
    # Basic helpers
    # ----------------------------

    def current_session(self) -> Optional[DocumentSession]:
        if 0 <= self.current_index < len(self.sessions):
            return self.sessions[self.current_index]
        return None

    def _session_display_name(self, path: str) -> str:
        return Path(path).name

    def _set_no_document_ui(self) -> None:
        self.current_index = -1
        self.page_input.blockSignals(True)
        self.page_input.setMinimum(1)
        self.page_input.setMaximum(1)
        self.page_input.setValue(1)
        self.page_input.blockSignals(False)
        self.view.clear_page()
        self.status_label.setText("No PDF loaded")
        self.zoom_label.setText("100%")

    def _render_page(self, session: DocumentSession, page_number: int) -> QPixmap:
        page = session.doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def _current_view_state(self) -> Optional[ViewState]:
        session = self.current_session()
        if session is None:
            return None
        return ViewState(
            page=session.current_page,
            scroll_x=self.view.horizontalScrollBar().value(),
            scroll_y=self.view.verticalScrollBar().value(),
            zoom=self.view.zoom_factor(),
        )

    def _save_active_session_view(self) -> None:
        session = self.current_session()
        if session is None:
            return
        session.saved_view_state = self._current_view_state()

    def _restore_view_state(self, session: DocumentSession, state: ViewState) -> None:
        self._show_page_for_session(session, state.page, push_history=False)
        self.view.set_zoom_factor(state.zoom)
        self.view.horizontalScrollBar().setValue(state.scroll_x)
        self.view.verticalScrollBar().setValue(state.scroll_y)
        self._update_status()

    def _default_state_for_session(self, session: DocumentSession) -> ViewState:
        return ViewState(page=max(1, session.current_page), scroll_x=0, scroll_y=0, zoom=1.0)

    def _push_history(self, session: DocumentSession) -> None:
        if self._suspend_history or self._switching_docs:
            return
        state = self._current_view_state()
        if state is not None:
            session.history_back.append(state)
            if len(session.history_back) > self.MAX_HISTORY:
                session.history_back.pop(0)
            session.history_forward.clear()

    def _show_page_for_session(self, session: DocumentSession, page_number: int, push_history: bool = True) -> None:
        page_number = max(1, min(page_number, len(session.doc)))
        is_current = (session is self.current_session())

        if push_history and is_current and page_number != session.current_page:
            self._push_history(session)

        session.current_page = page_number

        if is_current:
            self.view.set_page_image(self._render_page(session, page_number))
            self.page_input.blockSignals(True)
            self.page_input.setMaximum(len(session.doc))
            self.page_input.setValue(page_number)
            self.page_input.blockSignals(False)
            self._update_status()
            self.view.setFocus()

    def _update_status(self) -> None:
        session = self.current_session()
        if session is None:
            self.status_label.setText("No PDF loaded")
            self.zoom_label.setText("100%")
            return
        self.status_label.setText(f"Page {session.current_page} / {len(session.doc)}")
        self.zoom_label.setText(f"{round(self.view.zoom_factor() * 100)}%")

    def _on_view_changed(self) -> None:
        session = self.current_session()
        if session is None:
            self.status_label.setText("No PDF loaded")
            self.zoom_label.setText("100%")
            return
        if not self._switching_docs:
            session.saved_view_state = self._current_view_state()
        self._update_status()

    # ----------------------------
    # Open / close documents
    # ----------------------------

    def open_pdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if path:
            self.load_pdf(path, switch_to=True)

    def open_multiple_pdfs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open PDFs", "", "PDF Files (*.pdf)")
        if not paths:
            return
        for i, path in enumerate(paths):
            self.load_pdf(path, switch_to=(i == len(paths) - 1))

    def load_pdf(
        self,
        path: str,
        switch_to: bool = True,
        restored_state: Optional[ViewState] = None,
        restored_back: Optional[List[ViewState]] = None,
        restored_forward: Optional[List[ViewState]] = None,
    ) -> None:
        existing_index = self._find_session_by_path(path)
        if existing_index is not None:
            if switch_to:
                self.doc_tabs.setCurrentIndex(existing_index)
            return

        try:
            doc = fitz.open(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not open PDF:\n{exc}")
            return

        chapter_starts = dict(self.default_chapter_starts) if self.default_chapter_starts else None

        session = DocumentSession(
            path=path,
            doc=doc,
            current_page=1,
            history_back=restored_back[:] if restored_back else [],
            history_forward=restored_forward[:] if restored_forward else [],
            saved_view_state=restored_state,
            chapter_starts=chapter_starts,
        )

        if restored_state is not None:
            session.current_page = restored_state.page

        self.sessions.append(session)
        idx = len(self.sessions) - 1
        self.doc_tabs.addTab(self._session_display_name(path))
        self.doc_tabs.setTabToolTip(idx, path)

        if switch_to:
            self.doc_tabs.setCurrentIndex(idx)

    def _find_session_by_path(self, path: str) -> Optional[int]:
        norm = str(Path(path).resolve())
        for i, session in enumerate(self.sessions):
            try:
                if str(Path(session.path).resolve()) == norm:
                    return i
            except Exception:
                if session.path == path:
                    return i
        return None

    def close_current_document(self) -> None:
        if self.current_index >= 0:
            self.close_document_at_index(self.current_index)

    def close_document_at_index(self, index: int) -> None:
        if not (0 <= index < len(self.sessions)):
            return

        if index == self.current_index:
            self._save_active_session_view()

        session = self.sessions.pop(index)
        try:
            session.doc.close()
        except Exception:
            pass

        self.doc_tabs.blockSignals(True)
        self.doc_tabs.removeTab(index)
        self.doc_tabs.blockSignals(False)

        if not self.sessions:
            self._set_no_document_ui()
            return

        if index < self.current_index:
            new_index = self.current_index - 1
        elif index == self.current_index:
            new_index = min(index, len(self.sessions) - 1)
        else:
            new_index = self.current_index

        self.doc_tabs.setCurrentIndex(new_index)
        self._switch_to_document(new_index)

    def _tab_changed(self, index: int) -> None:
        if index == self.current_index:
            return
        self._switch_to_document(index)

    def _switch_to_document(self, index: int) -> None:
        if not (0 <= index < len(self.sessions)):
            self._set_no_document_ui()
            return

        self._save_active_session_view()

        self._switching_docs = True
        try:
            self.current_index = index
            session = self.sessions[index]

            self.page_input.blockSignals(True)
            self.page_input.setMinimum(1)
            self.page_input.setMaximum(len(session.doc))
            self.page_input.setValue(session.current_page)
            self.page_input.blockSignals(False)

            state = session.saved_view_state or self._default_state_for_session(session)
            self._restore_view_state(session, state)
        finally:
            self._switching_docs = False

        self.view.setFocus()

    # ----------------------------
    # Navigation
    # ----------------------------

    def _page_spin_changed(self, value: int) -> None:
        session = self.current_session()
        if session is None:
            return
        if value != session.current_page:
            self._show_page_for_session(session, value, push_history=True)

    def prev_page(self) -> None:
        session = self.current_session()
        if session is None:
            return
        self._show_page_for_session(session, session.current_page - 1, push_history=True)

    def next_page(self) -> None:
        session = self.current_session()
        if session is None:
            return
        self._show_page_for_session(session, session.current_page + 1, push_history=True)

    def go_back(self) -> None:
        session = self.current_session()
        if session is None or not session.history_back:
            return

        current = self._current_view_state()
        prev = session.history_back.pop()
        if current is not None:
            session.history_forward.append(current)

        self._suspend_history = True
        try:
            self._restore_view_state(session, prev)
            session.saved_view_state = prev
        finally:
            self._suspend_history = False

    def go_forward(self) -> None:
        session = self.current_session()
        if session is None or not session.history_forward:
            return

        current = self._current_view_state()
        nxt = session.history_forward.pop()
        if current is not None:
            session.history_back.append(current)

        self._suspend_history = True
        try:
            self._restore_view_state(session, nxt)
            session.saved_view_state = nxt
        finally:
            self._suspend_history = False

    # ----------------------------
    # Equation search
    # ----------------------------

    @staticmethod
    def _normalize_equation_id(raw: str) -> str:
        parts = re.findall(r"\d+", raw)
        if not parts:
            raise ValueError("Invalid equation number")
        return ".".join(parts)

    @staticmethod
    def _build_search_variants(eq_id: str) -> List[str]:
        parts = eq_id.split(".")
        dotted = ".".join(parts)
        spaced = " ".join(parts)
        comma = ",".join(parts)
        dashed = "-".join(parts)
        mixed = " . ".join(parts)
        variants = [
            dotted,
            f"({dotted})",
            f"{dotted})",
            f"({dotted}",
            spaced,
            f"({spaced})",
            dashed,
            f"({dashed})",
            comma,
            f"({comma})",
            mixed,
            f"({mixed})",
        ]
        out: List[str] = []
        seen = set()
        for v in variants:
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out

    def _has_equal_nearby(self, page: fitz.Page, rect: fitz.Rect) -> bool:
        window = fitz.Rect(
            max(0, rect.x0 - 80),
            max(0, rect.y0 - 20),
            min(page.rect.width, rect.x1 + 700),
            min(page.rect.height, rect.y1 + 20),
        )
        return any(window.intersects(r) for r in page.search_for("="))

    def _find_equation_page(self, session: DocumentSession, user_eq: str) -> Optional[int]:
        eq_id = self._normalize_equation_id(user_eq)
        variants = self._build_search_variants(eq_id)
        fallback = None
        start_page, end_page = get_search_page_range(eq_id, session.chapter_starts, len(session.doc))

        for page_index in range(start_page - 1, end_page):
            page = session.doc.load_page(page_index)
            for variant in variants:
                hits = page.search_for(variant)
                if not hits:
                    continue
                if fallback is None:
                    fallback = page_index + 1
                for hit in hits:
                    if self._has_equal_nearby(page, hit):
                        return page_index + 1
        return fallback

    def _collect_line_text_near_click(self, page: fitz.Page, pdf_x: float, pdf_y: float) -> Optional[str]:
        text_dict = page.get_text("dict")
        click_rect = fitz.Rect(
            pdf_x - self.CLICK_RADIUS_PDF,
            pdf_y - self.CLICK_RADIUS_PDF,
            pdf_x + self.CLICK_RADIUS_PDF,
            pdf_y + self.CLICK_RADIUS_PDF,
        )

        best_line_text = None
        best_dist2 = None

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_spans = line.get("spans", [])
                if not line_spans:
                    continue

                line_rect = fitz.Rect(line_spans[0]["bbox"])
                for sp in line_spans[1:]:
                    line_rect |= fitz.Rect(sp["bbox"])

                expanded_rect = fitz.Rect(
                    line_rect.x0,
                    line_rect.y0 - self.LINE_Y_TOL,
                    line_rect.x1,
                    line_rect.y1 + self.LINE_Y_TOL,
                )

                if not expanded_rect.intersects(click_rect):
                    continue

                line_text = "".join(sp.get("text", "") for sp in line_spans)
                if not line_text.strip():
                    continue

                cx = (line_rect.x0 + line_rect.x1) / 2.0
                cy = (line_rect.y0 + line_rect.y1) / 2.0
                d2 = (cx - pdf_x) ** 2 + (cy - pdf_y) ** 2

                if best_dist2 is None or d2 < best_dist2:
                    best_dist2 = d2
                    best_line_text = line_text

        return best_line_text

    def _find_clicked_equation_reference(self, page: fitz.Page, pdf_x: float, pdf_y: float) -> Optional[str]:
        line_text = self._collect_line_text_near_click(page, pdf_x, pdf_y)
        if not line_text:
            return None

        matches = list(EQUATION_REF_RE.finditer(line_text))
        if not matches:
            return None

        best_raw = max((m.group(0) for m in matches), key=len)
        try:
            return self._normalize_equation_id(best_raw)
        except ValueError:
            return None

    def jump_if_reference_clicked(self, scene_x: float, scene_y: float) -> None:
        session = self.current_session()
        if session is None:
            return

        zoom = self.view.zoom_factor()
        if zoom <= 0:
            return

        pdf_x = scene_x / zoom
        pdf_y = scene_y / zoom
        self.view.show_debug_box(scene_x, scene_y, self.CLICK_BOX_W, self.CLICK_BOX_H)

        page = session.doc.load_page(session.current_page - 1)
        eq_id = self._find_clicked_equation_reference(page, pdf_x, pdf_y)

        if not eq_id:
            self.statusBar().showMessage("Click detected, but no equation reference under cursor")
            return

        page_num = self._find_equation_page(session, eq_id)
        if page_num is None:
            self.statusBar().showMessage(f"Found {eq_id}, but destination page was not found")
            return

        self.statusBar().showMessage(f"Found {eq_id}, jumping to page {page_num}")
        self._show_page_for_session(session, page_num, push_history=True)

    def jump_to_equation(self) -> None:
        session = self.current_session()
        if session is None:
            return

        text = self.eq_input.text().strip()
        if not text:
            return

        try:
            page = self._find_equation_page(session, text)
        except ValueError:
            QMessageBox.warning(self, "Invalid", "Invalid equation number")
            return

        if page is None:
            QMessageBox.information(self, "Not found", "Equation not found")
            return

        self._show_page_for_session(session, page, push_history=True)

    # ----------------------------
    # Startup / shutdown persistence
    # ----------------------------

    @staticmethod
    def _serialize_view_state(state: ViewState) -> dict:
        return {
            "page": state.page,
            "scroll_x": state.scroll_x,
            "scroll_y": state.scroll_y,
            "zoom": state.zoom,
        }

    @staticmethod
    def _deserialize_view_state(data: dict) -> ViewState:
        return ViewState(
            page=max(1, int(data.get("page", 1))),
            scroll_x=int(data.get("scroll_x", 0)),
            scroll_y=int(data.get("scroll_y", 0)),
            zoom=float(data.get("zoom", 1.0)),
        )

    def _build_session_persistence(self) -> SessionPersistence:
        self._save_active_session_view()

        docs_data: List[dict] = []
        for session in self.sessions:
            state = session.saved_view_state or self._default_state_for_session(session)
            docs_data.append(
                {
                    "path": session.path,
                    "view_state": self._serialize_view_state(state),
                    "history_back": [self._serialize_view_state(v) for v in session.history_back],
                    "history_forward": [self._serialize_view_state(v) for v in session.history_forward],
                }
            )

        active_index = max(0, self.current_index) if self.sessions else 0
        return SessionPersistence(active_index=active_index, documents=docs_data)

    def _restore_startup_session(self) -> None:
        restored_any = False

        if self.startup_session_data:
            for doc_entry in self.startup_session_data.documents:
                path = doc_entry.get("path")
                if not path:
                    continue
                if not Path(path).exists():
                    continue

                try:
                    restored_state = self._deserialize_view_state(doc_entry.get("view_state", {}))
                    restored_back = [
                        self._deserialize_view_state(v)
                        for v in doc_entry.get("history_back", [])
                    ]
                    restored_forward = [
                        self._deserialize_view_state(v)
                        for v in doc_entry.get("history_forward", [])
                    ]
                except Exception:
                    restored_state = ViewState(page=1, scroll_x=0, scroll_y=0, zoom=1.0)
                    restored_back = []
                    restored_forward = []

                self.load_pdf(
                    path,
                    switch_to=False,
                    restored_state=restored_state,
                    restored_back=restored_back,
                    restored_forward=restored_forward,
                )
                restored_any = True

            if restored_any and self.sessions:
                idx = min(max(0, self.startup_session_data.active_index), len(self.sessions) - 1)
                self.doc_tabs.setCurrentIndex(idx)
                self._switch_to_document(idx)

        if not restored_any:
            self._set_no_document_ui()

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            session_data = self._build_session_persistence()
            chapters = self.default_chapter_starts or {}
            write_a_txt(self.a_txt_path, chapters, session_data)
        except Exception as exc:
            QMessageBox.warning(self, "Save warning", f"Could not save session state to a.txt:\n{exc}")

        for session in self.sessions:
            try:
                session.doc.close()
            except Exception:
                pass

        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = PdfReaderWindow()

    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            if Path(path).exists():
                window.load_pdf(path, switch_to=False)
        if window.sessions:
            window.doc_tabs.setCurrentIndex(0)
            window._switch_to_document(0)

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())