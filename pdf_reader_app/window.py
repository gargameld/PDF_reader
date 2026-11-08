import html
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
from PySide6.QtCore import Qt, QTimer, QPoint, QUrl, QBuffer, QIODevice, QEvent, QSize, QThread, Signal
from PySide6.QtGui import QAction, QCloseEvent, QColor, QCursor, QDesktopServices, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QTextBrowser,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .ai_document import (
    DEFAULT_CHAT_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVAL_LIMIT,
    AIReference,
    answer_question,
    append_chat_history,
    build_document_index,
    clear_chat_history,
    format_page_ranges,
    has_current_index,
    load_chat_history,
    load_ai_settings,
    openai_api_key,
    save_ai_settings,
    session_usage_totals,
)
from .application import PdfReaderApplication
from .constants import *
from .google_drive import DRIVE_FOLDER_NAME, CloudDocument, GoogleDriveClient, GoogleDriveError
from .models import *
from .session import *
from .widgets import *

BOOKMARKS_DOCK_MIN_WIDTH = 220
BOOKMARKS_DOCK_DEFAULT_WIDTH = 320
READER_PANE_MIN_HEIGHT = 24

class ShortcutSettingsDialog(QDialog):
    def __init__(
        self,
        definitions: List[Tuple[str, str, str]],
        shortcuts: Dict[str, str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Shortcuts")
        self.resize(560, 620)
        self._definitions = definitions
        self._shortcuts = dict(shortcuts)
        self._buttons: Dict[str, QPushButton] = {}
        self._listening_id: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        hint = QLabel("Click a shortcut, then press the new keys. Esc cancels; Backspace clears.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        for row, (shortcut_id, title, default_shortcut) in enumerate(self._definitions):
            label = QLabel(title)
            label.setToolTip(f"Default: {default_shortcut}")
            button = QPushButton(self._shortcuts.get(shortcut_id, ""))
            button.setMinimumWidth(150)
            button.clicked.connect(lambda _checked=False, sid=shortcut_id: self._listen_for_shortcut(sid))
            reset_button = QPushButton("Reset")
            reset_button.clicked.connect(lambda _checked=False, sid=shortcut_id, default=default_shortcut: self._set_shortcut(sid, default))
            grid.addWidget(label, row, 0)
            grid.addWidget(button, row, 1)
            grid.addWidget(reset_button, row, 2)
            self._buttons[shortcut_id] = button

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def shortcuts(self) -> Dict[str, str]:
        return dict(self._shortcuts)

    def _listen_for_shortcut(self, shortcut_id: str) -> None:
        self._listening_id = shortcut_id
        self._buttons[shortcut_id].setText("Press keys...")
        self._buttons[shortcut_id].setFocus()

    def _set_shortcut(self, shortcut_id: str, shortcut: str) -> None:
        self._shortcuts[shortcut_id] = shortcut
        self._buttons[shortcut_id].setText(shortcut or "None")
        self._listening_id = None

    def keyPressEvent(self, event) -> None:
        if self._listening_id is None:
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_Escape:
            self._buttons[self._listening_id].setText(self._shortcuts.get(self._listening_id, "") or "None")
            self._listening_id = None
            event.accept()
            return

        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            self._set_shortcut(self._listening_id, "")
            event.accept()
            return

        shortcut = self._shortcut_from_event(event)
        if shortcut:
            self._set_shortcut(self._listening_id, shortcut)
            event.accept()
            return

        super().keyPressEvent(event)

    @staticmethod
    def _shortcut_from_event(event) -> str:
        key = event.key()
        if key in (
            Qt.Key_Control,
            Qt.Key_Shift,
            Qt.Key_Alt,
            Qt.Key_Meta,
            Qt.Key_AltGr,
        ):
            return ""

        parts: List[str] = []
        modifiers = event.modifiers()
        if modifiers & Qt.ControlModifier:
            parts.append("Ctrl")
        if modifiers & Qt.AltModifier:
            parts.append("Alt")
        if modifiers & Qt.ShiftModifier:
            parts.append("Shift")
        if modifiers & Qt.MetaModifier:
            parts.append("Meta")

        key_name = ""
        for candidate, qt_key in SHORTCUT_KEY_CODES.items():
            if key == qt_key:
                key_name = candidate.title() if len(candidate) > 1 else candidate
                break
        if not key_name:
            key_name = QKeySequence(key).toString(QKeySequence.PortableText)
        if not key_name:
            return ""
        if key_name == "+":
            return "+".join(parts) + "++" if parts else "+"
        parts.append(key_name)
        return "+".join(parts)


class AIIndexWorker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        base_dir: Path,
        pdf_path: str,
        api_key: str,
        embedding_model: str,
        page_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.pdf_path = pdf_path
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.page_ranges = page_ranges

    def run(self) -> None:
        try:
            result = build_document_index(
                self.base_dir,
                self.pdf_path,
                self.api_key,
                self.embedding_model,
                page_ranges=self.page_ranges,
                progress=self.progress.emit,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class AIQuestionWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        base_dir: Path,
        pdf_path: str,
        api_key: str,
        question: str,
        chat_model: str,
        embedding_model: str,
        page_ranges: Optional[List[Tuple[int, int]]] = None,
        retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.pdf_path = pdf_path
        self.api_key = api_key
        self.question = question
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.page_ranges = page_ranges
        self.retrieval_limit = retrieval_limit

    def run(self) -> None:
        try:
            answer, references = answer_question(
                self.base_dir,
                self.pdf_path,
                self.api_key,
                self.question,
                self.chat_model,
                self.embedding_model,
                page_ranges=self.page_ranges,
                retrieval_limit=self.retrieval_limit,
            )
            self.completed.emit(
                {
                    "question": self.question,
                    "answer": answer,
                    "references": {
                        label: {
                            "page": ref.page,
                            "x": ref.x,
                            "y": ref.y,
                            "rect": list(ref.rect),
                            "text": ref.text,
                        }
                        for label, ref in references.items()
                    },
                    "page_ranges": format_page_ranges(self.page_ranges),
                    "retrieval_limit": self.retrieval_limit,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class PdfReaderWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PDF Reader with Clickable Equation References")
        self.resize(1350, 900)
        self.setAcceptDrops(True)

        self.base_dir = Path(__file__).resolve().parent.parent
        self.a_txt_path = self.base_dir / "a.txt"
        self.snippet_dir = self.base_dir / "ocr_snippets"
        self.snippet_dir.mkdir(exist_ok=True)
        self.cloud_cache_dir = self.base_dir / "cloud_documents"

        self.sessions: List[DocumentSession] = []
        self.current_index: int = -1
        self._suspend_history = False
        self._suspend_view_updates = False
        self._switching_docs = False
        self.recent_documents: List[str] = []
        self.recent_document_entries: Dict[str, dict] = {}
        self.document_activation_history: List[str] = []
        self._thumbnail_cache: Dict[str, QPixmap] = {}
        self._cloud_thumbnail_cache: Dict[str, QPixmap] = {}
        self.cloud_documents: List[CloudDocument] = []
        self.google_drive = GoogleDriveClient(
            self.base_dir / "google_drive_token.json",
            self.cloud_cache_dir,
            self.base_dir / "google_oauth_client.json",
        )
        self.ai_settings = load_ai_settings(self.base_dir)
        self._ai_threads: List[QThread] = []
        self._ai_message_counter = 0
        self.ai_reference_map: Dict[str, AIReference] = {}
        self._ai_history_document_path: Optional[str] = None
        self._layout_independent_shortcuts: List[Tuple[QAction, Qt.KeyboardModifiers, str]] = []
        self._shortcut_actions: Dict[str, QAction] = {}
        self._shortcut_defaults: Dict[str, str] = {
            shortcut_id: default for shortcut_id, _title, default in SHORTCUT_DEFINITIONS
        }
        self.shortcut_overrides: Dict[str, str] = {}

        self.default_chapter_starts: Optional[Dict[int, int]] = None
        self.startup_session_data: Optional[SessionPersistence] = None
        self.a_txt_preserved_lines: List[str] = []
        self.equation_macros: Dict[str, str] = {
            r"\e": "exp(",
            r"\dd": "d",
            r"\grad": r"\nabla",
            r"\del": r"\partial",
        }

        try:
            (
                self.default_chapter_starts,
                self.startup_session_data,
                self.a_txt_preserved_lines,
            ) = load_default_data(self.a_txt_path)
            print(
                f"[PDF Reader] Loaded a.txt from {self.a_txt_path}: "
                f"chapter_starts={self.default_chapter_starts or {}}, "
                f"session_state={'yes' if self.startup_session_data else 'no'}, "
                f"preserved_lines={len(self.a_txt_preserved_lines)}",
                flush=True,
            )
        except Exception as exc:
            QMessageBox.warning(self, "a.txt", f"Could not load a.txt:\n{exc}")
            self.default_chapter_starts = None
            self.startup_session_data = None
            self.a_txt_preserved_lines = []

        self.recent_documents, self.recent_document_entries = self._recent_documents_from_session_data(
            self.startup_session_data
        )
        if self.startup_session_data and self.startup_session_data.equation_macros:
            self.equation_macros.update(self.startup_session_data.equation_macros)
        if self.startup_session_data and self.startup_session_data.shortcut_overrides:
            self.shortcut_overrides = {
                shortcut_id: shortcut
                for shortcut_id, shortcut in self.startup_session_data.shortcut_overrides.items()
                if shortcut_id in self._shortcut_defaults
            }

        self.reader_splitter = QSplitter(Qt.Vertical)
        self.reader_splitter.setChildrenCollapsible(True)
        self.reader_splitter.setMinimumWidth(0)
        self.reader_splitter.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.reader_views: List[PdfView] = []
        self._view_frames: Dict[PdfView, QFrame] = {}
        self.view = self._create_pdf_view()
        self._add_reader_view(self.view)

        self.doc_tabs = QTabBar()
        self.doc_tabs.setAutoHide(False)
        self.doc_tabs.setDocumentMode(False)
        self.doc_tabs.setExpanding(False)
        self.doc_tabs.setMinimumHeight(30)
        self.doc_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        self.eq_input.setPlaceholderText("Equation number, e.g. 5.4.13 or 5.13")
        self.eq_input.returnPressed.connect(self.jump_to_equation)

        self.status_label = QLabel("No PDF loaded")
        self.zoom_label = QLabel("100%")
        self.setStatusBar(QStatusBar(self))

        self.equation_preview_label = QLabel()
        self.equation_preview_label.setWindowFlags(Qt.ToolTip)
        self.equation_preview_label.setStyleSheet(
            "QLabel { background: white; border: 1px solid #777; padding: 6px; }"
        )
        self.equation_preview_label.hide()
        self._hover_preview_timer = QTimer(self)
        self._hover_preview_timer.setSingleShot(True)
        self._hover_preview_timer.timeout.connect(self._show_equation_preview_from_hover)
        self._hover_preview_request: Optional[Tuple[int, float, float, QPoint]] = None
        self._last_preview_key: Optional[Tuple[str, int]] = None
        self._equation_preview_enabled = False

        self.bookmarks_tree = QTreeWidget()
        self.bookmarks_tree.setHeaderHidden(True)
        self.bookmarks_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bookmarks_tree.itemActivated.connect(self._bookmark_item_activated)
        self.bookmarks_tree.itemClicked.connect(self._bookmark_item_activated)

        self.bookmarks_dock = QDockWidget("Bookmarks", self)
        self.bookmarks_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.bookmarks_dock.setMinimumWidth(BOOKMARKS_DOCK_MIN_WIDTH)
        self.bookmarks_dock.setMaximumWidth(16777215)
        self.bookmarks_dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bookmarks_dock.setWidget(self.bookmarks_tree)
        self.addDockWidget(Qt.RightDockWidgetArea, self.bookmarks_dock)
        self.bookmarks_dock.hide()

        self.search_input = SearchLineEdit()
        self.search_input.setPlaceholderText("Search text")
        self.search_input.textEdited.connect(self._text_search_edited)
        self.search_input.search_next_requested.connect(self.search_next_text)
        self.search_input.search_previous_requested.connect(self.search_previous_text)
        self.search_count_label = QLabel("0/0")
        self.search_status_label = QLabel("")
        self.search_status_label.setWordWrap(True)
        self.search_close_button = QPushButton("Close")
        self.search_close_button.clicked.connect(self.close_text_search)

        search_panel = QWidget()
        search_layout = QVBoxLayout(search_panel)
        row = QHBoxLayout()
        row.addWidget(self.search_input)
        row.addWidget(self.search_count_label)
        search_layout.addLayout(row)
        search_layout.addWidget(self.search_status_label)
        search_layout.addWidget(self.search_close_button)
        search_layout.addStretch()

        self.search_dock = QDockWidget("Search", self)
        self.search_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.search_dock.setWidget(search_panel)
        self.search_dock.visibilityChanged.connect(self._text_search_visibility_changed)
        self.addDockWidget(Qt.RightDockWidgetArea, self.search_dock)
        self.search_dock.hide()

        self.ai_api_key_input = QLineEdit()
        self.ai_api_key_input.setEchoMode(QLineEdit.Password)
        saved_api_key = str(self.ai_settings.get("api_key") or "")
        if saved_api_key:
            self.ai_api_key_input.setText(saved_api_key)
        elif openai_api_key(self.base_dir):
            self.ai_api_key_input.setPlaceholderText("Using OPENAI_API_KEY")
        else:
            self.ai_api_key_input.setPlaceholderText("OpenAI API key")
        self.ai_save_key_button = QPushButton("Save Key")
        self.ai_save_key_button.clicked.connect(self.save_ai_api_key)

        self.ai_model_combo = QComboBox()
        self.ai_model_combo.setEditable(True)
        selected_model = str(self.ai_settings.get("chat_model") or "").strip()
        if not selected_model or selected_model == "gpt-5.6-terra":
            selected_model = DEFAULT_CHAT_MODELS[0]
        model_values = list(DEFAULT_CHAT_MODELS)
        if selected_model and selected_model not in model_values:
            model_values.insert(0, selected_model)
        self.ai_model_combo.addItems(model_values)
        self.ai_model_combo.setCurrentText(selected_model)

        self.ai_page_ranges_input = QLineEdit()
        self.ai_page_ranges_input.setPlaceholderText("Pages to scan, e.g. 1-20, 45, 70-90; blank = all")
        self.ai_scan_button = QPushButton("Scan Document")
        self.ai_scan_button.clicked.connect(self.start_ai_document_scan)
        self.ai_clear_history_button = QPushButton("Clear History")
        self.ai_clear_history_button.clicked.connect(self.clear_ai_history)
        self.ai_status_label = QLabel("Enter an API key, then scan the current document.")
        self.ai_status_label.setWordWrap(True)
        self.ai_cost_label = QLabel("")
        self.ai_cost_label.setWordWrap(True)

        self.ai_chat_view = QTextBrowser()
        self.ai_chat_view.setOpenLinks(False)
        self.ai_chat_view.anchorClicked.connect(self.open_ai_reference)
        self.ai_chat_view.setMinimumHeight(240)
        self.ai_chat_view.document().setDefaultStyleSheet(
            ".ai-math-inline { "
            "font-family: 'STIX Two Math', 'Cambria Math', 'Times New Roman', serif; "
            "white-space: nowrap; color: #EDEDED; background: rgba(255,255,255,0.08); "
            "padding: 0 3px; border-radius: 3px; "
            "}"
            ".ai-math-block { "
            "font-family: 'STIX Two Math', 'Cambria Math', 'Times New Roman', serif; "
            "white-space: pre-wrap; color: #EDEDED; background: rgba(255,255,255,0.06); "
            "margin: 6px 0; padding: 6px; border-left: 3px solid #1976D2; "
            "}"
            "a { color: #42A5F5; text-decoration: none; }"
            "a:hover { text-decoration: underline; }"
        )

        self.ai_question_input = QPlainTextEdit()
        self.ai_question_input.setPlaceholderText("Ask about the current document")
        self.ai_question_input.setMaximumHeight(96)
        self.ai_bookmark_scope_combo = QComboBox()
        self.ai_bookmark_scope_combo.currentIndexChanged.connect(self._ai_bookmark_scope_changed)
        self.ai_question_ranges_input = QLineEdit()
        self.ai_question_ranges_input.setPlaceholderText("Pages for this question, e.g. 120-150; blank = all scanned pages")
        self.ai_question_ranges_input.textEdited.connect(self._ai_question_range_text_edited)
        self.ai_chunk_limit_spin = QSpinBox()
        self.ai_chunk_limit_spin.setRange(1, 60)
        self.ai_chunk_limit_spin.setValue(
            max(1, min(60, int(self.ai_settings.get("retrieval_limit") or DEFAULT_RETRIEVAL_LIMIT)))
        )
        self.ai_ask_button = QPushButton("Ask")
        self.ai_ask_button.clicked.connect(self.ask_ai_question)

        ai_panel = QWidget()
        ai_layout = QVBoxLayout(ai_panel)
        key_row = QHBoxLayout()
        key_row.addWidget(self.ai_api_key_input, 1)
        key_row.addWidget(self.ai_save_key_button)
        ai_layout.addLayout(key_row)
        ai_layout.addWidget(QLabel("Model"))
        ai_layout.addWidget(self.ai_model_combo)
        ai_layout.addWidget(self.ai_page_ranges_input)
        ai_buttons_row = QHBoxLayout()
        ai_buttons_row.addWidget(self.ai_scan_button)
        ai_buttons_row.addWidget(self.ai_clear_history_button)
        ai_layout.addLayout(ai_buttons_row)
        ai_layout.addWidget(self.ai_status_label)
        ai_layout.addWidget(self.ai_cost_label)
        ai_layout.addWidget(self.ai_chat_view, 1)
        ai_layout.addWidget(QLabel("Ask within"))
        ai_layout.addWidget(self.ai_bookmark_scope_combo)
        ai_layout.addWidget(self.ai_question_ranges_input)
        chunk_row = QHBoxLayout()
        chunk_row.addWidget(QLabel("Chunks to use"))
        chunk_row.addWidget(self.ai_chunk_limit_spin)
        chunk_row.addStretch()
        ai_layout.addLayout(chunk_row)
        ai_layout.addWidget(self.ai_question_input)
        ai_layout.addWidget(self.ai_ask_button)

        self.ai_dock = QDockWidget("AI Chat", self)
        self.ai_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.ai_dock.setMinimumWidth(320)
        self.ai_dock.setWidget(ai_panel)
        self.ai_dock.visibilityChanged.connect(self._ai_chat_visibility_changed)
        self.addDockWidget(Qt.RightDockWidgetArea, self.ai_dock)
        self.ai_dock.hide()

        self._text_search_timer = QTimer(self)
        self._text_search_timer.timeout.connect(self._scan_text_search_chunk)
        self._manual_pattern_scan_timer = QTimer(self)
        self._manual_pattern_scan_timer.timeout.connect(self._scan_manual_pattern_index_chunk)
        self._manual_pattern_scan_session_id: Optional[int] = None
        self._manual_pattern_scan_patterns: Dict[int, str] = {}
        self._manual_pattern_scan_index: Dict[str, EquationLocation] = {}
        self._manual_pattern_scan_page: int = 1

        self._create_actions_and_toolbar()
        QApplication.instance().installEventFilter(self)

        reader_page = QWidget()
        reader_page.setMinimumWidth(0)
        reader_page.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        layout = QVBoxLayout(reader_page)
        layout.addWidget(self.doc_tabs)
        layout.addWidget(self.reader_splitter)

        bottom = QHBoxLayout()
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        bottom.addWidget(self.zoom_label)
        layout.addLayout(bottom)

        self.central_stack = QStackedWidget()
        self.central_stack.setMinimumWidth(0)
        self.central_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.central_stack.addWidget(reader_page)
        self.central_stack.addWidget(self._create_menu_page())
        self.setCentralWidget(self.central_stack)

        self._restore_startup_session()
        self._refresh_menu_documents()
        self._refresh_bookmarks_panel()
        self.view.setFocus()

    def _create_pdf_view(self) -> PdfView:
        view = PdfView()
        view.setMinimumSize(0, READER_PANE_MIN_HEIGHT)
        view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        view.view_changed.connect(self._on_view_changed)
        view.clicked.connect(lambda page, x, y, v=view: self._activate_and_call(v, self.jump_if_reference_clicked, page, x, y))
        view.right_clicked.connect(lambda page, x, y, v=view: self._activate_and_call(v, self.research_equation_reference_clicked, page, x, y))
        view.link_click_requested.connect(lambda page, x, y, v=view: self._activate_and_call(v, self.open_link_clicked, page, x, y))
        view.selection_line_requested.connect(lambda page, x, y, v=view: self._activate_and_call(v, self.add_selection_line_at_position, page, x, y))
        view.selection_context_menu_requested.connect(lambda pos, v=view: self._activate_and_call(v, self.show_selected_rectangle_context_menu, pos))
        view.pdf_files_dropped.connect(lambda paths, v=view: self._activate_and_call(v, self.open_dropped_pdfs, paths))
        view.hover_moved.connect(lambda page, x, y, pos, v=view: self._schedule_equation_preview(page, x, y, pos) if v is self.view else None)
        view.hover_left.connect(self._hide_equation_preview)
        view.comment_clicked.connect(lambda index, v=view: self._activate_and_call(v, self.open_comment_at_index, index))
        view.comment_context_menu_requested.connect(
            lambda index, pos, v=view: self._activate_and_call(v, self.show_comment_context_menu, index, pos)
        )
        view._pdf_reader_session_id = None
        return view

    def _add_reader_view(self, view: PdfView, index: Optional[int] = None) -> None:
        frame = QFrame()
        frame.setObjectName("readerPaneFrame")
        frame.setMinimumSize(0, READER_PANE_MIN_HEIGHT)
        frame.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(2, 2, 2, 2)
        frame_layout.setSpacing(0)
        frame_layout.addWidget(view)

        self._view_frames[view] = frame
        if index is None or index >= self.reader_splitter.count():
            self.reader_views.append(view)
            self.reader_splitter.addWidget(frame)
        else:
            self.reader_views.insert(index, view)
            self.reader_splitter.insertWidget(index, frame)
        self._refresh_active_view_styles()

    def _activate_and_call(self, view: PdfView, handler, *args) -> None:
        self._set_active_view(view)
        handler(*args)

    def _set_active_view(self, view: PdfView, focus: bool = False) -> None:
        if view not in self.reader_views:
            return
        if self.view is not view:
            self._save_active_session_view()
            self.view = view
            self._sync_text_search_panel()
            self._redraw_selection_lines()
            session = self.current_session()
            if session is not None:
                self._ensure_rendered_visible_pages(session, view)
                self._capture_session_view_layout(session)
        self._refresh_active_view_styles()
        self._update_status()
        if focus:
            view.setFocus()

    def _refresh_active_view_styles(self) -> None:
        for view, frame in self._view_frames.items():
            color = "#1976D2" if len(self.reader_views) > 1 and view is self.view else "transparent"
            frame.setStyleSheet(
                "QFrame#readerPaneFrame { "
                f"border: 2px solid {color}; "
                "padding: 0px; "
                "background: transparent; "
                "}"
            )

    def _view_for_event_object(self, obj) -> Optional[PdfView]:
        for view in self.reader_views:
            if obj is view or obj is view.viewport():
                return view
        return None

    def _clear_reader_views(self) -> None:
        for view, frame in list(self._view_frames.items()):
            frame.setParent(None)
            frame.deleteLater()
            view._pdf_reader_session_id = None
        self.reader_views = []
        self._view_frames = {}

    def _capture_session_view_layout(self, session: DocumentSession) -> None:
        states: List[ViewState] = []
        active_index = 0
        for view in self.reader_views:
            if not self._view_has_session(session, view):
                continue
            if view is self.view:
                active_index = len(states)
            state = self._view_state_for_view(view)
            if state is not None:
                states.append(state)

        if not states:
            state = self._current_view_state()
            if state is not None:
                states = [state]
                active_index = 0

        if not states:
            return

        active_index = max(0, min(active_index, len(states) - 1))
        session.viewer_states = states[:3]
        session.active_view_index = active_index
        session.saved_view_state = session.viewer_states[active_index]
        session.current_page = session.saved_view_state.page

    def _rebuild_reader_views_for_session(self, session: DocumentSession) -> None:
        states = session.viewer_states[:3]
        if not states:
            state = session.saved_view_state or self._default_state_for_session(session)
            states = [state]

        active_index = max(0, min(session.active_view_index, len(states) - 1))
        self._clear_reader_views()

        for state in states:
            view = self._create_pdf_view()
            self._add_reader_view(view)
            self._show_session_in_view(view, session)
            self._apply_view_state_to_view(view, state)

        self.view = self.reader_views[active_index]
        self._refresh_active_view_styles()
        self._ensure_rendered_visible_pages(session, self.view)
        session.saved_view_state = states[active_index]
        session.current_page = states[active_index].page

    def _create_action(self, text: str, shortcut: str, handler, shortcut_id: Optional[str] = None) -> QAction:
        action = QAction(text, self)
        active_shortcut = self._shortcut_for(shortcut_id, shortcut)
        action.setShortcut(QKeySequence(active_shortcut) if active_shortcut else QKeySequence())
        action.triggered.connect(handler)
        self.addAction(action)
        if shortcut_id:
            self._shortcut_actions[shortcut_id] = action
        self._register_layout_independent_shortcut(action, active_shortcut)
        return action

    def _shortcut_for(self, shortcut_id: Optional[str], default_shortcut: str) -> str:
        if shortcut_id is None:
            return default_shortcut
        return self.shortcut_overrides.get(shortcut_id, default_shortcut)

    def _current_shortcuts(self) -> Dict[str, str]:
        return {
            shortcut_id: self.shortcut_overrides.get(shortcut_id, default_shortcut)
            for shortcut_id, default_shortcut in self._shortcut_defaults.items()
        }

    def _shortcut_definitions_for_dialog(self) -> List[Tuple[str, str, str]]:
        return [
            (shortcut_id, title, default_shortcut)
            for shortcut_id, title, default_shortcut in SHORTCUT_DEFINITIONS
            if shortcut_id in self._shortcut_actions
        ]

    def _apply_shortcut_overrides(self) -> None:
        self._layout_independent_shortcuts.clear()
        for shortcut_id, action in self._shortcut_actions.items():
            default_shortcut = self._shortcut_defaults.get(shortcut_id, "")
            shortcut = self.shortcut_overrides.get(shortcut_id, default_shortcut)
            if shortcut_id == "toggle_last_documents" and shortcut_id not in self.shortcut_overrides:
                action.setShortcuts([QKeySequence(value) for value in SHORTCUT_TOGGLE_LAST_DOCUMENTS])
                for value in SHORTCUT_TOGGLE_LAST_DOCUMENTS:
                    self._register_layout_independent_shortcut(action, value)
                continue
            elif shortcut:
                action.setShortcut(QKeySequence(shortcut))
            else:
                action.setShortcut(QKeySequence())
            self._register_layout_independent_shortcut(action, shortcut)

    def _shortcut_conflict_message(self, shortcuts: Dict[str, str]) -> Optional[str]:
        seen: Dict[str, str] = {}
        labels = {shortcut_id: title for shortcut_id, title, _default in SHORTCUT_DEFINITIONS}
        for shortcut_id, shortcut in shortcuts.items():
            normalized = shortcut.strip().lower()
            if not normalized:
                continue
            if normalized in seen:
                return (
                    f"{labels.get(shortcut_id, shortcut_id)} and "
                    f"{labels.get(seen[normalized], seen[normalized])} both use {shortcut}."
                )
            seen[normalized] = shortcut_id
        return None

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop):
            if self._handle_window_pdf_drop_event(obj, event):
                return True
        if event.type() == QEvent.MouseButtonPress:
            view = self._view_for_event_object(obj)
            if view is not None:
                self._set_active_view(view, focus=True)
        if event.type() == QEvent.KeyPress and self._handle_document_tab_shortcut(event):
            return True
        if event.type() == QEvent.KeyPress and self._handle_layout_independent_shortcut(event):
            return True
        return super().eventFilter(obj, event)

    def _handle_window_pdf_drop_event(self, obj, event) -> bool:
        if QApplication.activeModalWidget() is not None:
            return False
        if not isinstance(obj, QWidget):
            return False
        if obj is not self and not self.isAncestorOf(obj):
            return False

        paths = pdf_paths_from_mime_data(event.mimeData())
        if not paths:
            return False

        if event.type() in (QEvent.DragEnter, QEvent.DragMove):
            event.acceptProposedAction()
            return True
        if event.type() == QEvent.Drop:
            event.acceptProposedAction()
            self.open_dropped_pdfs(paths)
            return True
        return False

    def _handle_document_tab_shortcut(self, event) -> bool:
        if QApplication.activeWindow() is not self or len(self.sessions) < 2:
            return False
        if event.key() not in (Qt.Key_Tab, Qt.Key_Backtab):
            return False

        relevant_modifiers = (
            Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier
        )
        event_modifiers = event.modifiers() & relevant_modifiers
        if event.key() == Qt.Key_Backtab:
            if event_modifiers != Qt.ShiftModifier:
                return False
            direction = -1
        elif event_modifiers == Qt.NoModifier:
            direction = 1
        elif event_modifiers == Qt.ShiftModifier:
            direction = -1
        else:
            return False

        self.switch_to_adjacent_document(direction)
        event.accept()
        return True

    def _register_layout_independent_shortcut(self, action: QAction, shortcut: str) -> None:
        if shortcut.endswith("++"):
            parts = shortcut[:-2].split("+") + ["+"]
        else:
            parts = shortcut.split("+")
        if not parts:
            return

        key_name = parts[-1].upper()
        if key_name not in SHORTCUT_KEY_CODES:
            return

        modifiers = Qt.NoModifier
        for part in parts[:-1]:
            normalized = part.lower()
            if normalized in ("meta", "cmd", "command"):
                modifiers |= Qt.MetaModifier
            elif normalized in ("ctrl", "control"):
                modifiers |= Qt.ControlModifier
            elif normalized == "alt":
                modifiers |= Qt.AltModifier
            elif normalized == "shift":
                modifiers |= Qt.ShiftModifier

        self._layout_independent_shortcuts.append((action, modifiers, key_name))

    def _handle_layout_independent_shortcut(self, event) -> bool:
        if not self._layout_independent_shortcuts or QApplication.activeWindow() is not self:
            return False

        key_name = self._layout_independent_key_name(event)
        if not key_name:
            return False

        relevant_modifiers = (
            Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier
        )
        event_modifiers = event.modifiers() & relevant_modifiers
        focus_widget = QApplication.focusWidget()
        text_input_has_focus = isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit))

        for action, modifiers, shortcut_key_name in self._layout_independent_shortcuts:
            if key_name != shortcut_key_name or event_modifiers != modifiers:
                continue
            if text_input_has_focus and modifiers == Qt.ShiftModifier:
                return False
            if not action.isEnabled():
                return False

            action.trigger()
            event.accept()
            return True

        return False

    @staticmethod
    def _layout_independent_key_name(event) -> Optional[str]:
        for key_name, qt_key in SHORTCUT_KEY_CODES.items():
            if event.key() == qt_key:
                return key_name

        if sys.platform == "darwin":
            native_key = event.nativeVirtualKey()
            for key_name, native_virtual_key in MACOS_NATIVE_VIRTUAL_KEYS.items():
                if native_key == native_virtual_key:
                    return key_name

        text = event.text()
        if not text:
            return None

        if len(text) == 1:
            return KEYBOARD_LAYOUT_FALLBACKS.get(text, text.upper())

        return None

    @staticmethod
    def _path_key(path: str) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return path

    @staticmethod
    def _pdf_identity_for_path(path: str) -> Optional[dict]:
        file_path = Path(path)
        if not file_path.exists():
            return None

        try:
            stat = file_path.stat()
            size = int(stat.st_size)
        except Exception:
            size = 0

        pages = 0
        first_rect = ""
        pdf_format = "PDF"
        title = ""
        try:
            doc = fitz.open(str(file_path))
            try:
                pages = len(doc)
                if pages:
                    rect = doc.load_page(0).rect
                    first_rect = f"{rect.width:.2f}x{rect.height:.2f}"
                metadata = dict(doc.metadata or {})
                pdf_format = metadata.get("format") or "PDF"
                title = metadata.get("title") or ""
            finally:
                doc.close()
        except Exception:
            return None

        return {
            "name": file_path.name,
            "size": size,
            "pages": pages,
            "first_page_rect": first_rect,
            "format": pdf_format,
            "title": title,
        }

    @staticmethod
    def _identity_from_entry(doc_entry: Optional[dict]) -> dict:
        if not isinstance(doc_entry, dict):
            return {}

        identity = doc_entry.get("file_identity")
        if isinstance(identity, dict):
            return dict(identity)

        signature = doc_entry.get("equation_index_signature")
        if not isinstance(signature, dict):
            return {}

        out = {
            key: signature.get(key)
            for key in ("size", "pages", "first_page_rect", "format", "title")
            if signature.get(key) not in (None, "")
        }
        raw_path = doc_entry.get("path")
        if isinstance(raw_path, str) and raw_path:
            out["name"] = Path(raw_path).name
        return out

    @staticmethod
    def _pdf_identity_matches(path: Path, identity: dict) -> bool:
        if not identity:
            return False
        expected_name = identity.get("name")
        if expected_name and path.name != expected_name:
            return False

        candidate = PdfReaderWindow._pdf_identity_for_path(str(path))
        if not candidate:
            return False

        for key in ("size", "pages", "first_page_rect", "format", "title"):
            expected = identity.get(key)
            if expected in (None, ""):
                continue
            if candidate.get(key) != expected:
                return False
        return True

    def _resolve_document_path(self, path: str, doc_entry: Optional[dict] = None) -> Optional[str]:
        file_path = Path(path)
        if file_path.exists():
            return str(file_path)

        identity = self._identity_from_entry(doc_entry)
        if "name" not in identity and file_path.name:
            identity["name"] = file_path.name
        if not identity:
            return None

        search_root = file_path.parent
        while not search_root.exists() and search_root != search_root.parent:
            search_root = search_root.parent

        if not search_root.exists():
            return None

        target_name = str(identity.get("name") or file_path.name)
        try:
            candidates = search_root.rglob(target_name)
            for candidate in candidates:
                if candidate.is_file() and candidate.suffix.lower() == ".pdf":
                    if self._pdf_identity_matches(candidate, identity):
                        return str(candidate)
        except Exception as exc:
            print(f"[PDF Reader] Could not search for moved PDF under {search_root}: {exc}", flush=True)

        return None

    def _replace_recent_document_path(self, old_path: str, new_path: str, entry: Optional[dict] = None) -> None:
        old_key = self._path_key(old_path)
        new_key = self._path_key(new_path)

        updated: List[str] = []
        seen = set()
        for existing in self.recent_documents:
            existing_key = self._path_key(existing)
            candidate = new_path if existing_key == old_key else existing
            candidate_key = self._path_key(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            updated.append(candidate)
        self.recent_documents = updated

        if entry is None:
            entry = self.recent_document_entries.get(old_key)
        if entry:
            repaired = dict(entry)
            repaired["path"] = new_path
            identity = self._pdf_identity_for_path(new_path)
            if identity:
                repaired["file_identity"] = identity
            self.recent_document_entries[new_key] = repaired
        if old_key in self.recent_document_entries and old_key != new_key:
            self.recent_document_entries.pop(old_key, None)

    @staticmethod
    def _strip_comment_metadata(entry: dict) -> dict:
        clean = dict(entry)
        clean.pop("comments", None)
        return clean

    @staticmethod
    def _recent_documents_from_session_data(session_data: Optional[SessionPersistence]) -> Tuple[List[str], Dict[str, dict]]:
        if session_data is None:
            return [], {}

        paths: List[str] = []
        entries: Dict[str, dict] = {}
        for recent_entry in session_data.recent_documents:
            if isinstance(recent_entry, str) and recent_entry:
                paths.append(recent_entry)
                continue
            if not isinstance(recent_entry, dict):
                continue
            raw_path = recent_entry.get("path")
            if isinstance(raw_path, str) and raw_path:
                paths.append(raw_path)
                entries[PdfReaderWindow._path_key(raw_path)] = PdfReaderWindow._strip_comment_metadata(recent_entry)
        for doc_entry in session_data.documents:
            if not isinstance(doc_entry, dict):
                continue
            raw_path = doc_entry.get("path")
            if isinstance(raw_path, str) and raw_path:
                paths.append(raw_path)
                entries[PdfReaderWindow._path_key(raw_path)] = PdfReaderWindow._strip_comment_metadata(doc_entry)

        out: List[str] = []
        seen = set()
        for raw_path in paths:
            try:
                key = PdfReaderWindow._path_key(raw_path)
            except Exception:
                key = raw_path
            if key in seen:
                continue
            seen.add(key)
            out.append(raw_path)
        return out[:40], entries

    def _remember_recent_document(self, path: str, save: bool = True) -> None:
        normalized = self._path_key(path)

        kept: List[str] = []
        for existing in self.recent_documents:
            existing_key = self._path_key(existing)
            if existing_key != normalized:
                kept.append(existing)

        self.recent_documents = [path] + kept
        self.recent_documents = self.recent_documents[:40]
        session_index = self._find_session_by_path(path)
        if session_index is not None:
            self.recent_document_entries[normalized] = self._session_persistence_entry(self.sessions[session_index])
        if hasattr(self, "menu_list_layout"):
            self._refresh_menu_documents()
        if save:
            self._save_session_state_now()

    def _create_menu_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        title = QLabel("Recent Documents")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        top_row.addWidget(title)
        top_row.addStretch()
        create_button = QPushButton("Create Document")
        create_button.clicked.connect(self.create_new_document)
        top_row.addWidget(create_button)
        open_button = QPushButton("Open PDFs")
        open_button.clicked.connect(self.open_multiple_pdfs)
        top_row.addWidget(open_button)
        reader_button = QPushButton("Reader")
        reader_button.clicked.connect(self.show_reader)
        top_row.addWidget(reader_button)
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.open_macro_settings)
        top_row.addWidget(settings_button)
        shortcuts_button = QPushButton("Shortcuts")
        shortcuts_button.clicked.connect(self.open_shortcut_settings)
        top_row.addWidget(shortcuts_button)
        self.google_drive_button = QPushButton()
        self.google_drive_button.clicked.connect(self.toggle_google_drive_login)
        top_row.addWidget(self.google_drive_button)
        refresh_cloud_button = QPushButton("Refresh Cloud")
        refresh_cloud_button.clicked.connect(self.refresh_cloud_documents)
        top_row.addWidget(refresh_cloud_button)
        upload_cloud_button = QPushButton("Upload Cloud")
        upload_cloud_button.clicked.connect(self.upload_pdfs_to_cloud)
        top_row.addWidget(upload_cloud_button)
        layout.addLayout(top_row)
        self._update_google_drive_button()

        self.menu_search_input = QLineEdit()
        self.menu_search_input.setPlaceholderText("Search documents")
        self.menu_search_input.textChanged.connect(self._refresh_menu_documents)
        layout.addWidget(self.menu_search_input)

        self.menu_scroll_area = QScrollArea()
        self.menu_scroll_area.setWidgetResizable(True)
        self.menu_list_widget = QWidget()
        self.menu_list_layout = QVBoxLayout(self.menu_list_widget)
        self.menu_list_layout.setSpacing(10)
        self.menu_list_layout.addStretch()
        self.menu_scroll_area.setWidget(self.menu_list_widget)
        layout.addWidget(self.menu_scroll_area, 1)
        return page

    def show_menu(self) -> None:
        self._save_active_session_view()
        self._refresh_menu_documents()
        if self.google_drive.is_logged_in and not self.cloud_documents:
            self.refresh_cloud_documents(show_errors=False)
        self.central_stack.setCurrentIndex(1)
        self.menu_search_input.setFocus()

    def show_reader(self) -> None:
        self.central_stack.setCurrentIndex(0)
        self.view.setFocus()

    def _view_state_for_view(self, view: PdfView) -> Optional[ViewState]:
        session = self.current_session()
        if session is None:
            return None
        visible_page = view.current_visible_page()
        return ViewState(
            page=visible_page or session.current_page,
            scroll_x=view.horizontalScrollBar().value(),
            scroll_y=view.verticalScrollBar().value(),
            zoom=view.zoom_factor(),
        )

    def _apply_view_state_to_view(self, view: PdfView, state: ViewState) -> None:
        view.set_zoom_factor(state.zoom)
        view.horizontalScrollBar().setValue(state.scroll_x)
        view.verticalScrollBar().setValue(state.scroll_y)
        if view.current_visible_page() != state.page:
            view.scroll_to_page(state.page)

    def _show_session_in_view(self, view: PdfView, session: DocumentSession) -> None:
        self._ensure_rendered_pages(session)
        view.set_document_images(
            session.rendered_pages,
            preserve_zoom=True,
            page_sizes=session.page_display_sizes,
        )
        view._pdf_reader_session_id = id(session)
        view.set_cover_overlays(session.covers)
        view.set_comment_overlays(self._display_comments_for_session(session), visible_count=len(session.comments))
        view.show_selection_lines(session.selection_lines)

    def split_active_viewer(self) -> None:
        session = self.current_session()
        if session is None:
            return
        if len(self.reader_views) >= 3:
            self.statusBar().showMessage("Already split into 3 viewers", 3000)
            return

        state = self._view_state_for_view(self.view) or self._default_state_for_session(session)
        active_index = self.reader_splitter.indexOf(self._view_frames[self.view])
        new_view = self._create_pdf_view()
        self._add_reader_view(new_view, active_index + 1)
        self._show_session_in_view(new_view, session)
        self._apply_view_state_to_view(new_view, state)
        self._set_active_view(new_view, focus=True)
        self._ensure_rendered_visible_pages(session, new_view)
        self._capture_session_view_layout(session)
        self.statusBar().showMessage(f"Split into {len(self.reader_views)} viewers", 3000)

    def rotate_viewer_segments(self) -> None:
        session = self.current_session()
        if session is None:
            return

        views = [view for view in self.reader_views if self._view_has_session(session, view)]
        if len(views) < 2:
            self.statusBar().showMessage("Split the viewer first to rotate segments", 3000)
            return

        states = [self._view_state_for_view(view) for view in views]
        if any(state is None for state in states):
            return

        rotated_states = states[1:] + states[:1]
        self._suspend_view_updates = True
        try:
            for view, state in zip(views, rotated_states):
                self._apply_view_state_to_view(view, state)
                self._ensure_rendered_visible_pages(session, view)
        finally:
            self._suspend_view_updates = False

        self._capture_session_view_layout(session)
        self._update_status()
        self.statusBar().showMessage(f"Rotated {len(views)} viewer segments", 3000)

    def close_active_viewer(self) -> None:
        if len(self.reader_views) <= 1:
            self.statusBar().showMessage("Cannot close the only viewer", 3000)
            return

        closing_view = self.view
        closing_frame = self._view_frames[closing_view]
        closing_index = self.reader_splitter.indexOf(closing_frame)
        next_index = min(closing_index, len(self.reader_views) - 2)

        self.reader_views.remove(closing_view)
        self._view_frames.pop(closing_view, None)
        closing_frame.setParent(None)
        closing_frame.deleteLater()

        next_view = self.reader_views[next_index]
        self.view = next_view
        self._refresh_active_view_styles()
        self._sync_text_search_panel()
        self._redraw_selection_lines()
        self._update_status()
        session = self.current_session()
        if session is not None:
            self._capture_session_view_layout(session)
        next_view.setFocus()
        self.statusBar().showMessage(f"Closed viewer; {len(self.reader_views)} remaining", 3000)

    def open_macro_settings(self) -> None:
        dialog = MacroSettingsDialog(self.equation_macros, self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.equation_macros = dialog.macros()
        self._save_session_state_now()
        self.statusBar().showMessage(f"Saved {len(self.equation_macros)} equation macro(s)", 5000)

    def open_shortcut_settings(self) -> None:
        dialog = ShortcutSettingsDialog(
            self._shortcut_definitions_for_dialog(),
            self._current_shortcuts(),
            self,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        shortcuts = dialog.shortcuts()
        conflict = self._shortcut_conflict_message(shortcuts)
        if conflict:
            QMessageBox.warning(self, "Shortcut conflict", conflict)
            return

        self.shortcut_overrides = {
            shortcut_id: shortcut
            for shortcut_id, shortcut in shortcuts.items()
            if shortcut_id in self._shortcut_defaults
            and shortcut != self._shortcut_defaults[shortcut_id]
        }
        self._apply_shortcut_overrides()
        self._save_session_state_now()
        self.statusBar().showMessage("Saved shortcuts", 5000)

    def _update_google_drive_button(self) -> None:
        if hasattr(self, "google_drive_button"):
            self.google_drive_button.setText("Sign Out Drive" if self.google_drive.is_logged_in else "Sign In Drive")

    def toggle_google_drive_login(self) -> None:
        if self.google_drive.is_logged_in:
            self.google_drive.sign_out()
            self.cloud_documents = []
            self._cloud_thumbnail_cache.clear()
            self._update_google_drive_button()
            self._refresh_menu_documents()
            self.statusBar().showMessage("Signed out of Google Drive", 5000)
            return

        try:
            self.google_drive.authenticate(
                lambda url: QDesktopServices.openUrl(QUrl(url)),
                QApplication.processEvents,
            )
            self._update_google_drive_button()
            self.refresh_cloud_documents(show_errors=True)
            self.statusBar().showMessage("Signed in to Google Drive", 5000)
        except GoogleDriveError as exc:
            QMessageBox.warning(self, "Google Drive", str(exc))
        finally:
            self._update_google_drive_button()

    def refresh_cloud_documents(self, show_errors: bool = True) -> None:
        if not self.google_drive.is_logged_in:
            if show_errors:
                QMessageBox.information(self, "Google Drive", "Sign in to Google Drive first.")
            self._refresh_menu_documents()
            return

        try:
            self.statusBar().showMessage(f"Loading Google Drive documents from {DRIVE_FOLDER_NAME}...", 0)
            QApplication.processEvents()
            self.cloud_documents = self.google_drive.list_documents(QApplication.processEvents)
            self._refresh_menu_documents()
            self.statusBar().showMessage(
                f"Loaded {len(self.cloud_documents)} cloud document(s) from {DRIVE_FOLDER_NAME}",
                5000,
            )
        except GoogleDriveError as exc:
            if show_errors:
                QMessageBox.warning(self, "Google Drive", str(exc))

    def _cloud_document_thumbnail(self, document: CloudDocument) -> QPixmap:
        if document.id in self._cloud_thumbnail_cache:
            return self._cloud_thumbnail_cache[document.id]

        placeholder = QPixmap(96, 128)
        placeholder.fill(QColor("#eef5ff"))
        painter = QPainter(placeholder)
        painter.setPen(QPen(QColor("#3567a8")))
        painter.drawRect(0, 0, 95, 127)
        painter.drawText(placeholder.rect(), Qt.AlignCenter, "Drive")
        painter.end()
        self._cloud_thumbnail_cache[document.id] = placeholder
        return placeholder

    def _open_cloud_document(self, document: CloudDocument) -> None:
        try:
            cached_path = self.google_drive.cached_document_path(document)
            if cached_path is not None:
                self.statusBar().showMessage(f"Opening cached cloud document: {document.name}", 3000)
                local_path = cached_path
            else:
                self.statusBar().showMessage(f"Downloading {document.name} from Google Drive...", 0)
                QApplication.processEvents()
                local_path = self.google_drive.download_document(document)
        except GoogleDriveError as exc:
            QMessageBox.warning(self, "Google Drive", str(exc))
            return

        self.load_pdf(str(local_path), switch_to=True)
        self.show_reader()

    def upload_pdfs_to_cloud(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Upload PDFs to Google Drive",
            "",
            "PDF Files (*.pdf)",
        )
        if not paths:
            return
        self.upload_paths_to_cloud(paths)

    def upload_paths_to_cloud(self, paths: List[str]) -> None:
        if not self.google_drive.is_logged_in:
            QMessageBox.information(self, "Google Drive", "Sign in to Google Drive before uploading.")
            return

        uploaded: List[CloudDocument] = []
        try:
            for index, path in enumerate(paths, start=1):
                self.statusBar().showMessage(
                    f"Uploading {Path(path).name} to Google Drive ({index}/{len(paths)})...",
                    0,
                )
                QApplication.processEvents()
                uploaded.append(self.google_drive.upload_pdf(Path(path)))
        except GoogleDriveError as exc:
            QMessageBox.warning(self, "Google Drive", str(exc))
            return

        self.cloud_documents = uploaded + [
            document
            for document in self.cloud_documents
            if document.id not in {uploaded_doc.id for uploaded_doc in uploaded}
        ]
        self.refresh_cloud_documents(show_errors=False)
        self.statusBar().showMessage(f"Uploaded {len(uploaded)} PDF file(s) to Google Drive", 6000)

    def _show_local_document_context_menu(self, button: QPushButton, pos: QPoint, path: str) -> None:
        menu = QMenu(self)
        open_action = menu.addAction("Open")
        upload_action = menu.addAction("Upload to Cloud")
        chosen = menu.exec(button.mapToGlobal(pos))
        if chosen == open_action:
            self._open_recent_document(path)
        elif chosen == upload_action:
            self.upload_paths_to_cloud([path])

    def create_new_document(self) -> None:
        dialog = DocumentBuilderDialog(self)
        if dialog.exec() != QDialog.Accepted or not dialog.saved_path:
            self.menu_search_input.setFocus()
            return
        self.open_created_document(dialog.saved_path)
        self.statusBar().showMessage(f"Created new document: {dialog.saved_path}", 7000)

    def open_created_document(self, path: str) -> None:
        existing_index = self._find_session_by_path(path)
        if existing_index is not None:
            self.close_document_at_index(existing_index)
        self.load_pdf(path, switch_to=True)

    def _selected_rectangle_pdf_rect(self) -> Optional[Tuple[int, Tuple[float, float, float, float]]]:
        selected = self.view.selected_learning_rect()
        if selected is None:
            return None
        page, x0, y0, x1, y1 = selected
        rect = (
            min(x0, x1) / RENDER_SCALE,
            min(y0, y1) / RENDER_SCALE,
            max(x0, x1) / RENDER_SCALE,
            max(y0, y1) / RENDER_SCALE,
        )
        return page, rect

    def _refresh_annotation_overlays(self, session: DocumentSession) -> None:
        comments = self._display_comments_for_session(session)
        for view in self.reader_views:
            if self._view_has_session(session, view):
                view.set_cover_overlays(session.covers)
                view.set_comment_overlays(comments, visible_count=len(session.comments))

    def _visible_saved_comment_indices(self, session: DocumentSession) -> List[int]:
        return [
            index
            for index in range(len(session.saved_pdf_comments))
            if index not in session.saved_comment_removals
        ]

    def _display_comments_for_session(self, session: DocumentSession) -> List[PdfComment]:
        saved_comments = [
            session.saved_comment_edits.get(index, session.saved_pdf_comments[index])
            for index in self._visible_saved_comment_indices(session)
        ]
        return session.comments + saved_comments

    def _comment_source_for_display_index(self, session: DocumentSession, index: int) -> Optional[Tuple[str, int]]:
        if index < 0:
            return None
        if index < len(session.comments):
            return "pending", index
        saved_position = index - len(session.comments)
        visible_saved = self._visible_saved_comment_indices(session)
        if not (0 <= saved_position < len(visible_saved)):
            return None
        return "saved", visible_saved[saved_position]

    def _repair_pdf_in_place_for_append(self, session: DocumentSession) -> bool:
        if session.doc.can_save_incrementally():
            return True

        source_path = Path(session.path)
        tmp_path: Optional[Path] = None

        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{source_path.stem}.repair-",
                suffix=source_path.suffix,
                dir=str(source_path.parent),
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)

            session.doc.save(str(tmp_path), garbage=4, deflate=True)
            session.doc.close()
            tmp_path.replace(source_path)
            session.doc = fitz.open(session.path)
            if not session.doc.can_save_incrementally():
                raise RuntimeError("The repaired PDF still cannot accept appended incremental updates.")
            session.saved_pdf_comments = self._load_pdf_reader_comments_from_pdf(session.doc)
            session.rendered_pages = []
            session.rendered_page_numbers.clear()
            session.page_display_sizes = []
            self._ensure_rendered_pages(session)
            index = self.sessions.index(session)
            self.doc_tabs.setTabText(index, self._session_display_name(session.path))
            self.doc_tabs.setTabToolTip(index, session.path)
            self._refresh_annotation_overlays(session)
            for view in self.reader_views:
                if not self._view_has_session(session, view):
                    continue
                state = self._view_state_for_view(view)
                self._show_session_in_view(view, session)
                if state is not None:
                    self._apply_view_state_to_view(view, state)
            self._remember_recent_document(session.path, save=False)
            self.statusBar().showMessage("Repaired PDF structure; saving comments by appended update...", 7000)
            return True
        except Exception as exc:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                if session.doc.is_closed:
                    session.doc = fitz.open(session.path)
            except Exception:
                pass
            QMessageBox.warning(self, "Repair PDF", f"Could not repair the PDF for appended comments:\n{exc}")
            self.view.setFocus()
            return False

    def show_selected_rectangle_context_menu(self, global_pos: QPoint) -> None:
        session = self.current_session()
        selected = self._selected_rectangle_pdf_rect()
        if session is None or selected is None:
            return

        menu = QMenu(self)
        comment_action = menu.addAction("Add Comment")
        learn_action = menu.addAction("Learn Equation Rectangle")
        copy_text_action = menu.addAction("Copy Text in Box")
        menu.addSeparator()

        cover_menu = menu.addMenu("Cover with Color")
        cover_white_action = cover_menu.addAction("White")
        cover_black_action = cover_menu.addAction("Black")
        cover_gray_action = cover_menu.addAction("Light Gray")
        cover_yellow_action = cover_menu.addAction("Yellow")
        cover_custom_action = cover_menu.addAction("Custom...")
        remove_covers_action = menu.addAction("Remove Covers in Box")

        chosen = menu.exec(global_pos)
        if chosen is None:
            self.view.setFocus()
            return
        if chosen == comment_action:
            self.create_comment_from_selected_rectangle()
        elif chosen == learn_action:
            self.learn_equation_format_from_selected_rectangle()
        elif chosen == copy_text_action:
            self.copy_text_from_selected_rectangle()
        elif chosen == remove_covers_action:
            self.remove_covers_in_selected_rectangle()
        elif chosen in (
            cover_white_action,
            cover_black_action,
            cover_gray_action,
            cover_yellow_action,
            cover_custom_action,
        ):
            color_by_action = {
                cover_white_action: "#FFFFFF",
                cover_black_action: "#000000",
                cover_gray_action: "#E5E7EB",
                cover_yellow_action: "#FFF59D",
            }
            color = color_by_action.get(chosen)
            if color is None:
                picked = QColorDialog.getColor(QColor("#FFFFFF"), self, "Cover Color")
                if not picked.isValid():
                    self.view.setFocus()
                    return
                color = picked.name()
            self.cover_selected_rectangle(color)
        self.view.setFocus()

    def cover_selected_rectangle(self, color: str) -> None:
        session = self.current_session()
        selected = self._selected_rectangle_pdf_rect()
        if session is None or selected is None:
            self.statusBar().showMessage("Select a rectangle first", 4000)
            return
        page, rect = selected
        session.covers.append(PdfCover(page=page, rect=rect, color=color))
        self._refresh_annotation_overlays(session)
        self._save_session_state_now()
        self.statusBar().showMessage(f"Covered selected box on page {page}", 5000)

    def remove_covers_in_selected_rectangle(self) -> None:
        session = self.current_session()
        selected = self._selected_rectangle_pdf_rect()
        if session is None or selected is None:
            return
        page, rect = selected
        selection_rect = fitz.Rect(rect)
        before = len(session.covers)
        session.covers = [
            cover
            for cover in session.covers
            if cover.page != page or not fitz.Rect(cover.rect).intersects(selection_rect)
        ]
        removed = before - len(session.covers)
        self._refresh_annotation_overlays(session)
        self._save_session_state_now()
        self.statusBar().showMessage(f"Removed {removed} cover{'s' if removed != 1 else ''}", 5000)

    def copy_text_from_selected_rectangle(self) -> None:
        session = self.current_session()
        selected = self._selected_rectangle_pdf_rect()
        if session is None or selected is None:
            return
        page_number, rect = selected
        try:
            page = session.doc.load_page(page_number - 1)
            text = page.get_text("text", clip=fitz.Rect(rect) & page.rect, sort=True).strip()
        except Exception as exc:
            self.statusBar().showMessage(f"Could not copy selected text: {exc}", 6000)
            return
        if not text:
            self.statusBar().showMessage("No extractable text in selected box", 4000)
            return
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"Copied text from selected box on page {page_number}", 5000)

    def create_comment_from_selected_rectangle(self) -> None:
        session = self.current_session()
        if session is None:
            return

        selected = self._selected_rectangle_pdf_rect()
        if selected is None:
            self.statusBar().showMessage(
                "Hold left-click and drag a rectangle, then press Ctrl+T to comment on it",
                7000,
            )
            return

        page, rect = selected
        dialog = CommentDialog(self.equation_macros, self)
        if dialog.exec() != QDialog.Accepted:
            self.view.setFocus()
            return

        if not dialog.has_comment_text():
            self.statusBar().showMessage("Empty comment was not saved", 4000)
            self.view.setFocus()
            return
        text = dialog.comment_text()

        session.comments.append(PdfComment(page=page, rect=rect, text=text, color=dialog.comment_color()))
        self._refresh_annotation_overlays(session)
        self._save_session_state_now()
        self.statusBar().showMessage(f"Saved comment on page {page}", 5000)
        self.view.setFocus()

    def open_comment_at_index(self, index: int) -> None:
        session = self.current_session()
        if session is None:
            return
        source = self._comment_source_for_display_index(session, index)
        if source is None:
            return
        source_type, source_index = source
        if source_type == "saved":
            self._show_saved_pdf_comment(
                session.saved_comment_edits.get(source_index, session.saved_pdf_comments[source_index])
            )
            return
        comment = session.comments[source_index]
        dialog = CommentDialog(
            self.equation_macros,
            self,
            initial_text=comment.text,
            initial_color=comment.color,
        )
        if dialog.exec() != QDialog.Accepted:
            self.view.setFocus()
            return
        if not dialog.has_comment_text():
            del session.comments[source_index]
            self.statusBar().showMessage("Deleted empty comment", 4000)
        else:
            text = dialog.comment_text()
            session.comments[source_index] = PdfComment(
                page=comment.page,
                rect=comment.rect,
                text=text,
                color=dialog.comment_color(),
            )
            self.statusBar().showMessage(f"Updated comment on page {comment.page}", 5000)
        self._refresh_annotation_overlays(session)
        self._save_session_state_now()
        self.view.setFocus()

    def show_comment_context_menu(self, index: int, global_pos: QPoint) -> None:
        session = self.current_session()
        if session is None or self._comment_source_for_display_index(session, index) is None:
            return

        menu = QMenu(self)
        edit_action = menu.addAction("Edit Comment")
        remove_action = menu.addAction("Remove Comment")
        chosen = menu.exec(global_pos)
        if chosen == edit_action:
            self.edit_comment_at_index(index)
        elif chosen == remove_action:
            self.remove_comment_at_index(index)
        self.view.setFocus()

    def edit_comment_at_index(self, index: int) -> None:
        session = self.current_session()
        if session is None:
            return
        source = self._comment_source_for_display_index(session, index)
        if source is None:
            return
        source_type, source_index = source
        if source_type == "pending":
            self.open_comment_at_index(index)
            return
        self._stage_saved_pdf_comment_edit(session, source_index)

    def remove_comment_at_index(self, index: int) -> None:
        session = self.current_session()
        if session is None:
            return
        source = self._comment_source_for_display_index(session, index)
        if source is None:
            return
        source_type, source_index = source
        if source_type == "pending":
            comment = session.comments[source_index]
            del session.comments[source_index]
            self._refresh_annotation_overlays(session)
            self._save_session_state_now()
            self.statusBar().showMessage(f"Removed comment on page {comment.page}", 5000)
            return
        self._stage_saved_pdf_comment_removal(session, source_index)

    @staticmethod
    def _rects_close(a: fitz.Rect, b: fitz.Rect, tolerance: float = 3.0) -> bool:
        return (
            abs(a.x0 - b.x0) <= tolerance
            and abs(a.y0 - b.y0) <= tolerance
            and abs(a.x1 - b.x1) <= tolerance
            and abs(a.y1 - b.y1) <= tolerance
        )

    @staticmethod
    def _annotation_content(annot) -> str:
        try:
            return str((annot.info or {}).get("content", "")).strip()
        except Exception:
            return ""

    @staticmethod
    def _comment_popup_rect(page: fitz.Page, rect: fitz.Rect) -> fitz.Rect:
        page_rect = page.rect
        width = min(260.0, max(160.0, page_rect.width * 0.4))
        height = min(160.0, max(90.0, page_rect.height * 0.18))
        x0 = rect.x1 + 12.0
        if x0 + width > page_rect.x1:
            x0 = max(page_rect.x0, rect.x0 - width - 12.0)
        y0 = rect.y0
        if y0 + height > page_rect.y1:
            y0 = max(page_rect.y0, page_rect.y1 - height)
        popup = fitz.Rect(x0, y0, x0 + width, y0 + height) & page_rect
        return popup if not popup.is_empty else fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x0 + width, page_rect.y0 + height) & page_rect

    @staticmethod
    def _set_pdf_reader_comment_annotation(
        page: fitz.Page,
        annot,
        rect: fitz.Rect,
        text: str,
        color: Tuple[float, float, float],
    ) -> None:
        annot.set_colors(stroke=color)
        annot.set_border(width=1.5, dashes=[4])
        annot.set_info(content=text, subject="PDF Reader Comment", title="PDF Reader")
        annot.set_popup(PdfReaderWindow._comment_popup_rect(page, rect))
        annot.set_open(False)
        annot.update()

    def _matching_saved_comment_annotations(self, session: DocumentSession, comment: PdfComment) -> Tuple[Optional[fitz.Page], List[object], List[object]]:
        if not (1 <= comment.page <= len(session.doc)):
            return None, [], []
        page = session.doc.load_page(comment.page - 1)
        target = fitz.Rect(comment.rect) & page.rect
        squares: List[object] = []
        notes: List[object] = []
        annotations = page.annots()
        if annotations is None:
            return page, squares, notes

        for annot in annotations:
            try:
                info = annot.info or {}
                if str(info.get("subject", "")) != "PDF Reader Comment":
                    continue
                annot_type = annot.type[0]
                if annot_type == fitz.PDF_ANNOT_SQUARE and self._rects_close(annot.rect & page.rect, target):
                    squares.append(annot)
                elif annot_type == fitz.PDF_ANNOT_TEXT and self._annotation_content(annot) == comment.text.strip():
                    icon_rect = annot.rect
                    near_left = abs(icon_rect.x0 - target.x0) <= 24.0
                    near_top = abs(icon_rect.y1 - target.y0) <= 32.0 or abs(icon_rect.y0 - target.y0) <= 32.0
                    if near_left and near_top:
                        notes.append(annot)
            except Exception:
                continue
        return page, squares, notes

    def _refresh_after_pdf_annotation_edit(self, session: DocumentSession) -> None:
        session.saved_pdf_comments = self._load_pdf_reader_comments_from_pdf(session.doc)
        session.saved_comment_edits.clear()
        session.saved_comment_removals.clear()
        session.rendered_pages = []
        session.rendered_page_numbers.clear()
        session.page_display_sizes = []
        self._refresh_annotation_overlays(session)
        self._refresh_bookmarks_panel()
        self._ensure_rendered_pages(session)
        for view in self.reader_views:
            if not self._view_has_session(session, view):
                continue
            state = self._view_state_for_view(view)
            self._show_session_in_view(view, session)
            if state is not None:
                self._apply_view_state_to_view(view, state)
        self._save_session_state_now()

    def _stage_saved_pdf_comment_edit(self, session: DocumentSession, saved_index: int) -> None:
        comment = session.saved_comment_edits.get(saved_index, session.saved_pdf_comments[saved_index])
        dialog = CommentDialog(
            self.equation_macros,
            self,
            initial_text=comment.text,
            initial_color=comment.color,
        )
        if dialog.exec() != QDialog.Accepted:
            self.view.setFocus()
            return
        if not dialog.has_comment_text():
            self._stage_saved_pdf_comment_removal(session, saved_index)
            return

        text = dialog.comment_text()
        session.saved_comment_edits[saved_index] = PdfComment(
            page=comment.page,
            rect=comment.rect,
            text=text,
            color=dialog.comment_color(),
        )
        self._refresh_annotation_overlays(session)
        self._save_session_state_now()
        self.statusBar().showMessage(f"Staged comment update on page {comment.page}; press Save Changes to write it", 7000)
        self.view.setFocus()

    def _stage_saved_pdf_comment_removal(self, session: DocumentSession, saved_index: int) -> None:
        comment = session.saved_comment_edits.get(saved_index, session.saved_pdf_comments[saved_index])
        result = QMessageBox.question(
            self,
            "Remove Comment",
            f"Remove the saved comment on page {comment.page}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if result != QMessageBox.Yes:
            self.view.setFocus()
            return

        session.saved_comment_removals.add(saved_index)
        session.saved_comment_edits.pop(saved_index, None)
        self._refresh_annotation_overlays(session)
        self._save_session_state_now()
        self.statusBar().showMessage(f"Staged comment removal on page {comment.page}; press Save Changes to write it", 7000)
        self.view.setFocus()

    def _show_saved_pdf_comment(self, comment: PdfComment) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Saved comment on page {comment.page}")
        dialog.resize(520, 320)

        layout = QVBoxLayout(dialog)
        viewer = QTextEdit(dialog)
        viewer.setReadOnly(True)
        if "<html" in comment.text.lower() or "<span" in comment.text.lower() or "<p" in comment.text.lower():
            viewer.setHtml(comment.text)
        else:
            viewer.setPlainText(comment.text)
        layout.addWidget(viewer, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.exec()
        self.view.setFocus()

    def _clear_menu_documents(self) -> None:
        while self.menu_list_layout.count():
            item = self.menu_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _document_thumbnail(self, path: str) -> QPixmap:
        try:
            key = str(Path(path).resolve())
        except Exception:
            key = path
        if key in self._thumbnail_cache:
            return self._thumbnail_cache[key]

        pixmap = QPixmap(96, 128)
        pixmap.fill(QColor("#f4f4f4"))
        try:
            doc = fitz.open(path)
            try:
                page = doc.load_page(0)
                page_pix = page.get_pixmap(matrix=fitz.Matrix(0.22, 0.22), alpha=False)
                image = QImage(page_pix.samples, page_pix.width, page_pix.height, page_pix.stride, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(image).scaled(96, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            finally:
                doc.close()
        except Exception:
            placeholder = QPixmap(96, 128)
            placeholder.fill(QColor("#eeeeee"))
            painter = QPainter(placeholder)
            painter.setPen(QPen(QColor("#777777")))
            painter.drawRect(0, 0, 95, 127)
            painter.drawText(placeholder.rect(), Qt.AlignCenter, "PDF")
            painter.end()
            pixmap = placeholder

        self._thumbnail_cache[key] = pixmap
        return pixmap

    def _refresh_menu_documents(self) -> None:
        if not hasattr(self, "menu_list_layout"):
            return

        query = self.menu_search_input.text().strip().lower() if hasattr(self, "menu_search_input") else ""
        self._clear_menu_documents()

        local_shown = 0
        local_title = QLabel("Local Documents")
        local_title.setStyleSheet("font-size: 16px; font-weight: 600; margin-top: 6px;")
        self.menu_list_layout.addWidget(local_title)
        for path in self.recent_documents:
            name = self._session_display_name(path)
            haystack = f"{name}\n{path}".lower()
            if query and query not in haystack:
                continue

            button = QPushButton(f"{name}\n{path}")
            button.setIcon(QIcon(self._document_thumbnail(path)))
            button.setIconSize(QSize(96, 128))
            button.setMinimumHeight(146)
            button.setStyleSheet("QPushButton { text-align: left; padding: 10px; }")
            button.clicked.connect(lambda _checked=False, p=path: self._open_recent_document(p))
            button.setContextMenuPolicy(Qt.CustomContextMenu)
            button.customContextMenuRequested.connect(
                lambda pos, b=button, p=path: self._show_local_document_context_menu(b, pos, p)
            )
            self.menu_list_layout.addWidget(button)
            local_shown += 1

        if local_shown == 0:
            empty = QLabel("No local recent documents")
            empty.setAlignment(Qt.AlignCenter)
            empty.setMinimumHeight(140)
            self.menu_list_layout.addWidget(empty)

        cloud_title = QLabel("Cloud Documents")
        cloud_title.setStyleSheet("font-size: 16px; font-weight: 600; margin-top: 12px;")
        self.menu_list_layout.addWidget(cloud_title)

        cloud_shown = 0
        if self.google_drive.is_logged_in:
            for document in self.cloud_documents:
                name = document.name
                detail = "Google Drive"
                if document.modified_time:
                    detail = f"Google Drive - modified {document.modified_time[:10]}"
                haystack = f"{name}\n{detail}".lower()
                if query and query not in haystack:
                    continue

                button = QPushButton(f"{name}\n{detail}")
                button.setIcon(QIcon(self._cloud_document_thumbnail(document)))
                button.setIconSize(QSize(96, 128))
                button.setMinimumHeight(146)
                button.setStyleSheet("QPushButton { text-align: left; padding: 10px; }")
                button.clicked.connect(lambda _checked=False, doc=document: self._open_cloud_document(doc))
                self.menu_list_layout.addWidget(button)
                cloud_shown += 1

        if cloud_shown == 0:
            if self.google_drive.is_logged_in:
                message = (
                    f"No cloud documents in the {DRIVE_FOLDER_NAME} Drive folder"
                    if not query
                    else f"No matching cloud documents in {DRIVE_FOLDER_NAME}"
                )
            else:
                message = "Sign in to Google Drive to show cloud documents"
            empty = QLabel(message)
            empty.setAlignment(Qt.AlignCenter)
            empty.setMinimumHeight(140)
            self.menu_list_layout.addWidget(empty)

        self.menu_list_layout.addStretch()

    def _open_recent_document(self, path: str) -> None:
        entry = self.recent_document_entries.get(self._path_key(path))
        resolved_path = self._resolve_document_path(path, entry)
        if not resolved_path:
            QMessageBox.warning(self, "Missing file", f"Could not find PDF:\n{path}")
            return
        if resolved_path != path:
            self._replace_recent_document_path(path, resolved_path, entry)
            if entry:
                entry = dict(entry)
                entry["path"] = resolved_path
        if self._find_session_by_path(resolved_path) is None:
            if entry:
                self._load_pdf_from_persistence_entry(entry, switch_to=True, update_recent=True)
            else:
                self.load_pdf(resolved_path, switch_to=True)
        else:
            self.load_pdf(resolved_path, switch_to=True)
        self.show_reader()

    def _create_actions_and_toolbar(self) -> None:
        self.menu_action = self._create_action("Menu", "Meta+M", self.show_menu, "menu")
        self.open_many_action = self._create_action("Open PDFs", SHORTCUT_OPEN, self.open_multiple_pdfs, "open_pdfs")
        self.toggle_last_documents_action = QAction("Toggle Last Documents", self)
        toggle_last_shortcut = self._shortcut_for("toggle_last_documents", SHORTCUT_TOGGLE_LAST_DOCUMENTS[0])
        if "toggle_last_documents" in self.shortcut_overrides:
            self.toggle_last_documents_action.setShortcut(QKeySequence(toggle_last_shortcut) if toggle_last_shortcut else QKeySequence())
        else:
            self.toggle_last_documents_action.setShortcuts(
                [QKeySequence(shortcut) for shortcut in SHORTCUT_TOGGLE_LAST_DOCUMENTS]
            )
        self.toggle_last_documents_action.triggered.connect(self.toggle_last_documents)
        self.addAction(self.toggle_last_documents_action)
        self._shortcut_actions["toggle_last_documents"] = self.toggle_last_documents_action
        if "toggle_last_documents" in self.shortcut_overrides:
            self._register_layout_independent_shortcut(self.toggle_last_documents_action, toggle_last_shortcut)
        else:
            for shortcut in SHORTCUT_TOGGLE_LAST_DOCUMENTS:
                self._register_layout_independent_shortcut(self.toggle_last_documents_action, shortcut)
        self.close_doc_action = self._create_action("Close Document", SHORTCUT_CLOSE_DOCUMENT, self.close_current_document, "close_document")
        self.back_action = self._create_action("Back", SHORTCUT_BACK, self.go_back, "back")
        self.forward_action = self._create_action("Forward", SHORTCUT_FORWARD, self.go_forward, "forward")
        self.prev_action = self._create_action("Previous Page", SHORTCUT_PREVIOUS_PAGE, self.prev_page, "previous_page")
        self.next_action = self._create_action("Next Page", SHORTCUT_NEXT_PAGE, self.next_page, "next_page")
        self.zoom_in_action = self._create_action(
            "Zoom In",
            SHORTCUT_ZOOM_IN,
            lambda: self.view.set_zoom_factor(self.view.zoom_factor() * 1.2),
            "zoom_in",
        )
        self.zoom_out_action = self._create_action(
            "Zoom Out",
            SHORTCUT_ZOOM_OUT,
            lambda: self.view.set_zoom_factor(self.view.zoom_factor() / 1.2),
            "zoom_out",
        )
        self.jump_action = self._create_action("Jump to Equation", SHORTCUT_JUMP_TO_EQUATION, self.jump_to_equation, "jump_to_equation")
        self.toggle_bookmarks_action = self._create_action("Toggle Bookmarks", SHORTCUT_TOGGLE_BOOKMARKS, self.toggle_bookmarks_panel, "toggle_bookmarks")
        self.add_bookmark_action = self._create_action("Add Bookmark", SHORTCUT_ADD_BOOKMARK, self.add_bookmark, "add_bookmark")
        self.return_to_equation_source_action = self._create_action(
            "Return to Equation Source",
            SHORTCUT_RETURN_TO_EQUATION_SOURCE,
            self.return_to_equation_source_page,
            "return_to_equation_source",
        )
        self.reload_a_txt_action = QAction("Reload a.txt", self)
        self.reload_a_txt_action.triggered.connect(self._reload_current_session_chapter_starts)
        self.copy_between_lines_action = self._create_action(
            "Copy Between Blue Lines",
            SHORTCUT_COPY_BETWEEN_LINES,
            self.copy_between_selection_lines,
            "copy_between_lines",
        )
        self.copy_between_lines_without_equations_action = self._create_action(
            "Copy Between Blue Lines Without Equations",
            SHORTCUT_COPY_BETWEEN_LINES_WITHOUT_EQUATIONS,
            self.copy_between_selection_lines_without_equations,
            "copy_between_lines_without_equations",
        )
        self.clear_selection_lines_action = self._create_action(
            "Clear Blue Lines",
            SHORTCUT_CLEAR_SELECTION_LINES,
            self.clear_selection_lines,
            "clear_selection_lines",
        )
        self.preview_equation_action = self._create_action(
            "Preview Equation Under Cursor",
            SHORTCUT_PREVIEW_EQUATION,
            self.preview_equation_under_cursor,
            "preview_equation",
        )
        self.open_text_search_action = self._create_action(
            "Search Text",
            SHORTCUT_OPEN_TEXT_SEARCH,
            self.open_text_search,
            "open_text_search",
        )
        self.open_ai_chat_action = self._create_action(
            "AI Chat",
            SHORTCUT_OPEN_AI_CHAT,
            self.open_ai_chat,
            "open_ai_chat",
        )
        self.split_viewer_action = self._create_action(
            "Split Viewer",
            SHORTCUT_SPLIT_VIEWER,
            self.split_active_viewer,
            "split_viewer",
        )
        self.close_viewer_action = self._create_action(
            "Close Viewer",
            SHORTCUT_CLOSE_VIEWER,
            self.close_active_viewer,
            "close_viewer",
        )
        self.rotate_viewer_segments_action = self._create_action(
            "Rotate Viewer Segments",
            SHORTCUT_ROTATE_VIEWER_SEGMENTS,
            self.rotate_viewer_segments,
            "rotate_viewer_segments",
        )
        self.return_to_text_search_source_action = self._create_action(
            "Return to Text Search Source",
            SHORTCUT_RETURN_TO_TEXT_SEARCH_SOURCE,
            self.return_to_text_search_source,
            "return_to_text_search_source",
        )
        self.scan_equation_index_action = self._create_action(
            "Scan Equation Index",
            SHORTCUT_SCAN_EQUATION_INDEX,
            self.choose_equation_index_scan_mode,
            "scan_equation_index",
        )
        self.learn_equation_rectangle_action = self._create_action(
            "Learn Equation Rectangle",
            SHORTCUT_LEARN_EQUATION_RECTANGLE,
            self.learn_equation_format_from_selected_rectangle,
            "learn_equation_rectangle",
        )
        self.create_comment_action = self._create_action(
            "Create Comment",
            SHORTCUT_CREATE_COMMENT,
            self.create_comment_from_selected_rectangle,
            "create_comment",
        )
        self.toggle_equation_lookup_action = self._create_action(
            "Toggle Equation Lookup",
            SHORTCUT_TOGGLE_EQUATION_LOOKUP,
            self.toggle_equation_lookup_mode,
            "toggle_equation_lookup",
        )
        self.save_changes_action = self._create_action(
            "Save Changes",
            SHORTCUT_SAVE_CHANGES,
            self.save_current_document_changes,
            "save_changes",
        )

        toolbar = QToolBar("Main")
        toolbar.addAction(self.menu_action)
        toolbar.addAction(self.open_many_action)
        toolbar.addAction(self.save_changes_action)
        toolbar.addSeparator()
        toolbar.addAction(self.back_action)
        toolbar.addAction(self.forward_action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Page: "))
        toolbar.addWidget(self.page_input)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Eq: "))
        toolbar.addWidget(self.eq_input)
        jump_button = QPushButton("Jump")
        jump_button.clicked.connect(self.jump_to_equation)
        toolbar.addWidget(jump_button)
        toolbar.addSeparator()
        toolbar.addAction(self.zoom_in_action)
        toolbar.addAction(self.zoom_out_action)
        toolbar.addSeparator()
        toolbar.addAction(self.toggle_bookmarks_action)
        toolbar.addAction(self.add_bookmark_action)
        toolbar.addAction(self.open_text_search_action)
        toolbar.addAction(self.open_ai_chat_action)
        toolbar.addAction(self.scan_equation_index_action)
        toolbar.addAction(self.learn_equation_rectangle_action)
        toolbar.addAction(self.create_comment_action)
        toolbar.addAction(self.toggle_equation_lookup_action)
        self.addToolBar(toolbar)

    def current_session(self) -> Optional[DocumentSession]:
        if 0 <= self.current_index < len(self.sessions):
            return self.sessions[self.current_index]
        return None

    def _remember_document_activation(self, session: DocumentSession) -> None:
        key = self._path_key(session.path)
        self.document_activation_history = [
            existing
            for existing in self.document_activation_history
            if existing != key and self._find_session_by_path(existing) is not None
        ]
        self.document_activation_history.append(key)
        self.document_activation_history = self.document_activation_history[-12:]

    def toggle_last_documents(self) -> None:
        current = self.current_session()
        current_key = self._path_key(current.path) if current is not None else None
        available = [
            key
            for key in reversed(self.document_activation_history)
            if key != current_key and self._find_session_by_path(key) is not None
        ]
        if not available:
            self.statusBar().showMessage("No previous open document", 3000)
            return

        target_index = self._find_session_by_path(available[0])
        if target_index is None:
            self.statusBar().showMessage("No previous open document", 3000)
            return

        self.show_reader()
        self.doc_tabs.setCurrentIndex(target_index)
        if self.current_index != target_index:
            self._switch_to_document(target_index)

    def switch_to_adjacent_document(self, direction: int) -> None:
        if len(self.sessions) < 2:
            return

        current = self.doc_tabs.currentIndex()
        if current < 0:
            current = self.current_index if self.current_index >= 0 else 0

        target_index = (current + direction) % len(self.sessions)
        self.show_reader()
        self.doc_tabs.setCurrentIndex(target_index)
        if self.current_index != target_index:
            self._switch_to_document(target_index)

    def _session_display_name(self, path: str) -> str:
        cloud_name = self._cloud_cache_display_name(path)
        if cloud_name:
            return cloud_name
        return Path(path).name

    @staticmethod
    def _cloud_cache_display_name(path: str) -> str:
        file_path = Path(path)
        meta_path = file_path.with_suffix(file_path.suffix + ".json")
        if not meta_path.exists():
            return ""
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        if not isinstance(data, dict):
            return ""
        name = str(data.get("name") or "").strip()
        return name

    @staticmethod
    def _cloud_cache_metadata(path: str) -> dict:
        file_path = Path(path)
        meta_path = file_path.with_suffix(file_path.suffix + ".json")
        if not meta_path.exists():
            return {}
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _cloud_document_id_for_path(self, path: str) -> str:
        return str(self._cloud_cache_metadata(path).get("id") or "").strip()

    @staticmethod
    def _annotation_cache_path_for_pdf(path: str) -> Path:
        file_path = Path(path)
        return file_path.with_suffix(file_path.suffix + ".annotations.json")

    @staticmethod
    def _qcolor_to_pdf_rgb(color_text: str, fallback: str = "#FFFFFF") -> Tuple[float, float, float]:
        color = QColor(color_text)
        if not color.isValid():
            color = QColor(fallback)
        return color.redF(), color.greenF(), color.blueF()

    @staticmethod
    def _annotation_stroke_rgb(annot, fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
        stroke = (annot.colors or {}).get("stroke")
        if stroke and len(stroke) >= 3:
            return (
                max(0.0, min(1.0, float(stroke[0]))),
                max(0.0, min(1.0, float(stroke[1]))),
                max(0.0, min(1.0, float(stroke[2]))),
            )
        return fallback

    @staticmethod
    def _annotation_stroke_color_text(annot, fallback: str = "#2EAD4A") -> str:
        return QColor.fromRgbF(*PdfReaderWindow._annotation_stroke_rgb(annot, PdfReaderWindow._qcolor_to_pdf_rgb(fallback))).name()

    def _embed_pending_annotations_in_pdf(self, session: DocumentSession) -> Tuple[int, int]:
        cover_count = 0
        comment_count = 0

        covers_by_page: Dict[int, List[PdfCover]] = {}
        for cover in session.covers:
            covers_by_page.setdefault(cover.page, []).append(cover)

        for page_number, covers in covers_by_page.items():
            if not (1 <= page_number <= len(session.doc)):
                continue
            page = session.doc.load_page(page_number - 1)
            added = 0
            for cover in covers:
                rect = fitz.Rect(cover.rect) & page.rect
                if rect.is_empty:
                    continue
                fill = self._qcolor_to_pdf_rgb(cover.color, "#FFFFFF")
                try:
                    page.add_redact_annot(rect, fill=fill)
                    added += 1
                except Exception:
                    continue
            if added:
                page.apply_redactions()
                cover_count += added

        for comment in session.comments:
            if not (1 <= comment.page <= len(session.doc)):
                continue
            page = session.doc.load_page(comment.page - 1)
            rect = fitz.Rect(comment.rect) & page.rect
            if rect.is_empty:
                continue
            color = self._qcolor_to_pdf_rgb(comment.color, "#2EAD4A")
            plain_text = self._plain_comment_text(comment.text) or "Comment"
            try:
                rect_annot = page.add_rect_annot(rect)
                self._set_pdf_reader_comment_annotation(page, rect_annot, rect, plain_text, color)
                comment_count += 1
            except Exception:
                continue

        return cover_count, comment_count

    def _embed_pending_bookmarks_in_pdf(self, session: DocumentSession) -> int:
        if not session.user_bookmarks:
            return 0

        try:
            toc = session.doc.get_toc(simple=True)
        except Exception:
            toc = []

        existing = set()
        for entry in toc:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            try:
                existing.add((str(entry[1]).strip(), int(entry[2])))
            except Exception:
                continue

        added = 0
        for bookmark in session.user_bookmarks:
            page = max(1, min(int(bookmark.page), len(session.doc)))
            title = str(bookmark.title).strip() or f"Page {page}"
            key = (title, page)
            if key in existing:
                continue
            toc.append([1, title, page])
            existing.add(key)
            added += 1

        if added:
            session.doc.set_toc(toc)
            session.section_starts = extract_section_starts_from_toc(session.doc)
        return added

    def _save_pdf_full_replace(self, session: DocumentSession) -> None:
        source_path = Path(session.path)
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{source_path.stem}.save-",
                suffix=source_path.suffix,
                dir=str(source_path.parent),
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)

            session.doc.save(str(tmp_path), garbage=4, deflate=True)
            session.doc.close()
            tmp_path.replace(source_path)
            session.doc = fitz.open(session.path)
        except Exception:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                if session.doc.is_closed:
                    session.doc = fitz.open(session.path)
            except Exception:
                pass
            raise

    def _apply_pending_saved_comment_changes(self, session: DocumentSession) -> Tuple[int, int]:
        removed_count = 0
        edited_count = 0

        for saved_index in sorted(session.saved_comment_removals, reverse=True):
            if not (0 <= saved_index < len(session.saved_pdf_comments)):
                continue
            original = session.saved_pdf_comments[saved_index]
            page, squares, notes = self._matching_saved_comment_annotations(session, original)
            if page is None:
                continue
            for annot in notes + squares:
                page.delete_annot(annot)
            if squares:
                removed_count += 1

        for saved_index, edited in sorted(session.saved_comment_edits.items()):
            if saved_index in session.saved_comment_removals:
                continue
            if not (0 <= saved_index < len(session.saved_pdf_comments)):
                continue
            original = session.saved_pdf_comments[saved_index]
            page, squares, notes = self._matching_saved_comment_annotations(session, original)
            if page is None or not squares:
                continue
            color = self._qcolor_to_pdf_rgb(edited.color, "#2EAD4A")
            plain_text = self._plain_comment_text(edited.text) or "Comment"
            for annot in squares:
                self._set_pdf_reader_comment_annotation(page, annot, fitz.Rect(original.rect) & page.rect, plain_text, color)
            for note in notes:
                page.delete_annot(note)
            edited_count += 1

        return edited_count, removed_count

    def _remove_pdf_reader_comment_note_icons(self, session: DocumentSession) -> int:
        removed = 0
        for page_index in range(len(session.doc)):
            try:
                page = session.doc.load_page(page_index)
                annotations = list(page.annots() or [])
            except Exception:
                continue
            for annot in annotations:
                try:
                    info = annot.info or {}
                    if str(info.get("subject", "")) != "PDF Reader Comment":
                        continue
                    if annot.type[0] != fitz.PDF_ANNOT_TEXT:
                        continue
                    page.delete_annot(annot)
                    removed += 1
                except Exception:
                    continue
        return removed

    def _repair_pdf_reader_comment_popups(self, session: DocumentSession) -> int:
        repaired = 0
        for page_index in range(len(session.doc)):
            try:
                page = session.doc.load_page(page_index)
                annotations = list(page.annots() or [])
            except Exception:
                continue
            for annot in annotations:
                try:
                    info = annot.info or {}
                    if str(info.get("subject", "")) != "PDF Reader Comment":
                        continue
                    if annot.type[0] != fitz.PDF_ANNOT_SQUARE:
                        continue
                    if annot.has_popup:
                        continue
                    text = str(info.get("content", "")).strip()
                    if not text:
                        continue
                    rect = annot.rect & page.rect
                    if rect.is_empty:
                        continue
                    color = PdfReaderWindow._annotation_stroke_rgb(annot, (0.18, 0.68, 0.29))
                    PdfReaderWindow._set_pdf_reader_comment_annotation(page, annot, rect, text, color)
                    repaired += 1
                except Exception:
                    continue
        return repaired

    @staticmethod
    def _load_pdf_reader_comments_from_pdf(doc: fitz.Document) -> List[PdfComment]:
        comments: List[PdfComment] = []
        for page_index in range(len(doc)):
            try:
                page = doc.load_page(page_index)
                annotations = page.annots()
            except Exception:
                continue
            if annotations is None:
                continue
            for annot in annotations:
                try:
                    info = annot.info or {}
                    if str(info.get("subject", "")) != "PDF Reader Comment":
                        continue
                    if annot.type[0] != fitz.PDF_ANNOT_SQUARE:
                        continue
                    text = str(info.get("content", "")).strip()
                    if not text:
                        continue
                    rect = annot.rect & page.rect
                    if rect.is_empty:
                        continue
                    color = PdfReaderWindow._annotation_stroke_color_text(annot, "#2EAD4A")
                    comments.append(
                        PdfComment(
                            page=page_index + 1,
                            rect=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                            text=text,
                            color=color,
                        )
                    )
                except Exception:
                    continue
        return comments

    @staticmethod
    def _pdf_incremental_save_error() -> str:
        return (
            "This PDF cannot accept an appended incremental update, likely because it was repaired when opened. "
            "No comments were written to the file."
        )

    @staticmethod
    def _ensure_pdf_can_append(session: DocumentSession) -> None:
        if not session.doc.can_save_incrementally():
            raise RuntimeError(PdfReaderWindow._pdf_incremental_save_error())

    @staticmethod
    def _save_pdf_incrementally(session: DocumentSession) -> None:
        PdfReaderWindow._ensure_pdf_can_append(session)
        session.doc.saveIncr()

    def _refresh_after_pdf_file_save(self, session: DocumentSession) -> None:
        session.comments.clear()
        session.covers.clear()
        session.user_bookmarks.clear()
        session.section_starts = extract_section_starts_from_toc(session.doc)
        session.saved_pdf_comments = self._load_pdf_reader_comments_from_pdf(session.doc)
        session.saved_comment_edits.clear()
        session.saved_comment_removals.clear()
        session.rendered_pages = []
        session.rendered_page_numbers.clear()
        session.page_display_sizes = []
        self._refresh_annotation_overlays(session)
        self._ensure_rendered_pages(session)
        for view in self.reader_views:
            if not self._view_has_session(session, view):
                continue
            state = self._view_state_for_view(view)
            self._show_session_in_view(view, session)
            if state is not None:
                self._apply_view_state_to_view(view, state)

    def _remove_annotation_cache_for_session(self, session: DocumentSession) -> None:
        try:
            self._annotation_cache_path_for_pdf(session.path).unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _upload_saved_pdf_to_cloud(self, session: DocumentSession) -> str:
        document_id = self._cloud_document_id_for_path(session.path)
        if not document_id:
            return ""
        if not self.google_drive.is_logged_in:
            raise GoogleDriveError("Sign in to Google Drive to upload the saved PDF.")

        metadata = self._cloud_cache_metadata(session.path)
        mime_type = str(metadata.get("mime_type") or "")
        if mime_type and mime_type != "application/pdf":
            uploaded = self.google_drive.upload_pdf(Path(session.path))
            self.cloud_documents = [uploaded] + [
                document for document in self.cloud_documents if document.id != uploaded.id
            ]
            return f"uploaded a PDF copy to Google Drive ({uploaded.name})"

        updated = self.google_drive.update_pdf_media(document_id, Path(session.path))
        self.cloud_documents = [updated] + [
            document for document in self.cloud_documents if document.id != updated.id
        ]
        meta_path = Path(session.path).with_suffix(Path(session.path).suffix + ".json")
        metadata.update(
            {
                "id": updated.id,
                "name": updated.name,
                "mime_type": updated.mime_type,
                "modified_time": updated.modified_time,
                "size": updated.size,
            }
        )
        try:
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception:
            pass
        return "uploaded the updated PDF to Google Drive"

    def save_current_document_changes(self) -> None:
        session = self.current_session()
        if session is None:
            self.statusBar().showMessage("No document is open", 3000)
            return

        pending_covers = len(session.covers)
        pending_comments = len(session.comments)
        pending_bookmarks = len(session.user_bookmarks)
        pending_saved_comment_changes = len(session.saved_comment_edits) + len(session.saved_comment_removals)
        if pending_covers == 0 and pending_comments == 0 and pending_bookmarks == 0 and pending_saved_comment_changes == 0:
            self._save_active_session_view()
            if not session.doc.can_save_incrementally():
                self._save_session_state_now()
                self.statusBar().showMessage("No pending comments, rectangles, or bookmarks to write into the PDF", 4000)
                return
            self.statusBar().showMessage("Checking saved PDF comments...", 0)
            QApplication.processEvents()
            try:
                removed_note_icons = self._remove_pdf_reader_comment_note_icons(session)
                repaired_comment_popups = self._repair_pdf_reader_comment_popups(session)
                if removed_note_icons or repaired_comment_popups:
                    self._save_pdf_incrementally(session)
            except Exception as exc:
                QMessageBox.warning(self, "Save PDF", f"Could not update saved PDF comments:\n{exc}")
                return

            if removed_note_icons or repaired_comment_popups:
                self._refresh_after_pdf_annotation_edit(session)
                self.recent_document_entries[self._path_key(session.path)] = self._session_persistence_entry(session)
                self._save_session_state_now()
                change_parts = []
                if repaired_comment_popups:
                    change_parts.append(f"made {repaired_comment_popups} comment(s) Acrobat-readable")
                if removed_note_icons:
                    change_parts.append(f"removed {removed_note_icons} old comment icon(s)")
                change_message = " and ".join(change_parts)
                document_id = self._cloud_document_id_for_path(session.path)
                if document_id:
                    try:
                        self.statusBar().showMessage("Uploading the updated PDF to Google Drive...", 0)
                        QApplication.processEvents()
                        cloud_message = self._upload_saved_pdf_to_cloud(session)
                        self.statusBar().showMessage(
                            f"{change_message} and {cloud_message}",
                            7000,
                        )
                    except GoogleDriveError as exc:
                        self.statusBar().showMessage(
                            f"{change_message} locally; Google Drive upload failed: {exc}",
                            9000,
                        )
                else:
                    self.statusBar().showMessage(
                        change_message,
                        6000,
                    )
                return

            self._save_session_state_now()
            self.statusBar().showMessage("No pending comments, rectangles, or bookmarks to write into the PDF", 4000)
            return

        self._save_active_session_view()
        self.statusBar().showMessage("Writing comments, rectangles, and bookmarks into the PDF...", 0)
        QApplication.processEvents()

        try:
            full_save_required = bool(pending_covers or pending_bookmarks)
            if not full_save_required and not self._repair_pdf_in_place_for_append(session):
                return
            removed_note_icons = self._remove_pdf_reader_comment_note_icons(session)
            repaired_comment_popups = self._repair_pdf_reader_comment_popups(session)
            edited_saved_comments, removed_saved_comments = self._apply_pending_saved_comment_changes(session)
            cover_count, comment_count = self._embed_pending_annotations_in_pdf(session)
            bookmark_count = self._embed_pending_bookmarks_in_pdf(session)
            if full_save_required:
                self._save_pdf_full_replace(session)
            else:
                self._save_pdf_incrementally(session)
        except Exception as exc:
            QMessageBox.warning(self, "Save PDF", f"Could not save changes into the PDF:\n{exc}")
            return

        self._remove_annotation_cache_for_session(session)
        self._refresh_after_pdf_file_save(session)
        self.recent_document_entries[self._path_key(session.path)] = self._session_persistence_entry(session)
        self._save_session_state_now()

        document_id = self._cloud_document_id_for_path(session.path)
        cleanup_parts = []
        if repaired_comment_popups:
            cleanup_parts.append(f"made {repaired_comment_popups} existing comment(s) Acrobat-readable")
        if removed_note_icons:
            cleanup_parts.append(f"removed {removed_note_icons} old comment icon(s)")
        if edited_saved_comments:
            cleanup_parts.append(f"updated {edited_saved_comments} saved comment(s)")
        if removed_saved_comments:
            cleanup_parts.append(f"removed {removed_saved_comments} saved comment(s)")
        cleanup_message = f" and {' and '.join(cleanup_parts)}" if cleanup_parts else ""
        main_parts = [
            f"{comment_count} comment(s)",
            f"{cover_count} rectangle(s)",
            f"{bookmark_count} bookmark(s)",
        ]
        main_message = f"Wrote {', '.join(main_parts)} into the PDF"
        if not document_id:
            self.statusBar().showMessage(
                f"{main_message}{cleanup_message}",
                6000,
            )
            return

        try:
            self.statusBar().showMessage("Uploading the updated PDF to Google Drive...", 0)
            QApplication.processEvents()
            cloud_message = self._upload_saved_pdf_to_cloud(session)
        except GoogleDriveError as exc:
            self.statusBar().showMessage(
                f"Saved into the local PDF; Google Drive upload failed: {exc}",
                9000,
            )
            return

        self.statusBar().showMessage(
            f"{main_message}{cleanup_message} and {cloud_message}",
            7000,
        )

    def _set_no_document_ui(self) -> None:
        self.current_index = -1
        self.page_input.blockSignals(True)
        self.page_input.setMinimum(1)
        self.page_input.setMaximum(1)
        self.page_input.setValue(1)
        self.page_input.blockSignals(False)
        self._clear_reader_views()
        self.view = self._create_pdf_view()
        self._add_reader_view(self.view)
        self.status_label.setText("No PDF loaded")
        self.zoom_label.setText("100%")
        self.bookmarks_tree.clear()
        self._text_search_timer.stop()
        self.search_input.clear()
        self.search_count_label.setText("0/0")
        self.search_status_label.setText("")
        if hasattr(self, "ai_status_label"):
            self._ai_history_document_path = None
            self.ai_reference_map.clear()
            self.ai_chat_view.clear()
            self._sync_ai_panel()
        if hasattr(self, "central_stack"):
            self._refresh_menu_documents()
            self.central_stack.setCurrentIndex(1)

    def _render_page(self, session: DocumentSession, page_number: int) -> QPixmap:
        page = session.doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE), alpha=False)
        image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def _placeholder_page(self, session: DocumentSession, page_number: int) -> QPixmap:
        pixmap = QPixmap(8, 8)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("#DDDDDD"), 1))
        painter.drawRect(0, 0, 7, 7)
        painter.end()
        return pixmap

    def _ensure_rendered_pages(self, session: DocumentSession) -> None:
        if len(session.rendered_pages) == len(session.doc) and len(session.page_display_sizes) == len(session.doc):
            return
        session.page_display_sizes = []
        for page_number in range(1, len(session.doc) + 1):
            page = session.doc.load_page(page_number - 1)
            session.page_display_sizes.append(
                (
                    max(1, int(round(float(page.rect.width) * RENDER_SCALE))),
                    max(1, int(round(float(page.rect.height) * RENDER_SCALE))),
                )
            )
        session.rendered_pages = [
            self._placeholder_page(session, page_number)
            for page_number in range(1, len(session.doc) + 1)
        ]
        session.rendered_page_numbers.clear()

    def _ensure_rendered_page(self, session: DocumentSession, page_number: int) -> None:
        if not (1 <= page_number <= len(session.doc)):
            return
        self._ensure_rendered_pages(session)
        if page_number in session.rendered_page_numbers:
            return
        pixmap = self._render_page(session, page_number)
        session.rendered_pages[page_number - 1] = pixmap
        session.rendered_page_numbers.add(page_number)
        for view in self.reader_views:
            if self._view_has_session(session, view):
                view.update_page_image(page_number, pixmap)

    def _ensure_rendered_window(self, session: DocumentSession, center_page: int) -> None:
        start = max(1, center_page - LAZY_RENDER_RADIUS)
        end = min(len(session.doc), center_page + LAZY_RENDER_RADIUS)
        for page_number in range(start, end + 1):
            self._ensure_rendered_page(session, page_number)

    def _ensure_rendered_visible_pages(self, session: DocumentSession, view: Optional[PdfView] = None) -> None:
        target_view = view or self.view
        visible_pages = target_view.visible_pages()
        if not visible_pages:
            visible_page = target_view.current_visible_page()
            if visible_page is None:
                return
            visible_pages = [visible_page]

        pages_to_render: List[int] = []
        seen = set()
        for visible_page in visible_pages:
            start = max(1, visible_page - LAZY_RENDER_RADIUS)
            end = min(len(session.doc), visible_page + LAZY_RENDER_RADIUS)
            for page_number in range(start, end + 1):
                if page_number not in seen:
                    pages_to_render.append(page_number)
                    seen.add(page_number)

        for page_number in pages_to_render:
            self._ensure_rendered_page(session, page_number)

    def _view_has_session(self, session: DocumentSession, view: Optional[PdfView] = None) -> bool:
        target_view = view or self.view
        return getattr(target_view, "_pdf_reader_session_id", None) == id(session)

    def _reader_view_is_active(self) -> bool:
        return not hasattr(self, "central_stack") or self.central_stack.currentIndex() == 0

    def _current_view_state(self) -> Optional[ViewState]:
        session = self.current_session()
        if session is None:
            return None
        visible_page = self.view.current_visible_page()
        if visible_page is not None:
            session.current_page = visible_page
        return ViewState(
            page=session.current_page,
            scroll_x=self.view.horizontalScrollBar().value(),
            scroll_y=self.view.verticalScrollBar().value(),
            zoom=self.view.zoom_factor(),
        )

    def _save_active_session_view(self) -> None:
        session = self.current_session()
        if session is None or not self._reader_view_is_active() or not self._view_has_session(session):
            return
        self._capture_session_view_layout(session)

    def _restore_view_state(self, session: DocumentSession, state: ViewState) -> None:
        self._show_page_for_session(session, state.page, push_history=False)
        self.view.set_zoom_factor(state.zoom)
        self.view.horizontalScrollBar().setValue(state.scroll_x)
        self.view.verticalScrollBar().setValue(state.scroll_y)
        if self.view.current_visible_page() != state.page:
            self.view.scroll_to_page(state.page)
        session.current_page = state.page
        self._update_status()

    def _default_state_for_session(self, session: DocumentSession) -> ViewState:
        return ViewState(page=max(1, session.current_page), scroll_x=0, scroll_y=0, zoom=1.0)

    def _push_history(self, session: DocumentSession) -> None:
        if self._suspend_history or self._switching_docs:
            return
        state = self._current_view_state()
        if state is not None:
            session.history_back.append(state)
            if len(session.history_back) > MAX_HISTORY:
                session.history_back.pop(0)
            session.history_forward.clear()

    def _show_page_for_session(self, session: DocumentSession, page_number: int, push_history: bool = True) -> None:
        page_number = max(1, min(page_number, len(session.doc)))
        is_current = session is self.current_session()

        if push_history and is_current and page_number != session.current_page:
            self._push_history(session)

        session.current_page = page_number

        if is_current:
            self._hide_equation_preview()
            self._ensure_rendered_pages(session)
            self._ensure_rendered_window(session, page_number)
            was_switching = self._switching_docs
            self._switching_docs = True
            try:
                for view in self.reader_views:
                    if not self._view_has_session(session, view):
                        self._show_session_in_view(view, session)
                        if view is not self.view:
                            view.scroll_to_page(page_number)
                self._ensure_rendered_window(session, page_number)
                self.view.scroll_to_page(page_number)
                self._ensure_rendered_visible_pages(session, self.view)
            finally:
                self._switching_docs = was_switching
            self._redraw_selection_lines()
            self._sync_text_search_panel()
            self.page_input.blockSignals(True)
            self.page_input.setMaximum(len(session.doc))
            self.page_input.setValue(page_number)
            self.page_input.blockSignals(False)
            self._refresh_annotation_overlays(session)
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

    def open_ai_chat(self) -> None:
        session = self.current_session()
        state = self._current_view_state() if session is not None else None
        self.ai_dock.show()
        self.ai_dock.raise_()
        self._sync_ai_panel()
        if not self._current_ai_api_key():
            self.ai_api_key_input.setFocus()
        else:
            self.ai_question_input.setFocus()
        if session is not None and state is not None:
            QTimer.singleShot(0, lambda s=session, st=state: self._restore_ai_chat_open_state(s, st))
            QTimer.singleShot(60, lambda s=session, st=state: self._restore_ai_chat_open_state(s, st))

    def _restore_ai_chat_open_state(self, session: DocumentSession, state: ViewState) -> None:
        if session is not self.current_session():
            return
        focus_key = not self._current_ai_api_key()
        self._suspend_history = True
        try:
            self.view.set_zoom_factor(state.zoom)
            self.view.horizontalScrollBar().setValue(
                max(0, min(state.scroll_x, self.view.horizontalScrollBar().maximum()))
            )
            self.view.verticalScrollBar().setValue(
                max(0, min(state.scroll_y, self.view.verticalScrollBar().maximum()))
            )
            session.current_page = state.page
            session.saved_view_state = state
            self.page_input.blockSignals(True)
            self.page_input.setValue(state.page)
            self.page_input.blockSignals(False)
            self._ensure_rendered_visible_pages(session, self.view)
            self._update_status()
        finally:
            self._suspend_history = False
        if focus_key:
            self.ai_api_key_input.setFocus()
        else:
            self.ai_question_input.setFocus()

    def _ai_chat_visibility_changed(self, visible: bool) -> None:
        if visible:
            self._sync_ai_panel()

    def _current_ai_api_key(self) -> str:
        typed = self.ai_api_key_input.text().strip() if hasattr(self, "ai_api_key_input") else ""
        if typed:
            return typed
        return openai_api_key(self.base_dir)

    def _current_ai_model(self) -> str:
        return self.ai_model_combo.currentText().strip() or DEFAULT_CHAT_MODELS[0]

    def _current_embedding_model(self) -> str:
        return str(self.ai_settings.get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL

    def _refresh_ai_bookmark_scope_combo(self) -> None:
        if not hasattr(self, "ai_bookmark_scope_combo"):
            return

        current_range = self.ai_question_ranges_input.text().strip()
        previous_data = self.ai_bookmark_scope_combo.currentData()
        session = self.current_session()
        items: List[Tuple[str, str]] = [("All scanned pages", ""), ("Manual page range", "__manual__")]

        if session is not None:
            items.extend(self._ai_bookmark_scope_items(session))

        self.ai_bookmark_scope_combo.blockSignals(True)
        try:
            self.ai_bookmark_scope_combo.clear()
            for label, value in items:
                self.ai_bookmark_scope_combo.addItem(label, value)

            target_index = 0
            if current_range:
                for index in range(self.ai_bookmark_scope_combo.count()):
                    if self.ai_bookmark_scope_combo.itemData(index) == current_range:
                        target_index = index
                        break
                else:
                    target_index = 1
            elif previous_data == "__manual__":
                target_index = 1
            self.ai_bookmark_scope_combo.setCurrentIndex(target_index)
        finally:
            self.ai_bookmark_scope_combo.blockSignals(False)

    def _ai_bookmark_scope_items(self, session: DocumentSession) -> List[Tuple[str, str]]:
        toc = self._get_pdf_toc(session)
        out: List[Tuple[str, str]] = []
        page_count = len(session.doc)

        normalized: List[Tuple[int, str, int]] = []
        for entry in toc:
            if len(entry) < 3:
                continue
            try:
                level = max(1, int(entry[0]))
                title = str(entry[1]).strip() or "Untitled bookmark"
                page = max(1, min(int(entry[2]), page_count))
            except Exception:
                continue
            normalized.append((level, title, page))

        for index, (level, title, start_page) in enumerate(normalized):
            end_page = page_count
            for next_level, _next_title, next_page in normalized[index + 1 :]:
                if next_level <= level:
                    end_page = max(start_page, next_page - 1)
                    break
            page_range = str(start_page) if start_page == end_page else f"{start_page}-{end_page}"
            indent = "  " * min(level - 1, 5)
            out.append((f"{indent}{title} (p. {page_range})", page_range))

        for bookmark in session.user_bookmarks:
            page = max(1, min(int(bookmark.page), page_count))
            title = str(bookmark.title).strip() or "Bookmark"
            out.append((f"Added: {title} (p. {page})", str(page)))
        return out

    def _ai_bookmark_scope_changed(self, _index: int) -> None:
        if not hasattr(self, "ai_bookmark_scope_combo"):
            return
        value = self.ai_bookmark_scope_combo.currentData()
        if value == "__manual__":
            self.ai_question_ranges_input.setFocus()
            return
        self.ai_question_ranges_input.blockSignals(True)
        try:
            self.ai_question_ranges_input.setText(str(value or ""))
        finally:
            self.ai_question_ranges_input.blockSignals(False)
        if value:
            self.statusBar().showMessage(f"AI question scope set to pages {value}", 3000)
        else:
            self.statusBar().showMessage("AI question scope set to all scanned pages", 3000)

    def _ai_question_range_text_edited(self, _text: str) -> None:
        if not hasattr(self, "ai_bookmark_scope_combo"):
            return
        manual_index = self.ai_bookmark_scope_combo.findData("__manual__")
        if manual_index >= 0 and self.ai_bookmark_scope_combo.currentIndex() != manual_index:
            self.ai_bookmark_scope_combo.blockSignals(True)
            try:
                self.ai_bookmark_scope_combo.setCurrentIndex(manual_index)
            finally:
                self.ai_bookmark_scope_combo.blockSignals(False)

    def save_ai_api_key(self) -> None:
        api_key = self.ai_api_key_input.text().strip()
        if not api_key:
            self.statusBar().showMessage("Enter an API key before saving", 4000)
            self.ai_api_key_input.setFocus()
            return
        self.ai_settings.update(
            {
                "api_key": api_key,
                "chat_model": self._current_ai_model(),
                "embedding_model": self._current_embedding_model(),
                "retrieval_limit": self.ai_chunk_limit_spin.value(),
            }
        )
        try:
            save_ai_settings(self.base_dir, self.ai_settings)
        except Exception as exc:
            QMessageBox.warning(self, "AI Chat", f"Could not save API key:\n{exc}")
            return
        self._sync_ai_panel()
        self.statusBar().showMessage("Saved OpenAI API key locally", 5000)

    def _sync_ai_panel(self) -> None:
        if not hasattr(self, "ai_status_label"):
            return
        self._sync_ai_cost_label()
        self._refresh_ai_bookmark_scope_combo()
        session = self.current_session()
        if session is None:
            if self._current_ai_api_key():
                self.ai_status_label.setText("Open a PDF, then scan it before asking questions.")
            else:
                self.ai_status_label.setText("Enter and save an OpenAI API key, then open a PDF.")
            self.ai_scan_button.setEnabled(False)
            self.ai_ask_button.setEnabled(False)
            self.ai_clear_history_button.setEnabled(False)
            self.ai_bookmark_scope_combo.setEnabled(False)
            return

        has_key = bool(self._current_ai_api_key())
        embedding_model = self._current_embedding_model()
        has_index = has_current_index(self.base_dir, session.path, embedding_model)
        self.ai_scan_button.setEnabled(has_key)
        self.ai_ask_button.setEnabled(has_key and has_index)
        self.ai_clear_history_button.setEnabled(True)
        self.ai_bookmark_scope_combo.setEnabled(True)
        self._load_ai_history_for_session(session)
        if not has_key:
            self.ai_status_label.setText("Enter and save an OpenAI API key first.")
        elif has_index:
            self.ai_status_label.setText("This document is scanned. Ask a question or rescan after PDF changes.")
        else:
            self.ai_status_label.setText("Scan this document before asking questions.")

    def _sync_ai_cost_label(self) -> None:
        if not hasattr(self, "ai_cost_label"):
            return
        usage = session_usage_totals()
        self.ai_cost_label.setText(
            "Cost since app open: "
            f"${usage.total_usd:.4f} "
            f"({usage.request_count} API request{'s' if usage.request_count != 1 else ''}, "
            f"{usage.input_tokens:,} input tokens, {usage.output_tokens:,} output tokens)"
        )

    def start_ai_document_scan(self) -> None:
        session = self.current_session()
        if session is None:
            self.statusBar().showMessage("Open a PDF before scanning", 4000)
            return
        api_key = self._current_ai_api_key()
        if not api_key:
            self.ai_api_key_input.setFocus()
            self.statusBar().showMessage("Enter an OpenAI API key first", 5000)
            return
        self.ai_settings.update(
            {
                "api_key": api_key,
                "chat_model": self._current_ai_model(),
                "embedding_model": self._current_embedding_model(),
                "retrieval_limit": self.ai_chunk_limit_spin.value(),
            }
        )
        try:
            save_ai_settings(self.base_dir, self.ai_settings)
        except Exception:
            pass

        try:
            page_ranges = self._parse_ai_page_ranges(self.ai_page_ranges_input.text(), len(session.doc))
        except ValueError as exc:
            QMessageBox.warning(self, "AI Scan Pages", str(exc))
            self.ai_page_ranges_input.setFocus()
            return

        range_label = format_page_ranges(page_ranges)

        self.ai_scan_button.setEnabled(False)
        self.ai_ask_button.setEnabled(False)
        if range_label:
            self.ai_status_label.setText(f"Starting AI scan for pages {range_label}...")
            self.statusBar().showMessage(f"Scanning pages {range_label} for AI chunks...", 0)
        else:
            self.ai_status_label.setText("Starting full AI scan...")
            self.statusBar().showMessage("Scanning full document for AI chunks...", 0)
        worker = AIIndexWorker(self.base_dir, session.path, api_key, self._current_embedding_model(), page_ranges)
        worker.progress.connect(self._ai_scan_progress)
        worker.completed.connect(self._ai_scan_completed)
        worker.failed.connect(self._ai_scan_failed)
        worker.finished.connect(lambda w=worker: self._ai_worker_finished(w))
        self._ai_threads.append(worker)
        worker.start()

    @staticmethod
    def _parse_ai_page_ranges(text: str, total_pages: int) -> Optional[List[Tuple[int, int]]]:
        raw = text.strip()
        if not raw:
            return None

        ranges: List[Tuple[int, int]] = []
        for part in re.split(r"[,;]\s*", raw):
            token = part.strip()
            if not token:
                continue
            match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", token)
            if not match:
                raise ValueError(
                    "Use page ranges like 1-20, 45, 70-90. Leave the field blank to scan the whole document."
                )
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if start < 1 or end < 1:
                raise ValueError("Page numbers must be 1 or higher.")
            if start > total_pages or end > total_pages:
                raise ValueError(f"Page range {token} is outside this document. It has {total_pages} page(s).")
            ranges.append((min(start, end), max(start, end)))

        if not ranges:
            return None
        ranges.sort()
        merged: List[Tuple[int, int]] = []
        for start, end in ranges:
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def _ai_scan_progress(self, message: str) -> None:
        self._sync_ai_cost_label()
        self.ai_status_label.setText(message)
        self.statusBar().showMessage(message, 0)

    def _ai_scan_completed(self, result: object) -> None:
        self._sync_ai_cost_label()
        data = result if isinstance(result, dict) else {}
        chunk_count = int(data.get("chunk_count") or 0)
        added_chunk_count = int(data.get("added_chunk_count") or chunk_count)
        page_count = int(data.get("page_count") or 0)
        text_page_count = int(data.get("text_page_count") or 0)
        scanned_text_page_count = int(data.get("scanned_text_page_count") or text_page_count)
        range_label = str(data.get("page_ranges") or "")
        if range_label:
            self.ai_status_label.setText(
                f"Added pages {range_label}: {added_chunk_count} new chunk(s) from "
                f"{scanned_text_page_count} page(s). Index now has {chunk_count} chunk(s) "
                f"from {text_page_count} / {page_count} page(s) with extractable text."
            )
        else:
            self.ai_status_label.setText(
                f"Scanned {chunk_count} chunk(s) from {text_page_count} / {page_count} page(s) with extractable text."
            )
        self.statusBar().showMessage(f"AI scan complete: {chunk_count} chunk(s)", 6000)
        self._sync_ai_panel()

    def _ai_scan_failed(self, message: str) -> None:
        self._sync_ai_cost_label()
        self.ai_status_label.setText(f"AI scan failed: {message}")
        self.statusBar().showMessage("AI scan failed", 6000)
        QMessageBox.warning(self, "AI Scan", message)
        self._sync_ai_panel()

    def ask_ai_question(self) -> None:
        session = self.current_session()
        if session is None:
            self.statusBar().showMessage("Open a PDF before asking AI", 4000)
            return
        api_key = self._current_ai_api_key()
        if not api_key:
            self.ai_api_key_input.setFocus()
            self.statusBar().showMessage("Enter an OpenAI API key first", 5000)
            return
        embedding_model = self._current_embedding_model()
        if not has_current_index(self.base_dir, session.path, embedding_model):
            self.statusBar().showMessage("Scan this document before asking questions", 5000)
            self.ai_status_label.setText("Scan this document before asking questions.")
            return
        question = self.ai_question_input.toPlainText().strip()
        if not question:
            self.ai_question_input.setFocus()
            return
        try:
            question_page_ranges = self._parse_ai_page_ranges(self.ai_question_ranges_input.text(), len(session.doc))
        except ValueError as exc:
            QMessageBox.warning(self, "AI Question Pages", str(exc))
            self.ai_question_ranges_input.setFocus()
            return

        chat_model = self._current_ai_model()
        self.ai_settings.update(
            {
                "api_key": api_key,
                "chat_model": chat_model,
                "embedding_model": embedding_model,
                "retrieval_limit": self.ai_chunk_limit_spin.value(),
            }
        )
        try:
            save_ai_settings(self.base_dir, self.ai_settings)
        except Exception:
            pass

        self.ai_question_input.clear()
        self.ai_ask_button.setEnabled(False)
        self.ai_scan_button.setEnabled(False)
        question_range_label = format_page_ranges(question_page_ranges)
        if question_range_label:
            self.ai_status_label.setText(f"Asking {chat_model} using pages {question_range_label}...")
            self.statusBar().showMessage(f"Asking {chat_model} using pages {question_range_label}...", 0)
        else:
            self.ai_status_label.setText(f"Asking {chat_model}...")
            self.statusBar().showMessage(f"Asking {chat_model}...", 0)
        worker = AIQuestionWorker(
            self.base_dir,
            session.path,
            api_key,
            question,
            chat_model,
            embedding_model,
            question_page_ranges,
            self.ai_chunk_limit_spin.value(),
        )
        worker.completed.connect(self._ai_question_completed)
        worker.failed.connect(self._ai_question_failed)
        worker.finished.connect(lambda w=worker: self._ai_worker_finished(w))
        self._ai_threads.append(worker)
        worker.start()

    def _ai_question_completed(self, result: object) -> None:
        self._sync_ai_cost_label()
        data = result if isinstance(result, dict) else {}
        question = str(data.get("question") or "")
        answer = str(data.get("answer") or "")
        page_ranges = str(data.get("page_ranges") or "")
        retrieval_limit = int(data.get("retrieval_limit") or self.ai_chunk_limit_spin.value())
        references = self._coerce_ai_references(data.get("references"))
        self._append_ai_exchange(question, answer, references, page_ranges=page_ranges, retrieval_limit=retrieval_limit)
        self.ai_status_label.setText("Answer ready. Click page references to jump in the PDF.")
        self.statusBar().showMessage("AI answer ready", 5000)
        self._sync_ai_panel()

    def _ai_question_failed(self, message: str) -> None:
        self._sync_ai_cost_label()
        self.ai_status_label.setText(f"AI question failed: {message}")
        self.statusBar().showMessage("AI question failed", 6000)
        QMessageBox.warning(self, "AI Chat", message)
        self._sync_ai_panel()

    def _ai_worker_finished(self, worker: QThread) -> None:
        if worker in self._ai_threads:
            self._ai_threads.remove(worker)
        worker.deleteLater()

    def _load_ai_history_for_session(self, session: DocumentSession, force: bool = False) -> None:
        if not hasattr(self, "ai_chat_view"):
            return
        if not force and self._ai_history_document_path == session.path:
            return
        self._ai_history_document_path = session.path
        self.ai_reference_map.clear()
        self.ai_chat_view.clear()
        self._ai_message_counter = 0
        for entry in load_chat_history(self.base_dir, session.path):
            question = str(entry.get("question") or "")
            answer = str(entry.get("answer") or "")
            model = str(entry.get("model") or self._current_ai_model())
            created_at = int(entry.get("created_at") or 0)
            page_ranges = str(entry.get("page_ranges") or "")
            retrieval_limit = int(entry.get("retrieval_limit") or 0)
            references = self._deserialize_ai_references(entry.get("references", {}))
            if question or answer:
                self._append_ai_exchange(
                    question,
                    answer,
                    references,
                    save_history=False,
                    model=model,
                    created_at=created_at,
                    page_ranges=page_ranges,
                    retrieval_limit=retrieval_limit,
                )

    def clear_ai_history(self) -> None:
        session = self.current_session()
        if session is None:
            return
        result = QMessageBox.question(
            self,
            "Clear AI History",
            f"Clear AI question history for {self._session_display_name(session.path)}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if result != QMessageBox.Yes:
            return
        clear_chat_history(self.base_dir, session.path)
        self._load_ai_history_for_session(session, force=True)
        self.statusBar().showMessage("Cleared AI history for this document", 5000)

    def _append_ai_exchange(
        self,
        question: str,
        answer: str,
        references: Dict[str, AIReference],
        save_history: bool = True,
        model: Optional[str] = None,
        created_at: int = 0,
        page_ranges: str = "",
        retrieval_limit: int = 0,
    ) -> None:
        self._ai_message_counter += 1
        message_id = f"m{self._ai_message_counter}"
        question_html = self._plain_text_to_html(question)
        answer_html = self._answer_to_html(answer, message_id, references)
        model_name = model or self._current_ai_model()
        time_html = ""
        if created_at:
            try:
                time_html = f" <span style='color:#777777;'>({time.strftime('%Y-%m-%d %H:%M', time.localtime(created_at))})</span>"
            except Exception:
                time_html = ""
        scope_html = f"<br><span style='color:#777777;'>Pages: {html.escape(page_ranges)}</span>" if page_ranges else ""
        chunks_html = (
            f"<br><span style='color:#777777;'>Chunks requested: {retrieval_limit}</span>"
            if retrieval_limit
            else ""
        )
        self.ai_chat_view.append(
            "<div style='margin: 8px 0;'>"
            "<b>You</b><br>"
            f"{question_html}"
            f"{scope_html}"
            f"{chunks_html}"
            "</div>"
            "<div style='margin: 10px 0 16px 0;'>"
            f"<b>AI ({html.escape(model_name)})</b>{time_html}<br>"
            f"{answer_html}"
            "</div>"
        )
        self.ai_chat_view.verticalScrollBar().setValue(self.ai_chat_view.verticalScrollBar().maximum())
        if save_history:
            session = self.current_session()
            if session is not None:
                append_chat_history(
                    self.base_dir,
                    session.path,
                    {
                        "created_at": int(time.time()),
                        "model": model_name,
                        "question": question,
                        "answer": answer,
                        "page_ranges": page_ranges,
                        "retrieval_limit": retrieval_limit,
                        "references": self._serialize_ai_references(references),
                    },
                )

    @staticmethod
    def _serialize_ai_references(references: Dict[str, AIReference]) -> Dict[str, dict]:
        return {
            label: {
                "page": ref.page,
                "x": ref.x,
                "y": ref.y,
                "rect": list(ref.rect),
                "text": ref.text,
            }
            for label, ref in references.items()
        }

    @staticmethod
    def _deserialize_ai_references(raw: object) -> Dict[str, AIReference]:
        return PdfReaderWindow._coerce_ai_references(raw)

    @staticmethod
    def _coerce_ai_references(raw: object) -> Dict[str, AIReference]:
        if not isinstance(raw, dict):
            return {}
        references: Dict[str, AIReference] = {}
        for label, value in raw.items():
            if isinstance(value, AIReference):
                references[str(label)] = value
                continue
            if not isinstance(value, dict):
                continue
            try:
                references[str(label)] = AIReference(
                    page=max(1, int(value.get("page") or 1)),
                    x=max(0.0, float(value.get("x") or 0.0)),
                    y=max(0.0, float(value.get("y") or 0.0)),
                    rect=PdfReaderWindow._deserialize_ai_reference_rect(value.get("rect")),
                    text=str(value.get("text") or ""),
                )
            except Exception:
                continue
        return references

    @staticmethod
    def _deserialize_ai_reference_rect(raw: object) -> Tuple[float, float, float, float]:
        if isinstance(raw, list) and len(raw) >= 4:
            try:
                return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
            except Exception:
                pass
        return (0.0, 0.0, 0.0, 0.0)

    def _answer_to_html(self, answer: str, message_id: str, references: Dict[str, AIReference]) -> str:
        escaped = self._ai_text_to_html(answer)
        cited_labels = set()

        def replace_citation(match) -> str:
            label = match.group(1)
            ref = references.get(label)
            if ref is None:
                cited_labels.add(label)
                return (
                    "<span style='color:#FFB74D;' "
                    f"title='No retrieved source was available for {html.escape(label)}'>"
                    f"[missing {html.escape(label)}]</span>"
                )
            anchor_id = f"{message_id}:{label}"
            self.ai_reference_map[anchor_id] = ref
            cited_labels.add(label)
            return self._ai_reference_anchor(anchor_id, ref)

        escaped = re.sub(r"\[\[\s*(C\d+)\s*\]\]", replace_citation, escaped)
        if references:
            links = []
            for label, ref in references.items():
                anchor_id = f"{message_id}:{label}"
                self.ai_reference_map[anchor_id] = ref
                prefix = f"{html.escape(label)}: "
                links.append(prefix + self._ai_reference_anchor(anchor_id, ref, include_coordinates=True))
            if links:
                escaped += "<br><br><b>Retrieved sources:</b> " + ", ".join(links)
        elif cited_labels:
            escaped += (
                "<br><br><span style='color:#FFB74D;'>"
                "This saved answer has citation labels but no stored source coordinates. Ask again to create clickable references."
                "</span>"
            )
        return escaped

    @staticmethod
    def _ai_reference_anchor(anchor_id: str, ref: AIReference, include_coordinates: bool = False) -> str:
        return (
            f"<a href='ai-cite:{html.escape(anchor_id)}' "
            f"title='{html.escape(PdfReaderWindow._ai_reference_tooltip(ref))}'>"
            f"{html.escape(PdfReaderWindow._ai_reference_link_text(ref, include_coordinates=include_coordinates))}</a>"
        )

    @staticmethod
    def _ai_reference_link_text(ref: AIReference, include_coordinates: bool = False) -> str:
        x = ref.x
        y = ref.y
        if ref.rect and len(ref.rect) >= 4 and (float(ref.rect[2]) > float(ref.rect[0]) or float(ref.rect[3]) > float(ref.rect[1])):
            x = float(ref.rect[0])
            y = float(ref.rect[1])
        if include_coordinates and (x > 0 or y > 0):
            return f"[p. {ref.page}, x {x:.0f}, y {y:.0f}]"
        return f"[p. {ref.page}]"

    @staticmethod
    def _ai_reference_tooltip(ref: AIReference) -> str:
        x = ref.x
        y = ref.y
        if ref.rect and len(ref.rect) >= 4 and (float(ref.rect[2]) > float(ref.rect[0]) or float(ref.rect[3]) > float(ref.rect[1])):
            x = float(ref.rect[0])
            y = float(ref.rect[1])
        if x > 0 or y > 0:
            return f"Jump to page {ref.page}, x {x:.0f}, y {y:.0f}"
        return f"Jump to page {ref.page}"

    @staticmethod
    def _plain_text_to_html(text: str) -> str:
        return html.escape(text).replace("\n", "<br>")

    @staticmethod
    def _ai_text_to_html(text: str) -> str:
        placeholders: List[Tuple[str, str]] = []

        def protect(pattern: str, css_class: str, source: str) -> str:
            def replace(match) -> str:
                token = f"@@AI_MATH_{len(placeholders)}@@"
                content = PdfReaderWindow._math_to_display_text(match.group(1).strip())
                if css_class == "ai-math-block":
                    html_text = f"<div class='{css_class}'>{html.escape(content)}</div>"
                else:
                    html_text = f"<span class='{css_class}'>{html.escape(content)}</span>"
                placeholders.append((token, html_text))
                return token

            return re.sub(pattern, replace, source, flags=re.DOTALL)

        protected = PdfReaderWindow._cleanup_broken_latex_text(text)
        protected = protect(r"\\\[(.*?)\\\]", "ai-math-block", protected)
        protected = protect(r"\$\$(.*?)\$\$", "ai-math-block", protected)
        protected = protect(r"\\\((.*?)\\\)", "ai-math-inline", protected)
        protected = protect(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", "ai-math-inline", protected)

        html_lines: List[str] = []
        for line in protected.splitlines():
            stripped = line.strip()
            if not stripped:
                html_lines.append("<div style='height:6px;'></div>")
            elif PdfReaderWindow._looks_like_formula_line(line):
                html_lines.append(f"<div class='ai-math-block'>{html.escape(stripped)}</div>")
            elif re.match(r"^[-*•]\s+", stripped):
                html_lines.append(f"<div style='margin: 5px 0 5px 0;'>{html.escape(stripped)}</div>")
            else:
                html_lines.append(f"<div style='margin: 3px 0;'>{html.escape(stripped)}</div>")
        out = "".join(html_lines)
        for token, replacement in placeholders:
            out = out.replace(html.escape(token), replacement)
            out = out.replace(token, replacement)
        return out

    @staticmethod
    def _cleanup_broken_latex_text(text: str) -> str:
        out = text
        # Model/PDF text extraction sometimes drops LaTeX braces and backslashes,
        # yielding strings like "mathrm3SAT" or "VERTEXtext-COVER".
        out = out.replace(r"\langle", "⟨").replace(r"\rangle", "⟩")
        out = out.replace("langle", "⟨").replace("rangle", "⟩")
        out = out.replace(r"\varphi", "φ").replace(r"\phi", "φ")
        out = out.replace("varphi", "φ")
        out = re.sub(r"\\k\b", "k", out)
        out = re.sub(r"\\?mathrm\s*\{([^{}]+)\}", r"\1", out)
        out = re.sub(r"\\?text\s*\{([^{}]+)\}", r"\1", out)
        out = re.sub(r"\\?operatorname\s*\{([^{}]+)\}", r"\1", out)
        out = re.sub(r"\\?mathsf\s*\{([^{}]+)\}", r"\1", out)
        out = re.sub(r"\\?mathtt\s*\{([^{}]+)\}", r"\1", out)
        out = re.sub(r"\\?mathbf\s*\{([^{}]+)\}", r"\1", out)
        out = re.sub(r"\\?mathrm(?=[A-Za-z0-9])", "", out)
        out = re.sub(r"\\?text(?=[A-Z0-9-])", "", out)
        out = re.sub(r"\\?operatorname(?=[A-Za-z0-9])", "", out)
        out = re.sub(r"\\?mathsf(?=[A-Za-z0-9])", "", out)
        out = re.sub(r"\\?mathtt(?=[A-Za-z0-9])", "", out)
        out = re.sub(r"\\?mathbf(?=[A-Za-z0-9])", "", out)
        return out

    @staticmethod
    def _looks_like_formula_line(line: str) -> bool:
        stripped = line.strip()
        if len(stripped) < 3 or len(stripped) > 220:
            return False
        if re.search(r"\[\[\s*C\d+\s*\]\]", stripped):
            return False
        if stripped.startswith(("-", "*", "•")):
            return False
        math_markers = (
            "=",
            "\\frac",
            "\\sum",
            "\\int",
            "\\partial",
            "\\nabla",
            "\\alpha",
            "\\beta",
            "\\gamma",
            "\\delta",
            "\\lambda",
            "\\omega",
            "^",
            "_",
        )
        if not any(marker in stripped for marker in math_markers):
            return False
        letters = sum(ch.isalpha() for ch in stripped)
        operators = sum(ch in "=+-*/^_()[]{}<>|\\∂∇∑∫√≈≤≥≠" for ch in stripped)
        return operators >= 2 and letters <= max(18, len(stripped) * 0.55)

    @staticmethod
    def _math_to_display_text(text: str) -> str:
        out = PdfReaderWindow._cleanup_broken_latex_text(text).strip()
        replacements = {
            r"\alpha": "α",
            r"\beta": "β",
            r"\gamma": "γ",
            r"\Gamma": "Γ",
            r"\delta": "δ",
            r"\Delta": "Δ",
            r"\epsilon": "ε",
            r"\varepsilon": "ε",
            r"\zeta": "ζ",
            r"\eta": "η",
            r"\theta": "θ",
            r"\Theta": "Θ",
            r"\vartheta": "ϑ",
            r"\iota": "ι",
            r"\kappa": "κ",
            r"\lambda": "λ",
            r"\Lambda": "Λ",
            r"\mu": "μ",
            r"\nu": "ν",
            r"\xi": "ξ",
            r"\Xi": "Ξ",
            r"\pi": "π",
            r"\Pi": "Π",
            r"\rho": "ρ",
            r"\sigma": "σ",
            r"\Sigma": "Σ",
            r"\tau": "τ",
            r"\upsilon": "υ",
            r"\phi": "φ",
            r"\varphi": "φ",
            r"\Phi": "Φ",
            r"\chi": "χ",
            r"\psi": "ψ",
            r"\Psi": "Ψ",
            r"\omega": "ω",
            r"\Omega": "Ω",
            r"\partial": "∂",
            r"\nabla": "∇",
            r"\infty": "∞",
            r"\sum": "∑",
            r"\prod": "∏",
            r"\int": "∫",
            r"\oint": "∮",
            r"\sqrt": "√",
            r"\cdot": "·",
            r"\times": "×",
            r"\pm": "±",
            r"\mp": "∓",
            r"\leq": "≤",
            r"\le": "≤",
            r"\geq": "≥",
            r"\ge": "≥",
            r"\neq": "≠",
            r"\ne": "≠",
            r"\approx": "≈",
            r"\sim": "∼",
            r"\propto": "∝",
            r"\in": "∈",
            r"\notin": "∉",
            r"\subset": "⊂",
            r"\subseteq": "⊆",
            r"\cup": "∪",
            r"\cap": "∩",
            r"\rightarrow": "→",
            r"\to": "→",
            r"\leftarrow": "←",
            r"\Rightarrow": "⇒",
            r"\Leftarrow": "⇐",
            r"\leftrightarrow": "↔",
            r"\forall": "∀",
            r"\exists": "∃",
            r"\neg": "¬",
            r"\land": "∧",
            r"\lor": "∨",
        }
        for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            out = out.replace(source, target)

        out = re.sub(r"\\(?:left|right|Bigl|Bigr|bigl|bigr|Big|big)\s*", "", out)
        out = PdfReaderWindow._replace_latex_groups(out, "frac", lambda a, b: f"({a})⁄({b})")
        out = PdfReaderWindow._replace_latex_one_group(out, "sqrt", lambda a: f"√({a})")
        out = PdfReaderWindow._replace_latex_one_group(out, "overline", lambda a: f"overline({a})")
        out = PdfReaderWindow._replace_latex_scripts(out)
        out = out.replace(r"\mid", "∣")
        out = re.sub(r"\\([A-Za-z]+)", r"\1", out)
        out = out.replace(r"\{", "{").replace(r"\}", "}")
        out = out.replace("{", "").replace("}", "")
        out = re.sub(r"\s+", " ", out).strip()
        return out

    @staticmethod
    def _replace_latex_groups(text: str, command: str, formatter) -> str:
        pattern = re.compile(rf"\\{command}\s*\{{([^{{}}]+)\}}\s*\{{([^{{}}]+)\}}")
        previous = None
        out = text
        while previous != out:
            previous = out
            out = pattern.sub(lambda match: formatter(match.group(1), match.group(2)), out)
        return out

    @staticmethod
    def _replace_latex_one_group(text: str, command: str, formatter) -> str:
        pattern = re.compile(rf"\\{command}\s*\{{([^{{}}]+)\}}")
        previous = None
        out = text
        while previous != out:
            previous = out
            out = pattern.sub(lambda match: formatter(match.group(1)), out)
        return out

    @staticmethod
    def _replace_latex_scripts(text: str) -> str:
        superscript_map = str.maketrans("0123456789+-=()nijk", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱʲᵏ")
        subscript_map = str.maketrans("0123456789+-=()aeijmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₘₙₒₚᵣₛₜᵤᵥₓ")

        def convert_script(raw: str, table, fallback_prefix: str) -> str:
            value = raw.strip("{}")
            converted = value.translate(table)
            if converted != value:
                return converted
            return f"{fallback_prefix}({value})"

        out = re.sub(
            r"\^\{([^{}]+)\}|\^([A-Za-z0-9+\-=()])",
            lambda match: convert_script(match.group(1) or match.group(2), superscript_map, "^"),
            text,
        )
        out = re.sub(
            r"_\{([^{}]+)\}|_([A-Za-z0-9+\-=()])",
            lambda match: convert_script(match.group(1) or match.group(2), subscript_map, "_"),
            out,
        )
        return out

    def open_ai_reference(self, url: QUrl) -> None:
        value = url.toString()
        prefix = "ai-cite:"
        if not value.startswith(prefix):
            return
        anchor_id = value[len(prefix) :]
        ref = self.ai_reference_map.get(anchor_id)
        session = self.current_session()
        if ref is None or session is None:
            return
        self.show_reader()
        self._show_page_for_session(session, ref.page, push_history=True)
        target_y = ref.y
        target_x: Optional[float] = ref.x if ref.x > 0 else None
        if ref.rect and len(ref.rect) >= 4 and (float(ref.rect[2]) > float(ref.rect[0]) or float(ref.rect[3]) > float(ref.rect[1])):
            target_x = float(ref.rect[0])
            target_y = float(ref.rect[1])
        target_x_scene = target_x * RENDER_SCALE if target_x is not None else None
        self.view.scroll_to_page(ref.page, target_y * RENDER_SCALE, target_x_scene)
        snippet = " ".join(ref.text.split())
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        coordinate_text = f", x {target_x:.0f}, y {target_y:.0f}" if target_x is not None else f", y {target_y:.0f}"
        self.statusBar().showMessage(
            f"AI reference: page {ref.page}{coordinate_text} - {snippet}",
            7000,
        )

    def open_text_search(self) -> None:
        session = self.current_session()
        if session is None:
            return
        self.search_dock.show()
        self.search_dock.raise_()
        if session.text_search_origin_state is None:
            session.text_search_origin_state = self._current_view_state()
        self.search_input.setFocus()
        self.search_input.selectAll()
        self._sync_text_search_panel()

    def close_text_search(self) -> None:
        self.search_dock.hide()

    def _text_search_visibility_changed(self, visible: bool) -> None:
        if visible:
            self._sync_text_search_panel()
            return
        if not hasattr(self, "_text_search_timer"):
            return
        self._text_search_timer.stop()
        self.view.clear_search_highlights()
        self.view.setFocus()

    def _text_search_edited(self, _text: str) -> None:
        session = self.current_session()
        if session is None:
            return
        self._reset_text_search(session)
        self._sync_text_search_panel()

    def _reset_text_search(self, session: DocumentSession) -> None:
        session.text_search_query = ""
        session.text_search_hits.clear()
        session.text_search_index = -1
        session.text_search_origin_state = None
        session.text_search_scan_page = 1
        session.text_search_complete = True
        self._text_search_timer.stop()
        self.view.clear_search_highlights()

    def _sync_text_search_panel(self) -> None:
        session = self.current_session()
        if session is None:
            self.search_count_label.setText("0/0")
            self.search_status_label.setText("")
            self.view.clear_search_highlights()
            return

        total = len(session.text_search_hits)
        current = session.text_search_index + 1 if 0 <= session.text_search_index < total else 0
        suffix = "+" if not session.text_search_complete else ""
        self.search_count_label.setText(f"{current}/{total}{suffix}")

        if session.text_search_query and not session.text_search_complete:
            self.search_status_label.setText(f"Searching page {session.text_search_scan_page} / {len(session.doc)}")
        elif session.text_search_query and total == 0:
            self.search_status_label.setText("No appearances found")
        else:
            self.search_status_label.setText("")

        if self.search_dock.isVisible():
            self.view.show_search_highlights(session.text_search_hits, session.text_search_index)

    def _start_text_search(self, session: DocumentSession, query: str) -> None:
        session.text_search_query = query
        session.text_search_hits = []
        session.text_search_index = -1
        session.text_search_origin_state = self._current_view_state()
        session.text_search_scan_page = 1
        session.text_search_complete = False
        self.view.clear_search_highlights()
        self._sync_text_search_panel()
        self._text_search_timer.start(0)

    def _scan_text_search_chunk(self) -> None:
        session = self.current_session()
        if session is None or session.text_search_complete:
            self._text_search_timer.stop()
            return

        query = session.text_search_query
        if not query:
            self._text_search_timer.stop()
            session.text_search_complete = True
            self._sync_text_search_panel()
            return

        pages_per_chunk = 4
        scanned = 0
        first_hit_added = False

        while session.text_search_scan_page <= len(session.doc) and scanned < pages_per_chunk:
            page_number = session.text_search_scan_page
            try:
                page = session.doc.load_page(page_number - 1)
                rects = page.search_for(query)
            except Exception as exc:
                print(f"[PDF Reader] Text search failed on page {page_number}: {exc}", flush=True)
                rects = []

            for rect in rects:
                session.text_search_hits.append(
                    SearchHit(
                        page=page_number,
                        rect=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                    )
                )
                if session.text_search_index < 0:
                    first_hit_added = True

            session.text_search_scan_page += 1
            scanned += 1

        if session.text_search_scan_page > len(session.doc):
            session.text_search_complete = True
            self._text_search_timer.stop()

        if first_hit_added:
            self._go_to_text_search_hit(0, push_history=True)
        else:
            self._sync_text_search_panel()

    def _go_to_text_search_hit(self, index: int, push_history: bool = True) -> None:
        session = self.current_session()
        if session is None or not session.text_search_hits:
            self._sync_text_search_panel()
            return

        index = index % len(session.text_search_hits)
        session.text_search_index = index
        hit = session.text_search_hits[index]
        self._show_page_for_session(session, hit.page, push_history=push_history)
        self.view.scroll_to_page(hit.page, hit.rect[1] * RENDER_SCALE)
        self._sync_text_search_panel()

    def search_next_text(self) -> None:
        session = self.current_session()
        if session is None:
            return
        query = self.search_input.text().strip()
        if not query:
            return
        if query != session.text_search_query:
            self._start_text_search(session, query)
            return
        if session.text_search_hits:
            next_index = 0 if session.text_search_index < 0 else session.text_search_index + 1
            self._go_to_text_search_hit(next_index, push_history=True)

    def search_previous_text(self) -> None:
        session = self.current_session()
        if session is None:
            return
        query = self.search_input.text().strip()
        if not query:
            return
        if query != session.text_search_query:
            self._start_text_search(session, query)
            return
        if session.text_search_hits:
            prev_index = len(session.text_search_hits) - 1 if session.text_search_index < 0 else session.text_search_index - 1
            self._go_to_text_search_hit(prev_index, push_history=True)

    def return_to_text_search_source(self) -> None:
        session = self.current_session()
        if session is None or session.text_search_origin_state is None:
            self.statusBar().showMessage("No saved source page for text search", 4000)
            return

        target = session.text_search_origin_state
        self._suspend_history = True
        try:
            self._restore_view_state(session, target)
            session.saved_view_state = target
        finally:
            self._suspend_history = False
        self._sync_text_search_panel()
        self.statusBar().showMessage(f"Returned to text search source page {target.page}", 4000)

    def _on_view_changed(self) -> None:
        session = self.current_session()
        if session is None:
            self.status_label.setText("No PDF loaded")
            self.zoom_label.setText("100%")
            return
        if self._suspend_view_updates:
            return
        sender = self.sender()
        changed_view = sender if isinstance(sender, PdfView) else self.view
        if not self._view_has_session(session, changed_view):
            return

        visible_page = changed_view.current_visible_page()
        self._ensure_rendered_visible_pages(session, changed_view)
        if changed_view is self.view and visible_page is not None and visible_page != session.current_page:
            session.current_page = visible_page
            self.page_input.blockSignals(True)
            self.page_input.setValue(visible_page)
            self.page_input.blockSignals(False)
        if changed_view is self.view and not self._switching_docs:
            self._save_active_session_view()
        if changed_view is self.view:
            self._update_status()

    def _remember_equation_source_page(self) -> None:
        session = self.current_session()
        if session is None:
            return
        state = self._current_view_state()
        if state is not None:
            session.last_equation_return_state = state

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

    def open_dropped_pdfs(self, paths: List[str]) -> None:
        if not paths:
            return
        for i, path in enumerate(paths):
            self.load_pdf(path, switch_to=(i == len(paths) - 1))
        self.statusBar().showMessage(f"Opened {len(paths)} dropped PDF file(s)", 5000)

    def dragEnterEvent(self, event) -> None:
        if pdf_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if pdf_paths_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = pdf_paths_from_mime_data(event.mimeData())
        if paths:
            event.acceptProposedAction()
            self.open_dropped_pdfs(paths)
            return
        super().dropEvent(event)

    def _a_txt_candidates_for_pdf(self, pdf_path: str) -> List[Path]:
        """Prefer a.txt next to the opened PDF, then a.txt next to this script."""
        candidates: List[Path] = []
        try:
            candidates.append(Path(pdf_path).resolve().parent / "a.txt")
        except Exception:
            candidates.append(Path(pdf_path).parent / "a.txt")
        candidates.append(self.a_txt_path)
        return candidates

    def _load_chapter_starts_for_pdf(self, pdf_path: str) -> Tuple[Optional[Dict[int, int]], Optional[str]]:
        starts, source = load_chapter_starts_from_candidate_paths(self._a_txt_candidates_for_pdf(pdf_path))
        if starts:
            print(
                f"[PDF Reader] Loaded chapter starts for {pdf_path} from {source}: {starts}",
                flush=True,
            )
            return starts, source

        print(
            f"[PDF Reader] No usable a.txt chapter-start map found for {pdf_path}. "
            f"Looked in: {[str(p) for p in self._a_txt_candidates_for_pdf(pdf_path)]}",
            flush=True,
        )
        return None, None

    def _reload_current_session_chapter_starts(self) -> None:
        """Reload a.txt for the current PDF and clear equation cache."""
        session = self.current_session()
        if session is None:
            return
        starts, source = self._load_chapter_starts_for_pdf(session.path)
        session.chapter_starts = starts
        session.chapter_starts_source = source
        session.equation_cache.clear()
        if starts:
            self.statusBar().showMessage(
                f"Reloaded a.txt chapter starts from {source}; equation cache cleared",
                8000,
            )
        else:
            self.statusBar().showMessage(
                "No usable a.txt was found next to the PDF or next to the script; equation cache cleared",
                9000,
            )

    def load_pdf(
        self,
        path: str,
        switch_to: bool = True,
        restored_state: Optional[ViewState] = None,
        restored_back: Optional[List[ViewState]] = None,
        restored_forward: Optional[List[ViewState]] = None,
        restored_viewer_states: Optional[List[ViewState]] = None,
        restored_active_view_index: int = 0,
        restored_user_bookmarks: Optional[List[UserBookmark]] = None,
        restored_selection_lines: Optional[List[SelectionLine]] = None,
        restored_comments: Optional[List[PdfComment]] = None,
        restored_covers: Optional[List[PdfCover]] = None,
        restored_last_equation_return_state: Optional[ViewState] = None,
        restored_equation_numbering_mode: str = EQUATION_MODE_SECTION,
        restored_equation_index: Optional[Dict[str, EquationLocation]] = None,
        restored_equation_rough_index: Optional[Dict[str, EquationLocation]] = None,
        restored_equation_pattern_index: Optional[Dict[str, EquationLocation]] = None,
        restored_equation_index_patterns: Optional[Dict[int, str]] = None,
        restored_equation_format: Optional[Dict[str, object]] = None,
        restored_manual_equation_samples: Optional[List[EquationCandidate]] = None,
        restored_equation_index_signature: Optional[dict] = None,
        restored_equation_lookup_mode: str = EQUATION_LOOKUP_SCAN,
        update_recent: bool = True,
    ) -> None:
        existing_index = self._find_session_by_path(path)
        if existing_index is not None:
            if update_recent:
                self._remember_recent_document(path, save=False)
            if switch_to:
                self.show_reader()
                self.doc_tabs.setCurrentIndex(existing_index)
                if self.current_index != existing_index:
                    self._switch_to_document(existing_index)
            if update_recent:
                self._save_session_state_now()
            return

        try:
            doc = fitz.open(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not open PDF:\n{exc}")
            return

        chapter_starts, chapter_starts_source = self._load_chapter_starts_for_pdf(path)
        section_starts = extract_section_starts_from_toc(doc)
        print(
            f"[PDF Reader] Loaded PDF {path}: extracted {len(section_starts)} numbered bookmark/TOC entries for section search",
            flush=True,
        )

        session = DocumentSession(
            path=path,
            doc=doc,
            current_page=1,
            section_starts=section_starts,
            history_back=restored_back[:] if restored_back else [],
            history_forward=restored_forward[:] if restored_forward else [],
            saved_view_state=restored_state,
            viewer_states=restored_viewer_states[:3] if restored_viewer_states else [],
            active_view_index=max(0, min(restored_active_view_index, 2)),
            chapter_starts=chapter_starts,
            chapter_starts_source=chapter_starts_source,
            user_bookmarks=restored_user_bookmarks[:] if restored_user_bookmarks else [],
            selection_lines=restored_selection_lines[:] if restored_selection_lines else [],
            comments=restored_comments[:] if restored_comments else [],
            saved_pdf_comments=self._load_pdf_reader_comments_from_pdf(doc),
            covers=restored_covers[:] if restored_covers else [],
            last_equation_return_state=restored_last_equation_return_state,
            equation_numbering_mode=(
                restored_equation_numbering_mode
                if restored_equation_numbering_mode in {EQUATION_MODE_SECTION, EQUATION_MODE_CHAPTER}
                else EQUATION_MODE_SECTION
            ),
            equation_index=dict(restored_equation_index or {}),
            equation_rough_index=dict(restored_equation_rough_index or restored_equation_index or {}),
            equation_pattern_index=dict(restored_equation_pattern_index or {}),
            equation_index_patterns=dict(restored_equation_index_patterns or {}),
            equation_format=dict(restored_equation_format or {}),
            manual_equation_samples=restored_manual_equation_samples[:] if restored_manual_equation_samples else [],
            equation_index_signature=restored_equation_index_signature,
            equation_lookup_mode=(
                EQUATION_LOOKUP_ROUGH_INDEX
                if restored_equation_lookup_mode == "index"
                else (
                    restored_equation_lookup_mode
                    if restored_equation_lookup_mode in EQUATION_LOOKUP_MODES
                    else EQUATION_LOOKUP_SCAN
                )
            ),
        )

        if restored_state is not None:
            session.current_page = restored_state.page
            if not session.viewer_states:
                session.viewer_states = [restored_state]
                session.active_view_index = 0
        elif not session.viewer_states:
            session.viewer_states = [self._default_state_for_session(session)]

        if session.equation_index and session.equation_index_signature is not None:
            current_signature = self._document_signature(session)
            stable_keys = ("size", "pages", "first_page_rect", "format", "equation_index_algorithm")
            if any(session.equation_index_signature.get(key) != current_signature.get(key) for key in stable_keys):
                session.equation_index.clear()
                session.equation_rough_index.clear()
                session.equation_pattern_index.clear()
                session.equation_index_patterns.clear()
                session.equation_format.clear()
                session.equation_index_signature = None
                if session.equation_lookup_mode in {EQUATION_LOOKUP_ROUGH_INDEX, EQUATION_LOOKUP_PATTERN_INDEX}:
                    session.equation_lookup_mode = EQUATION_LOOKUP_SCAN
                print(
                    f"[PDF Reader] Ignored saved equation index for {path}: document signature changed",
                    flush=True,
                )

        self.sessions.append(session)
        idx = len(self.sessions) - 1
        self.doc_tabs.addTab(self._session_display_name(path))
        self.doc_tabs.setTabToolTip(idx, path)
        if update_recent:
            self._remember_recent_document(path, save=False)

        if switch_to:
            self.show_reader()
            self.doc_tabs.setCurrentIndex(idx)
            if self.current_index != idx:
                self._switch_to_document(idx)
        if update_recent:
            self._save_session_state_now()

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
        self.recent_document_entries[self._path_key(session.path)] = self._session_persistence_entry(session)
        self._remember_recent_document(session.path, save=False)
        try:
            session.doc.close()
        except Exception:
            pass

        self.doc_tabs.blockSignals(True)
        self.doc_tabs.removeTab(index)
        self.doc_tabs.blockSignals(False)

        if not self.sessions:
            self._set_no_document_ui()
            self._refresh_menu_documents()
            return

        if index < self.current_index:
            new_index = self.current_index - 1
        elif index == self.current_index:
            new_index = min(index, len(self.sessions) - 1)
        else:
            new_index = self.current_index

        self.doc_tabs.setCurrentIndex(new_index)
        self._switch_to_document(new_index)
        self._refresh_menu_documents()

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

            self._text_search_timer.stop()
            self.search_input.blockSignals(True)
            self.search_input.setText(session.text_search_query)
            self.search_input.blockSignals(False)

            self._rebuild_reader_views_for_session(session)
            self.page_input.blockSignals(True)
            self.page_input.setValue(session.current_page)
            self.page_input.blockSignals(False)
            self._remember_document_activation(session)
        finally:
            self._switching_docs = False

        self._refresh_bookmarks_panel()
        self._refresh_annotation_overlays(session)
        self._sync_text_search_panel()
        self._sync_ai_panel()
        if self.search_dock.isVisible() and session.text_search_query and not session.text_search_complete:
            self._text_search_timer.start(0)
        self.view.setFocus()

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

    def return_to_equation_source_page(self) -> None:
        session = self.current_session()
        if session is None or session.last_equation_return_state is None:
            self.statusBar().showMessage("No saved source page for equation jump", 4000)
            return

        target = session.last_equation_return_state
        self._suspend_history = True
        try:
            self._restore_view_state(session, target)
            session.saved_view_state = target
        finally:
            self._suspend_history = False

        self.statusBar().showMessage(f"Returned to page {target.page}", 4000)

    @staticmethod
    def _normalize_equation_id(raw: str) -> str:
        parts = re.findall(r"\d+", raw)
        if not parts:
            raise ValueError("Invalid equation number")
        return ".".join(parts)

    @staticmethod
    def _build_fast_search_variants(eq_id: str) -> List[str]:
        dotted = ".".join(eq_id.split("."))
        return [f"({dotted})", dotted]

    @staticmethod
    def _build_search_variants(eq_id: str) -> List[str]:
        parts = eq_id.split(".")
        dotted = ".".join(parts)
        spaced = " ".join(parts)
        comma = ",".join(parts)
        dashed = "-".join(parts)
        mixed = " . ".join(parts)
        variants = [
            f"({dotted})",
            dotted,
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
        # Search for an equality sign anywhere on the same page as the
        # equation-number hit. Some PDFs encode a visual "=" as a private-use
        # math glyph (for example U+F03D), so search for both normal and
        # PDF-font encoded equality signs.
        equal_variants = ["=", "\uf03d", "＝", ""]
        for equal in equal_variants:
            try:
                if page.search_for(equal):
                    return True
            except Exception:
                continue

        # Last-resort text extraction check. This catches cases where search_for
        # cannot index the glyph even though get_text can expose it.
        try:
            text = page.get_text() or ""
            return any(equal in text for equal in equal_variants)
        except Exception:
            return False

    def _page_text_has_parenthesized_equation_id(self, page: fitz.Page, eq_id: str) -> bool:
        """Return True when the extracted text contains the equation number in parentheses.

        In some right-to-left PDFs, a visible equation number such as (5.3.11)
        is extracted as separate lines or with reversed parentheses, for example
        ")\n5.3.11\n(". PyMuPDF's search_for("(5.3.11)") may therefore fail even
        though the page text still contains the correct parenthesized equation number.
        """
        try:
            text = page.get_text() or ""
        except Exception:
            return False

        compact = re.sub(r"\s+", "", text)
        return f"({eq_id})" in compact or f"){eq_id}(" in compact

    def _toc_chapter_range_for_equation(
        self,
        eq_id: str,
        session: DocumentSession,
    ) -> Tuple[int, int, Optional[Tuple[int, ...]]]:
        """Return a chapter range from TOC bookmarks, if chapter bookmarks exist."""
        if not session.section_starts:
            return 1, len(session.doc), None

        try:
            chapter = int(eq_id.split(".")[0])
        except Exception:
            return 1, len(session.doc), None

        chapter_key = (chapter,)
        start_page = session.section_starts.get(chapter_key)

        # Some PDFs do not include explicit chapter bookmarks but do include
        # section bookmarks.  In that case, use the earliest bookmark inside the
        # requested chapter as a safe starting point.
        if start_page is None:
            pages_in_chapter = [
                page
                for section, page in session.section_starts.items()
                if section and section[0] == chapter
            ]
            if not pages_in_chapter:
                return 1, len(session.doc), None
            start_page = min(pages_in_chapter)

        next_chapter_pages = sorted(
            page
            for section, page in session.section_starts.items()
            if len(section) == 1 and section[0] > chapter
        )
        end_page = next_chapter_pages[0] - 1 if next_chapter_pages else len(session.doc)

        start_page = max(1, min(start_page, len(session.doc)))
        end_page = max(start_page, min(end_page, len(session.doc)))
        return start_page, end_page, chapter_key

    def _section_range_for_equation(
        self,
        eq_id: str,
        session: DocumentSession,
        chapter_start_page: int,
        chapter_end_page: int,
    ) -> Tuple[int, int, Optional[Tuple[int, ...]]]:
        """Refine the search range using PDF bookmarks/TOC section numbers.

        Handles TOCs with non-numbered "Part" nodes, numbered chapter nodes
        such as "1 The Celestial Sphere", and deeper section nodes such as
        "1.1 The Greek Tradition".
        """
        if not session.section_starts:
            return chapter_start_page, chapter_end_page, None

        parts = tuple(int(part) for part in eq_id.split("."))
        best_prefix: Optional[Tuple[int, ...]] = None

        # Prefer the most specific bookmark number that is a prefix of the
        # equation number, but allow chapter-level prefixes too.
        for length in range(len(parts), 0, -1):
            prefix = parts[:length]
            if prefix in session.section_starts:
                best_prefix = prefix
                break

        if best_prefix is None:
            return chapter_start_page, chapter_end_page, None

        start_page = max(chapter_start_page, session.section_starts[best_prefix])
        chapter = parts[0]

        if len(best_prefix) == 1:
            later_pages = sorted(
                page
                for section, page in session.section_starts.items()
                if len(section) == 1 and section[0] > chapter and page > start_page
            )
        else:
            later_pages = sorted(
                page
                for section, page in session.section_starts.items()
                if (
                    len(section) >= len(best_prefix)
                    and section[0] == chapter
                    and section[: len(best_prefix)] != best_prefix
                    and page > start_page
                )
            )

        end_page = min(chapter_end_page, later_pages[0] - 1) if later_pages else chapter_end_page
        end_page = max(start_page, end_page)
        return start_page, end_page, best_prefix

    def _search_debug_text(
        self,
        eq_id: str,
        session: DocumentSession,
        start_page: int,
        end_page: int,
        section_prefix: Optional[Tuple[int, ...]] = None,
    ) -> str:
        total_pages = len(session.doc)
        chapter_starts = session.chapter_starts or {}
        chapter_label = "?"
        reason = ""

        try:
            chapter = int(eq_id.split(".")[0])
            chapter_label = str(chapter)
            if section_prefix is not None:
                section = ".".join(str(x) for x in section_prefix)
                section_page = session.section_starts.get(section_prefix)
                reason = f"bookmark section {section} starts at page {section_page}"
            elif chapter_starts:
                if chapter in chapter_starts:
                    reason = f"chapter {chapter} starts at page {chapter_starts[chapter]} from {session.chapter_starts_source or 'a.txt'}"
                else:
                    reason = f"chapter {chapter} is not in a.txt; falling back to full document"
            else:
                reason = "no chapter starts were loaded from a.txt; falling back to full document"
        except Exception:
            reason = "could not detect chapter from equation; falling back to full document"

        return (
            f"Equation search: eq={eq_id}, chapter={chapter_label}, "
            f"range={start_page}-{end_page} of {total_pages} pages ({reason})"
        )

    def _debug_equation_search(self, message: str) -> None:
        # Print is important because the status bar cannot repaint while the PDF
        # search loop is running. Run the app from Terminal to see this live.
        print(f"[PDF Reader] {message}", flush=True)
        self.statusBar().showMessage(message, 12000)
        QApplication.processEvents()

    def _ensure_equation_format_for_query(self, session: DocumentSession, eq_id: str) -> None:
        if session.equation_format.get("source") in {
            "equation_index_scan",
            "manual_equation_samples",
            "manual_and_equation_index_scan",
        }:
            return

        part_count = len(eq_id.split("."))
        if part_count == 2:
            mode = EQUATION_MODE_CHAPTER
        elif part_count >= 3:
            mode = EQUATION_MODE_SECTION
        else:
            return

        existing_mode = session.equation_format.get("numbering_mode")
        if existing_mode == mode and session.equation_numbering_mode == mode:
            return

        session.equation_numbering_mode = mode
        session.equation_cache.clear()
        session.equation_format = {
            "version": EQUATION_FORMAT_VERSION,
            "numbering_mode": mode,
            "part_counts": {str(part_count): 1},
            "patterns": dict(session.equation_index_patterns),
            "examples": {str(part_count): eq_id},
            "source": "query_shape",
            "sample_count": 1,
        }
        self._save_session_state_now()

    def _search_equation_with_variants(
        self,
        session: DocumentSession,
        eq_id: str,
        variants: List[str],
        start_page: int,
        end_page: int,
        phase_name: str,
    ) -> Tuple[Optional[int], Optional[int], int]:
        fallback = None
        pages_checked = 0

        self._debug_equation_search(
            f"Equation search: {phase_name}; variants={variants}; pages={start_page}-{end_page}"
        )

        for page_index in range(start_page - 1, end_page):
            pages_checked += 1
            page = session.doc.load_page(page_index)

            # First try PyMuPDF's native text search. This is fast when the PDF
            # text layer is clean.
            for variant in variants:
                hits = page.search_for(variant)
                if not hits:
                    continue
                if fallback is None:
                    fallback = page_index + 1
                    self._debug_equation_search(
                        f"Equation search: first text hit for {eq_id} on page {fallback}; checking page-wide '='"
                    )
                for hit in hits:
                    if self._has_equal_nearby(page, hit):
                        found_page = page_index + 1
                        self._debug_equation_search(
                            f"Equation search: found {eq_id} on page {found_page} during {phase_name} after checking {pages_checked} pages"
                        )
                        return found_page, fallback, pages_checked

            # Fallback for RTL/math PDFs: search_for("(5.3.11)") can fail because
            # the extracted text may be ")\n5.3.11\n(". In that case, look at the
            # extracted page text directly and still require a page-wide "=".
            if self._page_text_has_parenthesized_equation_id(page, eq_id):
                if fallback is None:
                    fallback = page_index + 1
                    self._debug_equation_search(
                        f"Equation search: extracted-text hit for {eq_id} on page {fallback}; checking page-wide '='"
                    )
                if self._has_equal_nearby(page, page.rect):
                    found_page = page_index + 1
                    self._debug_equation_search(
                        f"Equation search: found {eq_id} on page {found_page} by extracted-text fallback during {phase_name} after checking {pages_checked} pages"
                    )
                    return found_page, fallback, pages_checked

        return None, fallback, pages_checked

    def _find_equation_page(self, session: DocumentSession, user_eq: str) -> Optional[int]:
        eq_id = self._normalize_equation_id(user_eq)
        self._ensure_equation_format_for_query(session, eq_id)
        if eq_id in session.equation_cache:
            page = session.equation_cache[eq_id]
            self._debug_equation_search(f"Equation search: cache hit for {eq_id}; page {page}")
            return page

        # Always refresh a.txt before a new uncached search. This is important
        # because chapter-equation mode relies on a.txt rather than TOC sections,
        # and it also lets the user fix a.txt without restarting the app.
        fresh_starts, fresh_source = self._load_chapter_starts_for_pdf(session.path)
        if fresh_starts != session.chapter_starts or fresh_source != session.chapter_starts_source:
            session.chapter_starts = fresh_starts
            session.chapter_starts_source = fresh_source
            session.equation_cache.clear()

        chapter_start, chapter_end = get_search_page_range(eq_id, session.chapter_starts, len(session.doc))

        # In section-aware mode only, use PDF bookmarks as a fallback when no
        # usable a.txt range exists. In chapter-equation mode, never interpret
        # (a.b) as section a.b; the range is strictly from a.txt.
        toc_chapter_prefix = None
        if session.equation_numbering_mode != EQUATION_MODE_CHAPTER:
            toc_start, toc_end, toc_chapter_prefix = self._toc_chapter_range_for_equation(eq_id, session)
            if (chapter_start, chapter_end) == (1, len(session.doc)) and toc_chapter_prefix is not None:
                chapter_start, chapter_end = toc_start, toc_end

        ranges: List[Tuple[int, int, Optional[Tuple[int, ...]], str]] = []

        def add_range(a: int, b: int, prefix: Optional[Tuple[int, ...]], label: str) -> None:
            a = max(1, min(a, len(session.doc)))
            b = max(a, min(b, len(session.doc)))
            key = (a, b)
            if not any((r[0], r[1]) == key for r in ranges):
                ranges.append((a, b, prefix, label))

        if session.equation_numbering_mode == EQUATION_MODE_CHAPTER:
            # Chapter-equation books number equations as (a.b), where a is the
            # chapter and b is the equation number inside that chapter. In this
            # mode, do not interpret (1.2) as section 1.2; search only the
            # chapter span defined in a.txt.
            if not session.chapter_starts:
                self._debug_equation_search(
                    "Chapter-equation mode needs a.txt chapter starts. "
                    "Expected lines such as '1 - 14'. Falling back to full-document search."
                )
            else:
                try:
                    chapter_num = int(eq_id.split(".")[0])
                except Exception:
                    chapter_num = None
                if chapter_num is not None and chapter_num not in session.chapter_starts:
                    self._debug_equation_search(
                        f"Chapter {chapter_num} is not listed in a.txt. Falling back to full-document search."
                    )
            add_range(chapter_start, chapter_end, None, "a.txt chapter-equation range")
        else:
            start_page, end_page, section_prefix = self._section_range_for_equation(
                eq_id,
                session,
                chapter_start,
                chapter_end,
            )

            # Try the refined range first, then the whole chapter range if the
            # bookmark/TOC refinement was too narrow, and finally the full document.
            # This keeps normal searches fast but avoids false "not found" results
            # when PDF bookmarks or a.txt page offsets are slightly off.
            add_range(start_page, end_page, section_prefix, "refined range")
            add_range(chapter_start, chapter_end, None, "chapter range")
            add_range(1, len(session.doc), None, "full document fallback")

        fast_variants = self._build_fast_search_variants(eq_id)
        slow_variants = [v for v in self._build_search_variants(eq_id) if v not in fast_variants]

        fallback = None
        total_checked = 0

        for range_start, range_end, prefix, label in ranges:
            self._debug_equation_search(
                f"Equation search: trying {label}; "
                + self._search_debug_text(eq_id, session, range_start, range_end, prefix)
            )

            found, range_fallback, fast_checked = self._search_equation_with_variants(
                session,
                eq_id,
                fast_variants,
                range_start,
                range_end,
                f"fast exact search ({label})",
            )
            total_checked += fast_checked
            if range_fallback is not None and fallback is None:
                fallback = range_fallback
            if found is not None:
                session.equation_cache[eq_id] = found
                return found

            found, range_fallback, slow_checked = self._search_equation_with_variants(
                session,
                eq_id,
                slow_variants,
                range_start,
                range_end,
                f"loose fallback search ({label})",
            )
            total_checked += slow_checked
            if range_fallback is not None and fallback is None:
                fallback = range_fallback
            if found is not None:
                session.equation_cache[eq_id] = found
                return found

        if fallback is not None:
            self._debug_equation_search(
                f"Equation search: no page-wide '=' found; using first text hit for {eq_id} on page {fallback} after checking {total_checked} page-passes"
            )
            session.equation_cache[eq_id] = fallback
        else:
            self._debug_equation_search(
                f"Equation search: {eq_id} not found after checking {total_checked} page-passes"
            )
        return fallback

    def _jump_to_equation_location(
        self,
        session: DocumentSession,
        location: EquationLocation,
        push_history: bool = True,
    ) -> None:
        self._show_page_for_session(session, location.page, push_history=push_history)
        self.view.scroll_to_page(location.page, location.y * RENDER_SCALE)

    def _find_equation_location_from_index(
        self,
        session: DocumentSession,
        user_eq: str,
        show_missing_message: bool = True,
    ) -> Optional[EquationLocation]:
        eq_id = self._normalize_equation_id(user_eq)
        if session.equation_lookup_mode == EQUATION_LOOKUP_PATTERN_INDEX:
            location = session.equation_pattern_index.get(eq_id)
            label = "learned-pattern"
        else:
            location = session.equation_rough_index.get(eq_id) or session.equation_index.get(eq_id)
            label = "rough first-scan"
        if location is None and show_missing_message:
            self.statusBar().showMessage(
                f"Equation ({eq_id}) is not in the {label} equation index. Press Ctrl+D to rescan, or Ctrl+G for another mode.",
                7000,
            )
        return location

    def _find_equation_destination(
        self,
        session: DocumentSession,
        user_eq: str,
    ) -> Optional[EquationLocation]:
        if session.equation_lookup_mode in {EQUATION_LOOKUP_ROUGH_INDEX, EQUATION_LOOKUP_PATTERN_INDEX}:
            location = self._find_equation_location_from_index(
                session,
                user_eq,
                show_missing_message=False,
            )
            if location is not None:
                return location
            eq_id = self._normalize_equation_id(user_eq)
            self._debug_equation_search(
                f"Equation search: {eq_id} is not in the selected equation index; falling back to text scan"
            )

        page = self._find_equation_page(session, user_eq)
        if page is None:
            return None
        return EquationLocation(page=page, y=0.0)

    def _document_signature(self, session: DocumentSession) -> dict:
        path = Path(session.path)
        try:
            stat = path.stat()
            size = int(stat.st_size)
            mtime_ns = int(stat.st_mtime_ns)
        except Exception:
            size = 0
            mtime_ns = 0

        first_rect = ""
        try:
            rect = session.doc.load_page(0).rect
            first_rect = f"{rect.width:.2f}x{rect.height:.2f}"
        except Exception:
            pass

        metadata = {}
        try:
            metadata = dict(session.doc.metadata or {})
        except Exception:
            metadata = {}

        return {
            "path": str(path.resolve()) if path.exists() else session.path,
            "size": size,
            "mtime_ns": mtime_ns,
            "pages": len(session.doc),
            "first_page_rect": first_rect,
            "format": metadata.get("format") or "PDF",
            "title": metadata.get("title") or "",
            "equation_index_algorithm": EQUATION_INDEX_ALGORITHM_VERSION,
        }

    @staticmethod
    def _line_equation_ids(text: str) -> List[str]:
        ids: List[str] = []
        for match in re.finditer(r"\(?\d+(?:[.\s,\-]+\d+){1,2}\)?", text):
            try:
                eq_id = PdfReaderWindow._normalize_equation_id(match.group(0))
            except ValueError:
                continue
            parts = eq_id.split(".")
            if len(parts) in {2, 3} and eq_id not in ids:
                ids.append(eq_id)
        return ids

    @staticmethod
    def _equation_matches_in_text(text: str) -> List[Tuple[str, str, int, int]]:
        matches: List[Tuple[str, str, int, int]] = []
        for match in re.finditer(r"\(?\d+(?:[.\s,\-]+\d+){1,2}\)?", text):
            raw = match.group(0).strip()
            if raw.startswith("(") and not raw.endswith(")"):
                continue
            if "." not in raw:
                continue
            try:
                eq_id = PdfReaderWindow._normalize_equation_id(raw)
            except ValueError:
                continue
            if len(eq_id.split(".")) in {2, 3}:
                matches.append((eq_id, raw, match.start(), match.end()))
        return matches

    @staticmethod
    def _format_template(raw_text: str) -> str:
        return re.sub(r"\d+", "#", raw_text.strip())

    @staticmethod
    def _regex_from_template(template: str) -> re.Pattern:
        parts = re.split(r"(#+)", template)
        regex = ""
        for part in parts:
            if not part:
                continue
            if set(part) == {"#"}:
                regex += r"(\d+)"
            else:
                regex += re.escape(part)
        return re.compile(regex)

    @staticmethod
    def _related_template(template: str, target_part_count: int) -> Optional[str]:
        tokens = re.split(r"(#+)", template)
        digit_indexes = [index for index, token in enumerate(tokens) if token and set(token) == {"#"}]
        if len(digit_indexes) < 2:
            return None

        prefix = "".join(tokens[: digit_indexes[0]])
        separator = "".join(tokens[digit_indexes[0] + 1 : digit_indexes[1]]) or "."
        suffix = "".join(tokens[digit_indexes[-1] + 1 :])
        if target_part_count not in {2, 3}:
            return None
        return prefix + separator.join("#" for _ in range(target_part_count)) + suffix

    @staticmethod
    def _learn_equation_format_patterns(candidates: List[EquationCandidate]) -> Dict[int, str]:
        if not candidates:
            return {}

        counts: Dict[Tuple[int, str], int] = {}
        quality: Dict[Tuple[int, str], int] = {}
        for candidate in candidates:
            part_count = len(candidate.eq_id.split("."))
            if part_count not in {2, 3}:
                continue
            template = PdfReaderWindow._format_template(candidate.raw_text)
            key = (part_count, template)
            counts[key] = counts.get(key, 0) + 1
            raw = candidate.raw_text.strip()
            if raw.startswith("(") and raw.endswith(")"):
                template_quality = 2
            elif raw.startswith("(") or raw.endswith(")"):
                template_quality = 1
            else:
                template_quality = 0
            quality[key] = max(quality.get(key, 0), template_quality)

        patterns: Dict[int, str] = {}
        for (part_count, template), _count in sorted(
            counts.items(),
            key=lambda item: (-item[1], -quality.get(item[0], 0), item[0][0], item[0][1]),
        ):
            patterns.setdefault(part_count, template)

        part_totals: Dict[int, int] = {}
        for part_count, _template in counts:
            part_totals[part_count] = part_totals.get(part_count, 0) + counts[(part_count, _template)]

        if (
            2 in patterns
            and 3 in patterns
            and "(" not in patterns[2]
            and ")" not in patterns[2]
            and "(" not in patterns[3]
            and ")" not in patterns[3]
            and part_totals.get(3, 0) >= part_totals.get(2, 0)
        ):
            patterns.pop(2, None)

        if 3 in patterns and quality.get((2, patterns.get(2, "")), 0) == 0 and quality.get((3, patterns[3]), 0) > 0:
            related = PdfReaderWindow._related_template(patterns[3], 2)
            if related:
                patterns[2] = related
        if 2 in patterns and 3 not in patterns and quality.get((2, patterns[2]), 0) > 0:
            related = PdfReaderWindow._related_template(patterns[2], 3)
            if related:
                patterns[3] = related
        if 3 in patterns and 2 not in patterns and quality.get((3, patterns[3]), 0) > 0:
            related = PdfReaderWindow._related_template(patterns[3], 2)
            if related:
                patterns[2] = related
        return patterns

    @staticmethod
    def _infer_equation_numbering_mode(candidates: List[EquationCandidate]) -> str:
        part_counts: Dict[int, int] = {}
        for candidate in candidates:
            part_count = len(candidate.eq_id.split("."))
            if part_count in {2, 3}:
                part_counts[part_count] = part_counts.get(part_count, 0) + 1

        if part_counts.get(2, 0) > part_counts.get(3, 0):
            return EQUATION_MODE_CHAPTER
        return EQUATION_MODE_SECTION

    @staticmethod
    def _build_equation_format_metadata(
        candidates: List[EquationCandidate],
        patterns: Dict[int, str],
    ) -> Dict[str, object]:
        part_counts: Dict[str, int] = {}
        examples: Dict[str, str] = {}
        for candidate in candidates:
            part_count = len(candidate.eq_id.split("."))
            if part_count not in {2, 3}:
                continue
            key = str(part_count)
            part_counts[key] = part_counts.get(key, 0) + 1
            examples.setdefault(key, candidate.raw_text)

        mode = PdfReaderWindow._infer_equation_numbering_mode(candidates)
        return {
            "version": EQUATION_FORMAT_VERSION,
            "numbering_mode": mode,
            "part_counts": part_counts,
            "patterns": {
                str(part_count): pattern
                for part_count, pattern in sorted(patterns.items())
            },
            "examples": examples,
            "source": "equation_index_scan",
            "sample_count": sum(part_counts.values()),
        }

    @staticmethod
    def _line_has_equal(words: List[tuple], y0: float, y1: float) -> bool:
        equal_variants = ("=", "\uf03d", "＝", "")
        center = (y0 + y1) / 2
        for word in words:
            text = str(word[4])
            if any(equal in text for equal in equal_variants):
                wy0 = float(word[1])
                wy1 = float(word[3])
                if abs(((wy0 + wy1) / 2) - center) <= 35:
                    return True
        return False

    @staticmethod
    def _span_is_bold(span: dict) -> bool:
        flags = int(span.get("flags") or 0)
        font = str(span.get("font") or "").lower()
        return bool(flags & 16) or "bold" in font

    @staticmethod
    def _bold_number_locations_on_page(page: fitz.Page) -> List[Tuple[str, fitz.Rect]]:
        try:
            text_dict = page.get_text("rawdict")
        except Exception:
            return []

        locations: List[Tuple[str, fitz.Rect]] = []
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                chars: List[Tuple[str, fitz.Rect, bool]] = []
                for span in line.get("spans", []):
                    is_bold = PdfReaderWindow._span_is_bold(span)
                    for char in span.get("chars", []):
                        text = str(char.get("c", ""))
                        try:
                            rect = fitz.Rect(char["bbox"])
                        except Exception:
                            continue
                        chars.append((text, rect, is_bold))

                if not chars:
                    continue

                line_text = "".join(char[0] for char in chars)
                for match in re.finditer(r"\d+(?:[.\s,\-]+\d+){1,2}", line_text):
                    matched_chars = chars[match.start() : match.end()]
                    digit_chars = [char for char in matched_chars if char[0].isdigit()]
                    if not digit_chars or not all(char[2] for char in digit_chars):
                        continue
                    try:
                        eq_id = PdfReaderWindow._normalize_equation_id(match.group(0))
                    except ValueError:
                        continue
                    if len(eq_id.split(".")) not in {2, 3}:
                        continue

                    rect = matched_chars[0][1]
                    for _text, char_rect, _is_bold in matched_chars[1:]:
                        rect |= char_rect
                    locations.append((eq_id, rect))
        return locations

    @staticmethod
    def _line_has_bold_number(
        bold_number_locations: List[Tuple[str, fitz.Rect]],
        eq_id: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> bool:
        if not bold_number_locations:
            return False

        line_rect = fitz.Rect(x0 - 2.0, y0 - 2.0, x1 + 2.0, y1 + 2.0)
        return any(
            bold_id == eq_id and line_rect.intersects(rect)
            for bold_id, rect in bold_number_locations
        )

    @staticmethod
    def _word_text_is_math_fragment(text: str) -> bool:
        if not text:
            return False
        if any("\uf000" <= ch <= "\uf8ff" for ch in text):
            return True
        return any(ch in text for ch in "=+-−*/√∑∫≈≤≥<>[]{}")

    @staticmethod
    def _has_adjacent_parenthesis_token(
        words: List[tuple],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> bool:
        center_y = (y0 + y1) / 2.0
        found_left = False
        found_right = False
        for word in words:
            text = str(word[4]).strip()
            if text not in {"(", ")"}:
                continue
            wx0 = float(word[0])
            wy0 = float(word[1])
            wx1 = float(word[2])
            wy1 = float(word[3])
            if abs(((wy0 + wy1) / 2.0) - center_y) > 5.0:
                continue
            if 0 <= x0 - wx1 <= 8.0:
                found_left = True
            if 0 <= wx0 - x1 <= 8.0:
                found_right = True
        return found_left or found_right

    @staticmethod
    def _has_nearby_math_fragment(
        words: List[tuple],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> bool:
        center_y = (y0 + y1) / 2.0
        for word in words:
            text = str(word[4]).strip()
            if not PdfReaderWindow._word_text_is_math_fragment(text):
                continue
            wx0 = float(word[0])
            wy0 = float(word[1])
            wx1 = float(word[2])
            wy1 = float(word[3])
            dx = 0.0 if wx0 <= x0 <= wx1 or x0 <= wx0 <= x1 else min(abs(wx0 - x1), abs(x0 - wx1))
            dy = abs(((wy0 + wy1) / 2.0) - center_y)
            if dx <= 260.0 and dy <= 16.0:
                return True
        return False

    @staticmethod
    def _has_display_equation_number_context(
        text: str,
        line_words: List[tuple],
        words: List[tuple],
        page_width: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> bool:
        if len(line_words) > 2:
            return False
        if len(re.findall(r"[A-Za-z\u0590-\u05ff]", text)) > 0:
            return False
        left_aligned = x0 <= page_width * 0.32 and x1 <= page_width * 0.58
        right_aligned = x1 >= page_width * 0.68
        if not (left_aligned or right_aligned):
            return False
        return (
            PdfReaderWindow._has_adjacent_parenthesis_token(words, x0, y0, x1, y1)
            or PdfReaderWindow._has_nearby_math_fragment(words, x0, y0, x1, y1)
        )

    def _equation_line_entries(self, page: fitz.Page) -> Tuple[List[Tuple[float, float, float, float, str, List[tuple]]], List[tuple]]:
        try:
            words = page.get_text("words", sort=True)
        except TypeError:
            words = page.get_text("words")
        except Exception:
            return [], []

        lines: Dict[Tuple[int, int], List[tuple]] = {}
        for word in words:
            if len(word) < 7:
                continue
            key = (int(word[5]), int(word[6]))
            lines.setdefault(key, []).append(word)

        line_entries: List[Tuple[float, float, float, float, str, List[tuple]]] = []
        for line_words in lines.values():
            line_words = sorted(line_words, key=lambda w: (float(w[0]), float(w[1])))
            x0 = min(float(w[0]) for w in line_words)
            y0 = min(float(w[1]) for w in line_words)
            x1 = max(float(w[2]) for w in line_words)
            y1 = max(float(w[3]) for w in line_words)
            text = " ".join(str(w[4]) for w in line_words)
            line_entries.append((x0, y0, x1, y1, text, line_words))

        line_entries.sort(key=lambda item: (item[1], item[0]))
        return line_entries, words

    def _rough_equation_candidates_on_page(self, page: fitz.Page, page_number: int) -> List[EquationCandidate]:
        line_entries, words = self._equation_line_entries(page)
        candidates: List[EquationCandidate] = []
        page_width = float(page.rect.width)
        bold_number_locations = self._bold_number_locations_on_page(page)

        for index, (x0, y0, x1, y1, text, line_words) in enumerate(line_entries):
            matches = self._equation_matches_in_text(text)
            if not matches:
                continue
            has_equal = self._line_has_equal(words, y0, y1)
            matches = [
                match
                for match in matches
                if not self._line_has_bold_number(bold_number_locations, match[0], x0, y0, x1, y1)
            ]
            if not matches:
                continue

            prev_gap = 999.0
            next_gap = 999.0
            alphabetic_chars = len(re.findall(r"[A-Za-z\u0590-\u05ff]", text))
            right_aligned = x1 >= page_width * 0.72
            left_aligned = x0 <= page_width * 0.32 and x1 <= page_width * 0.58
            display_number_context = self._has_display_equation_number_context(
                text,
                line_words,
                words,
                page_width,
                x0,
                y0,
                x1,
                y1,
            )

            cluster_y0 = y0
            cluster_y1 = y1
            for previous in reversed(line_entries[:index]):
                gap = cluster_y0 - previous[3]
                if gap >= 10 or gap < -1:
                    break
                if has_equal or self._line_has_equal(words, previous[1], previous[3]):
                    cluster_y0 = min(cluster_y0, previous[1])
                    continue
                break
            for following in line_entries[index + 1 :]:
                gap = following[1] - cluster_y1
                if gap >= 10 or gap < -1:
                    break
                if has_equal or self._line_has_equal(words, following[1], following[3]):
                    cluster_y1 = max(cluster_y1, following[3])
                    continue
                break

            for previous in reversed(line_entries[:index]):
                if previous[3] < cluster_y0 - 1:
                    prev_gap = max(0.0, cluster_y0 - previous[3])
                    break
            for following in line_entries[index + 1 :]:
                if following[1] > cluster_y1 + 1:
                    next_gap = max(0.0, following[1] - cluster_y1)
                    break

            isolated = prev_gap >= 10 and next_gap >= 10
            compact_equation_line = alphabetic_chars <= 12 and len(line_words) <= 18
            likely_display_equation = (
                (right_aligned or left_aligned)
                and isolated
                and compact_equation_line
                and (has_equal or alphabetic_chars <= 4 or len(line_words) <= 8)
            ) or display_number_context
            if not likely_display_equation:
                continue

            target_y = max(0.0, cluster_y0 - 40.0)
            for eq_id, raw_text, _match_start, _match_end in matches:
                candidates.append(
                    EquationCandidate(
                        eq_id=eq_id,
                        raw_text=raw_text,
                        location=EquationLocation(page=page_number, y=target_y),
                    )
                )

        return candidates

    def _pattern_equation_candidates_on_page(
        self,
        page: fitz.Page,
        page_number: int,
        patterns: Dict[int, str],
    ) -> List[EquationCandidate]:
        if not patterns:
            return []

        line_entries, words = self._equation_line_entries(page)
        candidates: List[EquationCandidate] = []
        page_width = float(page.rect.width)
        bold_number_locations = self._bold_number_locations_on_page(page)
        compiled = {part_count: self._regex_from_template(pattern) for part_count, pattern in patterns.items()}

        for index, (x0, y0, x1, y1, text, line_words) in enumerate(line_entries):
            right_aligned = x1 >= page_width * 0.68
            left_aligned = x0 <= page_width * 0.32 and x1 <= page_width * 0.58
            if not (right_aligned or left_aligned):
                continue

            alphabetic_chars = len(re.findall(r"[A-Za-z\u0590-\u05ff]", text))
            if alphabetic_chars > 12 or len(line_words) > 18:
                continue

            has_equal = self._line_has_equal(words, y0, y1)
            display_number_context = self._has_display_equation_number_context(
                text,
                line_words,
                words,
                page_width,
                x0,
                y0,
                x1,
                y1,
            )
            prev_gap = 999.0
            next_gap = 999.0
            for previous in reversed(line_entries[:index]):
                if previous[3] < y0 - 1:
                    prev_gap = max(0.0, y0 - previous[3])
                    break
            for following in line_entries[index + 1 :]:
                if following[1] > y1 + 1:
                    next_gap = max(0.0, following[1] - y1)
                    break
            isolated = prev_gap >= 10 and next_gap >= 10
            bare_number_line = any(pattern and "(" not in pattern and ")" not in pattern for pattern in patterns.values())
            if bare_number_line and (alphabetic_chars > 4 or len(line_words) > 6):
                continue
            if bare_number_line and not has_equal and not isolated and not display_number_context:
                continue
            if not has_equal and alphabetic_chars > 4 and len(line_words) > 8:
                continue

            for part_count, pattern_re in compiled.items():
                for match in pattern_re.finditer(text):
                    raw_text = match.group(0).strip()
                    parts = match.groups()
                    if len(parts) != part_count:
                        continue
                    eq_id = ".".join(parts)
                    if self._line_has_bold_number(bold_number_locations, eq_id, x0, y0, x1, y1):
                        continue
                    candidates.append(
                        EquationCandidate(
                            eq_id=eq_id,
                            raw_text=raw_text,
                            location=EquationLocation(page=page_number, y=max(0.0, y0 - 40.0)),
                        )
                    )

        return candidates

    def _dotted_equation_candidates_on_page(
        self,
        page: fitz.Page,
        page_number: int,
    ) -> List[EquationCandidate]:
        line_entries, words = self._equation_line_entries(page)
        candidates: List[EquationCandidate] = []
        page_width = float(page.rect.width)
        bold_number_locations = self._bold_number_locations_on_page(page)

        for index, (x0, y0, x1, y1, text, line_words) in enumerate(line_entries):
            if not (x0 <= page_width * 0.32 and x1 <= page_width * 0.58):
                continue
            if len(line_words) > 6:
                continue
            if len(re.findall(r"[A-Za-z\u0590-\u05ff]", text)) > 4:
                continue

            has_equal = self._line_has_equal(words, y0, y1)
            display_number_context = self._has_display_equation_number_context(
                text,
                line_words,
                words,
                page_width,
                x0,
                y0,
                x1,
                y1,
            )
            prev_gap = 999.0
            next_gap = 999.0
            for previous in reversed(line_entries[:index]):
                if previous[3] < y0 - 1:
                    prev_gap = max(0.0, y0 - previous[3])
                    break
            for following in line_entries[index + 1 :]:
                if following[1] > y1 + 1:
                    next_gap = max(0.0, following[1] - y1)
                    break
            if not has_equal and not (prev_gap >= 10 and next_gap >= 10) and not display_number_context:
                continue

            for eq_id, raw_text, _start, _end in self._equation_matches_in_text(text):
                if len(eq_id.split(".")) != 3:
                    continue
                if self._line_has_bold_number(bold_number_locations, eq_id, x0, y0, x1, y1):
                    continue
                candidates.append(
                    EquationCandidate(
                        eq_id=eq_id,
                        raw_text=raw_text,
                        location=EquationLocation(page=page_number, y=max(0.0, y0 - 40.0)),
                    )
                )

        return candidates

    def _equation_locations_on_page(self, page: fitz.Page, page_number: int) -> Dict[str, EquationLocation]:
        return {
            candidate.eq_id: candidate.location
            for candidate in self._rough_equation_candidates_on_page(page, page_number)
        }

    def _available_learned_equation_patterns(self, session: DocumentSession) -> Dict[int, str]:
        patterns = dict(session.equation_index_patterns)
        if patterns:
            return patterns
        if session.manual_equation_samples:
            patterns = self._learn_equation_format_patterns(list(session.manual_equation_samples))
            if patterns:
                session.equation_index_patterns = patterns
                session.equation_format = self._build_equation_format_metadata(
                    session.manual_equation_samples,
                    patterns,
                )
                session.equation_format["source"] = "manual_equation_samples"
                self._save_session_state_now()
        return patterns

    def choose_equation_index_scan_mode(self) -> None:
        session = self.current_session()
        if session is None:
            return

        box = QMessageBox(self)
        box.setWindowTitle("Scan Equation Index")
        box.setText("Choose how to scan equations in this document.")
        automatic_button = box.addButton("Automatic format learn", QMessageBox.AcceptRole)
        learned_button = box.addButton("Use existing learned format", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        existing_patterns = self._available_learned_equation_patterns(session)
        has_patterns = bool(existing_patterns)
        learned_button.setEnabled(has_patterns)
        if not has_patterns:
            box.setInformativeText("No learned format is available yet. Select equation boxes with long-click + Ctrl+R first, or use automatic learning.")
        else:
            patterns = ", ".join(
                f"{part_count}: {pattern}"
                for part_count, pattern in sorted(existing_patterns.items())
            )
            box.setInformativeText(f"Existing learned format: {patterns}")

        box.exec()
        clicked = box.clickedButton()
        if clicked is automatic_button:
            self.scan_equation_index_for_current_document()
        elif clicked is learned_button:
            self.scan_equation_index_with_existing_format()

    def scan_equation_index_with_existing_format(self) -> None:
        session = self.current_session()
        if session is None:
            return
        patterns = self._available_learned_equation_patterns(session)
        if not patterns:
            self.statusBar().showMessage(
                "No learned equation format is available. Select boxes and press Ctrl+R, or use automatic format learning.",
                8000,
            )
            return
        session.equation_lookup_mode = EQUATION_LOOKUP_PATTERN_INDEX
        self._save_session_state_now()
        self._start_manual_pattern_index_scan(session, patterns)

    def scan_equation_index_for_current_document(self) -> None:
        session = self.current_session()
        if session is None:
            return

        self._manual_pattern_scan_timer.stop()
        previous_patterns = dict(session.equation_index_patterns)
        previous_pattern_index = dict(session.equation_pattern_index)
        previous_format = dict(session.equation_format)
        self._debug_equation_search("Equation rough index scan started")
        rough_index: Dict[str, EquationLocation] = {}
        rough_candidates: List[EquationCandidate] = []
        for page_number in range(1, len(session.doc) + 1):
            if page_number == 1 or page_number % 10 == 0:
                self._debug_equation_search(
                    f"Equation rough index scan: page {page_number} / {len(session.doc)}"
                )
            try:
                page = session.doc.load_page(page_number - 1)
                candidates = self._rough_equation_candidates_on_page(page, page_number)
            except Exception as exc:
                print(f"[PDF Reader] Equation rough index scan failed on page {page_number}: {exc}", flush=True)
                continue
            rough_candidates.extend(candidates)
            for candidate in candidates:
                rough_index.setdefault(candidate.eq_id, candidate.location)

        learning_candidates = rough_candidates + list(session.manual_equation_samples)
        patterns, pattern_index = self._scan_learned_equation_patterns(session, learning_candidates)
        if not pattern_index and session.manual_equation_samples:
            manual_patterns = self._learn_equation_format_patterns(list(session.manual_equation_samples))
            if manual_patterns and manual_patterns != patterns:
                self._debug_equation_search(
                    "Automatic learned-pattern scan found no matches; trying manual learned format"
                )
                manual_pattern_index: Dict[str, EquationLocation] = {}
                for page_number in range(1, len(session.doc) + 1):
                    try:
                        page = session.doc.load_page(page_number - 1)
                        candidates = self._pattern_equation_candidates_on_page(page, page_number, manual_patterns)
                    except Exception as exc:
                        print(f"[PDF Reader] Manual-format fallback scan failed on page {page_number}: {exc}", flush=True)
                        continue
                    for candidate in candidates:
                        manual_pattern_index.setdefault(candidate.eq_id, candidate.location)
                if manual_pattern_index:
                    patterns = manual_patterns
                    pattern_index = manual_pattern_index

        if not pattern_index and previous_patterns:
            patterns = previous_patterns
            pattern_index = previous_pattern_index
            if pattern_index:
                self._debug_equation_search(
                    f"Automatic scan found no learned-pattern matches; kept existing learned index ({len(pattern_index)} equations)"
                )
        if not pattern_index and rough_index:
            dotted_index: Dict[str, EquationLocation] = {}
            for page_number in range(1, len(session.doc) + 1):
                try:
                    page = session.doc.load_page(page_number - 1)
                    candidates = self._dotted_equation_candidates_on_page(page, page_number)
                except Exception as exc:
                    print(f"[PDF Reader] Dotted equation fallback scan failed on page {page_number}: {exc}", flush=True)
                    continue
                for candidate in candidates:
                    dotted_index.setdefault(candidate.eq_id, candidate.location)
            if dotted_index:
                pattern_index = dotted_index
                patterns = {3: "#.#.#"}
                self._debug_equation_search(
                    f"Learned-pattern scan recovered {len(pattern_index)} dotted equation labels"
                )

        if not pattern_index and rough_index:
            if not patterns:
                patterns = self._learn_equation_format_patterns(rough_candidates)
            pattern_index = dict(rough_index)
            self._debug_equation_search(
                f"Learned-pattern scan found no matches; using rough equation hits as the learned index ({len(pattern_index)} equations)"
            )
        elif pattern_index and rough_index:
            before_merge = len(pattern_index)
            for eq_id, location in rough_index.items():
                pattern_index.setdefault(eq_id, location)
            if len(pattern_index) > before_merge:
                self._debug_equation_search(
                    f"Equation index scan added {len(pattern_index) - before_merge} rough display-label matches to the learned-pattern index"
                )
        if session.manual_equation_samples:
            self._debug_equation_search(
                f"Equation index scan included {len(session.manual_equation_samples)} manual format sample(s)"
            )
        sort_key = lambda item: tuple(int(p) for p in item[0].split("."))
        equation_format = self._build_equation_format_metadata(learning_candidates, patterns)
        if not patterns and previous_format:
            equation_format = previous_format
        equation_format["source"] = (
            "manual_and_equation_index_scan"
            if session.manual_equation_samples
            else "equation_index_scan"
        )
        detected_mode = str(equation_format.get("numbering_mode", EQUATION_MODE_SECTION))
        session.equation_rough_index = dict(sorted(rough_index.items(), key=sort_key))
        session.equation_pattern_index = dict(sorted(pattern_index.items(), key=sort_key))
        session.equation_index_patterns = patterns
        session.equation_format = equation_format
        if detected_mode in {EQUATION_MODE_SECTION, EQUATION_MODE_CHAPTER}:
            session.equation_numbering_mode = detected_mode
            session.equation_cache.clear()
        session.equation_index = session.equation_rough_index
        session.equation_index_signature = self._document_signature(session)
        session.equation_lookup_mode = (
            EQUATION_LOOKUP_PATTERN_INDEX
            if session.equation_pattern_index
            else EQUATION_LOOKUP_ROUGH_INDEX
        )
        self._save_session_state_now()
        mode_label = "chapter" if session.equation_numbering_mode == EQUATION_MODE_CHAPTER else "section-aware"
        self.statusBar().showMessage(
            f"Equation indexes saved: {mode_label} format; rough {len(session.equation_rough_index)}, learned-pattern {len(session.equation_pattern_index)}",
            9000,
        )

    def _scan_learned_equation_patterns(
        self,
        session: DocumentSession,
        learning_candidates: List[EquationCandidate],
    ) -> Tuple[Dict[int, str], Dict[str, EquationLocation]]:
        patterns = self._learn_equation_format_patterns(learning_candidates)
        pattern_index: Dict[str, EquationLocation] = {}
        if patterns:
            pattern_label = ", ".join(f"{part_count} parts: {pattern}" for part_count, pattern in sorted(patterns.items()))
            self._debug_equation_search(f"Learned equation number format from sample: {pattern_label}")
            for page_number in range(1, len(session.doc) + 1):
                if page_number == 1 or page_number % 10 == 0:
                    self._debug_equation_search(
                        f"Equation learned-pattern scan: page {page_number} / {len(session.doc)}"
                    )
                try:
                    page = session.doc.load_page(page_number - 1)
                    candidates = self._pattern_equation_candidates_on_page(page, page_number, patterns)
                except Exception as exc:
                    print(f"[PDF Reader] Equation learned-pattern scan failed on page {page_number}: {exc}", flush=True)
                    continue
                for candidate in candidates:
                    pattern_index.setdefault(candidate.eq_id, candidate.location)
        else:
            self._debug_equation_search("Could not learn an equation number format; keeping rough index only")
        return patterns, pattern_index

    def _start_manual_pattern_index_scan(
        self,
        session: DocumentSession,
        patterns: Dict[int, str],
    ) -> None:
        self._manual_pattern_scan_timer.stop()
        self._manual_pattern_scan_session_id = id(session)
        self._manual_pattern_scan_patterns = dict(patterns)
        self._manual_pattern_scan_index = {}
        self._manual_pattern_scan_page = 1
        self.statusBar().showMessage("Scanning with existing learned equation format...", 6000)
        self._manual_pattern_scan_timer.start(25)

    def _scan_manual_pattern_index_chunk(self) -> None:
        session = self.current_session()
        if (
            session is None
            or id(session) != self._manual_pattern_scan_session_id
            or not self._manual_pattern_scan_patterns
        ):
            self._manual_pattern_scan_timer.stop()
            return

        pages_per_chunk = 1
        scanned = 0
        while self._manual_pattern_scan_page <= len(session.doc) and scanned < pages_per_chunk:
            page_number = self._manual_pattern_scan_page
            try:
                page = session.doc.load_page(page_number - 1)
                candidates = self._pattern_equation_candidates_on_page(
                    page,
                    page_number,
                    self._manual_pattern_scan_patterns,
                )
            except Exception as exc:
                print(f"[PDF Reader] Manual learned-pattern scan failed on page {page_number}: {exc}", flush=True)
                candidates = []

            for candidate in candidates:
                self._manual_pattern_scan_index.setdefault(candidate.eq_id, candidate.location)

            self._manual_pattern_scan_page += 1
            scanned += 1

        if self._manual_pattern_scan_page <= len(session.doc):
            if self._manual_pattern_scan_page == 1 or self._manual_pattern_scan_page % 40 == 0:
                self.statusBar().showMessage(
                    f"Learning equation index: page {self._manual_pattern_scan_page} / {len(session.doc)}",
                    4000,
                )
            return

        self._manual_pattern_scan_timer.stop()
        sort_key = lambda item: tuple(int(p) for p in item[0].split("."))
        if not self._manual_pattern_scan_index:
            dotted_index: Dict[str, EquationLocation] = {}
            for page_number in range(1, len(session.doc) + 1):
                try:
                    page = session.doc.load_page(page_number - 1)
                    candidates = self._dotted_equation_candidates_on_page(page, page_number)
                except Exception as exc:
                    print(f"[PDF Reader] Dotted existing-format fallback failed on page {page_number}: {exc}", flush=True)
                    continue
                for candidate in candidates:
                    dotted_index.setdefault(candidate.eq_id, candidate.location)
            if dotted_index:
                self._manual_pattern_scan_index = dotted_index

        if session.equation_rough_index:
            before_merge = len(self._manual_pattern_scan_index)
            for eq_id, location in session.equation_rough_index.items():
                self._manual_pattern_scan_index.setdefault(eq_id, location)
            if before_merge == 0 and self._manual_pattern_scan_index:
                self.statusBar().showMessage(
                    "Existing learned format found no extra matches; using rough equation hits",
                    7000,
                )
            elif len(self._manual_pattern_scan_index) > before_merge:
                self.statusBar().showMessage(
                    f"Added {len(self._manual_pattern_scan_index) - before_merge} rough display-label matches to the learned equation index",
                    7000,
                )
        if not self._manual_pattern_scan_index and session.equation_rough_index:
            self._manual_pattern_scan_index = dict(session.equation_rough_index)
            self.statusBar().showMessage(
                "Existing learned format found no extra matches; using rough equation hits",
                7000,
            )
        session.equation_pattern_index = dict(sorted(self._manual_pattern_scan_index.items(), key=sort_key))
        session.equation_lookup_mode = EQUATION_LOOKUP_PATTERN_INDEX
        session.equation_index_signature = self._document_signature(session)
        self._save_session_state_now()
        self.statusBar().showMessage(
            f"Existing learned format scan complete; {len(session.equation_pattern_index)} equations indexed",
            9000,
        )

    def _save_learning_rect_snippet(
        self,
        page: int,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> Optional[Path]:
        pixmap = self.view.base_pixmap(page)
        if pixmap is None or pixmap.isNull():
            return None

        left = max(0, int(round(min(x0, x1))))
        top = max(0, int(round(min(y0, y1))))
        right = min(pixmap.width(), int(round(max(x0, x1))))
        bottom = min(pixmap.height(), int(round(max(y0, y1))))
        if right <= left or bottom <= top:
            return None

        snippet = pixmap.copy(left, top, right - left, bottom - top)
        out_path = self.snippet_dir / "last_equation_rectangle.png"
        snippet.save(str(out_path), "PNG")
        return out_path

    def learn_equation_format_from_selected_rectangle(self) -> None:
        session = self.current_session()
        if session is None:
            return

        selected = self.view.selected_learning_rect()
        if selected is None:
            self.statusBar().showMessage(
                "Hold left-click and drag a rectangle around an equation number, then press Ctrl+R",
                7000,
            )
            return

        page_number, x0, y0, x1, y1 = selected
        try:
            page = session.doc.load_page(page_number - 1)
        except Exception as exc:
            self.statusBar().showMessage(f"Could not read selected page: {exc}", 7000)
            return

        clip = fitz.Rect(
            min(x0, x1) / RENDER_SCALE,
            min(y0, y1) / RENDER_SCALE,
            max(x0, x1) / RENDER_SCALE,
            max(y0, y1) / RENDER_SCALE,
        ) & page.rect
        if clip.is_empty or clip.width <= 0 or clip.height <= 0:
            self.statusBar().showMessage("Selected rectangle is outside the PDF page", 6000)
            return

        raw_text = ""
        try:
            raw_text = page.get_text("text", clip=clip, sort=True) or ""
        except Exception:
            raw_text = ""

        matches = self._equation_matches_in_text(raw_text)
        snippet_path: Optional[Path] = None
        if not matches:
            snippet_path = self._save_learning_rect_snippet(page_number, x0, y0, x1, y1)
            if snippet_path is not None:
                try:
                    ocr_text = self._run_builtin_ocr(snippet_path)
                except Exception as exc:
                    ocr_text = ""
                    print(f"[PDF Reader] Manual equation rectangle OCR failed: {exc}", flush=True)
                raw_text = ocr_text or raw_text
                matches = self._equation_matches_in_text(raw_text)

        if not matches:
            suffix = f"; snippet: {snippet_path.name}" if snippet_path is not None else ""
            self.statusBar().showMessage(
                f"No equation number found in selected rectangle{suffix}",
                8000,
            )
            return

        eq_id, raw_match, _start, _end = max(matches, key=lambda item: (len(item[0].split(".")), len(item[1])))
        candidate = EquationCandidate(
            eq_id=eq_id,
            raw_text=raw_match,
            location=EquationLocation(page=page_number, y=max(0.0, clip.y0 - 40.0)),
        )

        session.manual_equation_samples = [
            sample
            for sample in session.manual_equation_samples
            if sample.eq_id != candidate.eq_id
        ]
        session.manual_equation_samples.append(candidate)
        session.manual_equation_samples = session.manual_equation_samples[-MAX_MANUAL_EQUATION_SAMPLES:]

        patterns = self._learn_equation_format_patterns(list(session.manual_equation_samples))
        if not patterns:
            self._save_session_state_now()
            self.statusBar().showMessage(
                f"Saved manual sample ({eq_id}), but could not learn a reusable equation format yet",
                8000,
            )
            return

        sort_key = lambda item: tuple(int(p) for p in item[0].split("."))
        equation_format = self._build_equation_format_metadata(session.manual_equation_samples, patterns)
        equation_format["source"] = "manual_equation_samples"
        session.equation_pattern_index[candidate.eq_id] = candidate.location
        session.equation_pattern_index = dict(sorted(session.equation_pattern_index.items(), key=sort_key))
        session.equation_index[candidate.eq_id] = candidate.location
        session.equation_index_patterns = patterns
        session.equation_format = equation_format
        session.equation_index_signature = self._document_signature(session)
        session.equation_lookup_mode = EQUATION_LOOKUP_PATTERN_INDEX
        detected_mode = str(equation_format.get("numbering_mode", EQUATION_MODE_SECTION))
        if detected_mode in {EQUATION_MODE_SECTION, EQUATION_MODE_CHAPTER}:
            session.equation_numbering_mode = detected_mode
            session.equation_cache.clear()
        self._save_session_state_now()
        self.statusBar().showMessage(
            f"Learned equation format from ({eq_id}); press Ctrl+D to scan with this format",
            9000,
        )

    def toggle_equation_lookup_mode(self) -> None:
        session = self.current_session()
        if session is None:
            return

        current = session.equation_lookup_mode
        if current not in EQUATION_LOOKUP_MODES:
            current = EQUATION_LOOKUP_SCAN
        next_mode = EQUATION_LOOKUP_MODES[(EQUATION_LOOKUP_MODES.index(current) + 1) % len(EQUATION_LOOKUP_MODES)]
        session.equation_lookup_mode = next_mode

        if next_mode == EQUATION_LOOKUP_SCAN:
            msg = "Equation lookup mode: live scan"
        elif next_mode == EQUATION_LOOKUP_ROUGH_INDEX:
            count = len(session.equation_rough_index or session.equation_index)
            msg = f"Equation lookup mode: rough first-scan index ({count} equations)"
            if count == 0:
                msg += "; press Ctrl+D to scan this book"
        else:
            count = len(session.equation_pattern_index)
            msg = f"Equation lookup mode: learned-pattern index ({count} equations)"
            if count == 0:
                msg += "; press Ctrl+D to build it"

        self.statusBar().showMessage(msg, 6000)
        self._save_session_state_now()

    @staticmethod
    def _normalize_figure_id(raw: str) -> str:
        parts = re.findall(r"\d+", raw)
        if len(parts) < 2:
            raise ValueError("Invalid figure number")
        return ".".join(parts)

    @staticmethod
    def _extract_figure_from_text(text: str) -> Optional[str]:
        match = FIGURE_REF_RE.search(text)
        if not match:
            return None
        try:
            return PdfReaderWindow._normalize_figure_id(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _word_rect_contains_point(word: tuple, point: fitz.Point, tolerance: float = 4.0) -> bool:
        rect = fitz.Rect(word[:4])
        rect.x0 -= tolerance
        rect.y0 -= tolerance
        rect.x1 += tolerance
        rect.y1 += tolerance
        return rect.contains(point)

    def _collect_text_line_at_position(
        self,
        session: DocumentSession,
        page_number: int,
        page_x: float,
        page_y: float,
    ) -> Optional[str]:
        """Read the text-layer line under the clicked rendered-page coordinate."""
        try:
            pdf_page = session.doc.load_page(page_number - 1)
            text_dict = pdf_page.get_text("dict")
        except Exception:
            return None

        pdf_x = page_x / RENDER_SCALE
        pdf_y = page_y / RENDER_SCALE
        click_rect = fitz.Rect(
            pdf_x - PDF_TEXT_CLICK_RADIUS,
            pdf_y - PDF_TEXT_CLICK_RADIUS,
            pdf_x + PDF_TEXT_CLICK_RADIUS,
            pdf_y + PDF_TEXT_CLICK_RADIUS,
        )

        best_line_text: Optional[str] = None
        best_dist2: Optional[float] = None

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_spans = line.get("spans", [])
                if not line_spans:
                    continue

                line_rect = fitz.Rect(line_spans[0]["bbox"])
                for span in line_spans[1:]:
                    line_rect |= fitz.Rect(span["bbox"])

                expanded_rect = fitz.Rect(
                    line_rect.x0,
                    line_rect.y0 - PDF_TEXT_LINE_Y_TOLERANCE,
                    line_rect.x1,
                    line_rect.y1 + PDF_TEXT_LINE_Y_TOLERANCE,
                )
                if not expanded_rect.intersects(click_rect):
                    continue

                line_text = "".join(str(span.get("text", "")) for span in line_spans)
                if not line_text.strip():
                    continue

                cx = (line_rect.x0 + line_rect.x1) / 2.0
                cy = (line_rect.y0 + line_rect.y1) / 2.0
                dist2 = (cx - pdf_x) ** 2 + (cy - pdf_y) ** 2
                if best_dist2 is None or dist2 < best_dist2:
                    best_dist2 = dist2
                    best_line_text = line_text

        return best_line_text

    def _text_layer_equation_candidates_near_position(
        self,
        session: DocumentSession,
        page_number: int,
        page_x: float,
        page_y: float,
    ) -> List[Tuple[str, str, fitz.Rect, float]]:
        try:
            pdf_page = session.doc.load_page(page_number - 1)
            words = pdf_page.get_text("words")
        except Exception:
            return []

        if not words:
            return []

        pdf_x = page_x / RENDER_SCALE
        pdf_y = page_y / RENDER_SCALE
        nearby_rect = fitz.Rect(
            max(0, pdf_x - PDF_TEXT_NEARBY_X_MARGIN),
            max(0, pdf_y - PDF_TEXT_NEARBY_Y_MARGIN),
            min(pdf_page.rect.width, pdf_x + PDF_TEXT_NEARBY_X_MARGIN),
            min(pdf_page.rect.height, pdf_y + PDF_TEXT_NEARBY_Y_MARGIN),
        )

        candidates: List[Tuple[str, str, fitz.Rect, float]] = []
        seen = set()
        for word in words:
            word_text = str(word[4]).strip()
            if not word_text:
                continue

            word_rect = fitz.Rect(word[:4])
            if not word_rect.intersects(nearby_rect):
                continue
            dx = 0.0 if word_rect.x0 <= pdf_x <= word_rect.x1 else min(abs(pdf_x - word_rect.x0), abs(pdf_x - word_rect.x1))
            dy = abs(pdf_y - ((word_rect.y0 + word_rect.y1) / 2.0))
            if dx > PDF_TEXT_NEARBY_X_MARGIN or dy > PDF_TEXT_NEARBY_Y_MARGIN:
                continue

            for match in TEXT_LAYER_EQUATION_REF_RE.finditer(word_text):
                raw = match.group(0)
                if "." not in raw and "," not in raw and "-" not in raw:
                    continue
                try:
                    eq_id = self._normalize_equation_id(raw)
                except ValueError:
                    continue

                key = (eq_id, round(word_rect.x0, 2), round(word_rect.y0, 2), round(word_rect.x1, 2), round(word_rect.y1, 2))
                if key in seen:
                    continue
                seen.add(key)

                candidates.append((
                    eq_id,
                    raw,
                    word_rect,
                    float((word_rect.y0 + word_rect.y1) / 2.0),
                ))

        return candidates

    def _equation_reference_at_position(
        self,
        session: DocumentSession,
        page_number: int,
        page_x: float,
        page_y: float,
    ) -> Optional[str]:
        candidates = self._text_layer_equation_candidates_near_position(session, page_number, page_x, page_y)
        if not candidates:
            return None

        pdf_x = page_x / RENDER_SCALE
        pdf_y = page_y / RENDER_SCALE

        def score(candidate: Tuple[str, str, fitz.Rect, float]) -> Tuple[int, float, float]:
            _eq_id, _raw, rect, line_y = candidate
            dx = 0.0 if rect.x0 <= pdf_x <= rect.x1 else min(abs(pdf_x - rect.x0), abs(pdf_x - rect.x1))
            dy = abs(pdf_y - line_y)
            direct_hit = dy <= PDF_TEXT_CLICK_RADIUS and dx <= PDF_TEXT_CLICK_RADIUS
            return (0 if direct_hit else 1, dy, dx)

        best = min(candidates, key=score)
        return best[0]

    def _clicked_reference_from_text_layer(
        self,
        session: DocumentSession,
        page_number: int,
        page_x: float,
        page_y: float,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        raw_text = self._collect_text_line_at_position(session, page_number, page_x, page_y)

        if raw_text is not None:
            fig_id = self._extract_figure_from_text(raw_text)
            if fig_id is not None:
                return None, fig_id, raw_text

        eq_id = self._equation_reference_at_position(session, page_number, page_x, page_y)
        if eq_id is not None:
            return eq_id, None, raw_text

        return None, None, raw_text

    def _clicked_reference_with_ocr_fallback(
        self,
        session: DocumentSession,
        page_number: int,
        page_x: float,
        page_y: float,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Path], str, str]:
        eq_id, fig_id, raw_text = self._clicked_reference_from_text_layer(session, page_number, page_x, page_y)
        if eq_id is not None or fig_id is not None:
            return eq_id, fig_id, raw_text, None, "PDF text layer", "text"

        raw_ocr_text, snippet_path, details = self._ocr_clicked_text(page_number, page_x, page_y)
        if raw_ocr_text is None:
            if raw_text:
                details = f"PDF text did not contain an equation or figure reference: {raw_text!r}; {details}"
            return None, None, None, snippet_path, details, "ocr"

        fig_id = self._extract_figure_from_text(raw_ocr_text)
        if fig_id is not None:
            return None, fig_id, raw_ocr_text, snippet_path, raw_ocr_text, "ocr"

        eq_id = self._extract_equation_from_text(raw_ocr_text)
        if eq_id is not None:
            return eq_id, None, raw_ocr_text, snippet_path, raw_ocr_text, "ocr"

        return None, None, raw_ocr_text, snippet_path, raw_ocr_text, "ocr"

    def _clicked_equation_reference_with_ocr_fallback(
        self,
        session: DocumentSession,
        page_number: int,
        page_x: float,
        page_y: float,
    ) -> Tuple[Optional[str], Optional[str], Optional[Path], str]:
        eq_id, _fig_id, raw_text = self._clicked_reference_from_text_layer(session, page_number, page_x, page_y)
        if eq_id is not None:
            return eq_id, raw_text, None, "text"

        raw_ocr_text, snippet_path, details = self._ocr_clicked_text(page_number, page_x, page_y)
        if raw_ocr_text is None:
            if raw_text:
                details = f"PDF text did not contain an equation reference: {raw_text!r}; {details}"
            return None, details, snippet_path, "ocr"

        eq_id = self._extract_equation_from_text(raw_ocr_text)
        if eq_id is None:
            return None, f"OCR text did not contain an equation reference: {raw_ocr_text!r}", snippet_path, "ocr"

        return eq_id, raw_ocr_text, snippet_path, "ocr"

    def _figure_reference_at_position(
        self,
        session: DocumentSession,
        page_number: int,
        page_x: float,
        page_y: float,
    ) -> Optional[str]:
        """Read a Fig./Figure reference from the PDF text layer under the click."""
        try:
            pdf_page = session.doc.load_page(page_number - 1)
            words = pdf_page.get_text("words")
        except Exception:
            return None

        if not words:
            return None

        pdf_point = fitz.Point(page_x / RENDER_SCALE, page_y / RENDER_SCALE)
        clicked_indexes = [
            index for index, word in enumerate(words)
            if self._word_rect_contains_point(word, pdf_point)
        ]
        if not clicked_indexes:
            return None

        line_words_by_key: Dict[Tuple[int, int], List[Tuple[int, tuple]]] = {}
        for index, word in enumerate(words):
            key = (int(word[5]), int(word[6]))
            line_words_by_key.setdefault(key, []).append((index, word))

        for clicked_index in clicked_indexes:
            clicked_word = words[clicked_index]
            key = (int(clicked_word[5]), int(clicked_word[6]))
            line_words = sorted(line_words_by_key.get(key, []), key=lambda item: (item[1][7], item[1][0]))

            chunks: List[str] = []
            spans: List[Tuple[int, int, int]] = []
            cursor = 0
            for original_index, word in line_words:
                if chunks:
                    chunks.append(" ")
                    cursor += 1
                token = str(word[4])
                start = cursor
                chunks.append(token)
                cursor += len(token)
                spans.append((start, cursor, original_index))

            line_text = "".join(chunks)
            for match in FIGURE_REF_RE.finditer(line_text):
                match_start, match_end = match.span()
                matched_word_indexes = {
                    original_index
                    for start, end, original_index in spans
                    if start < match_end and end > match_start
                }
                if clicked_index not in matched_word_indexes:
                    continue
                try:
                    return self._normalize_figure_id(match.group(1))
                except ValueError:
                    continue

        return None

    def _find_figure_destination(self, session: DocumentSession, fig_id: str) -> Optional[FigureDestination]:
        fig_id = self._normalize_figure_id(fig_id)
        cached = session.figure_cache.get(fig_id)
        if cached is not None:
            return cached

        caption_variants = [f"FIGURE {fig_id}", f"Figure {fig_id}"]
        reference_variants = [f"Fig. {fig_id}", f"Fig {fig_id}", f"Figure {fig_id}"]
        fallback: Optional[FigureDestination] = None

        for page_index in range(len(session.doc)):
            page = session.doc.load_page(page_index)
            for variant in caption_variants:
                try:
                    hits = page.search_for(variant)
                except Exception:
                    hits = []
                if hits:
                    destination = FigureDestination(page=page_index + 1, y=float(hits[0].y0))
                    session.figure_cache[fig_id] = destination
                    return destination

            if fallback is None:
                for variant in reference_variants:
                    try:
                        hits = page.search_for(variant)
                    except Exception:
                        hits = []
                    if hits:
                        fallback = FigureDestination(page=page_index + 1, y=float(hits[0].y0))
                        break

        if fallback is not None:
            session.figure_cache[fig_id] = fallback
        return fallback

    def _jump_to_figure_destination(
        self,
        session: DocumentSession,
        fig_id: str,
        source_page: int,
        snippet_path: Optional[Path] = None,
    ) -> bool:
        destination = self._find_figure_destination(session, fig_id)
        if destination is None:
            suffix = f" | snippet: {snippet_path.name}" if snippet_path is not None else ""
            self.statusBar().showMessage(f"Figure {fig_id} was recognized, but its caption was not found{suffix}", 8000)
            return False

        session.current_page = source_page
        self._remember_equation_source_page()
        self._show_page_for_session(session, destination.page, push_history=True)
        self.view.scroll_to_page(destination.page, destination.y * RENDER_SCALE)

        suffix = f" | snippet: {snippet_path.name}" if snippet_path is not None else ""
        self.statusBar().showMessage(f"Jumped to Figure {fig_id} on page {destination.page}{suffix}", 8000)
        return True

    def _save_snippet(self, page: int, scene_x: float, scene_y: float) -> Optional[Path]:
        pixmap = self.view.base_pixmap(page)
        if pixmap is None or pixmap.isNull():
            return None

        x0 = max(0, int(round(scene_x - CLICK_BOX_W / 2)))
        y0 = max(0, int(round(scene_y - CLICK_BOX_H / 2)))
        w = min(CLICK_BOX_W, pixmap.width() - x0)
        h = min(CLICK_BOX_H, pixmap.height() - y0)

        if w <= 0 or h <= 0:
            return None

        snippet = pixmap.copy(x0, y0, w, h)
        out_path = self.snippet_dir / "last_equation_click.png"
        snippet.save(str(out_path), "PNG")
        return out_path

    def _extract_equation_from_text(self, text: str) -> Optional[str]:
        candidates = list(EQUATION_REF_RE.finditer(text))
        if not candidates:
            cleaned = re.sub(r"[^0-9().,\-\s]", "", text)
            candidates = list(EQUATION_REF_RE.finditer(cleaned))
            if not candidates:
                return None

        best = max((m.group(0) for m in candidates), key=len)
        try:
            return self._normalize_equation_id(best)
        except ValueError:
            return None

    def _run_builtin_ocr(self, image_path: Path) -> str:
        swift_code = r'''
import Foundation
import Vision
import AppKit

let path = CommandLine.arguments[1]
let url = URL(fileURLWithPath: path)

guard let image = NSImage(contentsOf: url) else {
    fputs("Failed to load image\n", stderr)
    exit(1)
}

guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let cgImage = bitmap.cgImage else {
    fputs("Failed to convert image\n", stderr)
    exit(1)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
request.recognitionLanguages = ["en-US"]

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

do {
    try handler.perform([request])
    let results = request.results ?? []
    let strings = results.compactMap { obs in
        obs.topCandidates(1).first?.string
    }
    print(strings.joined(separator: "\n"))
} catch {
    fputs("OCR failed: \(error)\n", stderr)
    exit(1)
}
'''
        with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as f:
            f.write(swift_code)
            swift_file = Path(f.name)

        try:
            result = subprocess.run(
                ["swift", str(swift_file), str(image_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        finally:
            swift_file.unlink(missing_ok=True)

    def _ocr_clicked_text(self, page: int, scene_x: float, scene_y: float) -> Tuple[Optional[str], Optional[Path], str]:
        snippet_path = self._save_snippet(page, scene_x, scene_y)
        if snippet_path is None:
            return None, None, "Could not capture click snippet"

        try:
            raw_text = self._run_builtin_ocr(snippet_path)
        except FileNotFoundError:
            return None, snippet_path, "Could not run OCR: Swift is not installed or not in PATH"
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            return None, snippet_path, f"OCR failed{': ' + stderr if stderr else ''}"
        except Exception as exc:
            return None, snippet_path, f"OCR failed: {exc}"

        return raw_text, snippet_path, raw_text

    def _ocr_clicked_reference(self, page: int, scene_x: float, scene_y: float) -> Tuple[Optional[str], Optional[Path], str]:
        raw_text, snippet_path, details = self._ocr_clicked_text(page, scene_x, scene_y)
        if raw_text is None:
            return None, snippet_path, details

        eq_id = self._extract_equation_from_text(raw_text)
        if eq_id is None:
            return None, snippet_path, f"OCR text did not contain an equation reference: {raw_text!r}"
        return eq_id, snippet_path, raw_text

    def _schedule_equation_preview(self, page: int, scene_x: float, scene_y: float, global_pos: QPoint) -> None:
        """Debounce hover lookup and show a small image preview of the target equation.

        Moving over an equation reference only previews the target equation. The
        actual jump still happens only in jump_if_reference_clicked(), after a
        click.
        """
        if not self._equation_preview_enabled:
            return
        if self.current_session() is None or self.view.base_pixmap(page) is None:
            self._hide_equation_preview()
            return
        self._hover_preview_request = (page, scene_x, scene_y, global_pos)
        self._hover_preview_timer.start(450)

    def _hide_equation_preview(self) -> None:
        if hasattr(self, "_hover_preview_timer"):
            self._hover_preview_timer.stop()
        self._hover_preview_request = None
        self._last_preview_key = None
        if hasattr(self, "equation_preview_label"):
            self.equation_preview_label.hide()

    def _find_equation_number_rect_on_page(self, page: fitz.Page, eq_id: str) -> Optional[fitz.Rect]:
        variants = self._build_search_variants(eq_id)
        # Prefer the plain dotted number because RTL PDFs often split or reverse
        # the visible parentheses around the equation number.
        dotted = ".".join(eq_id.split("."))
        ordered_variants = [dotted] + [v for v in variants if v != dotted]
        for variant in ordered_variants:
            try:
                hits = page.search_for(variant)
            except Exception:
                hits = []
            if hits:
                return hits[0]
        return None

    def _render_equation_preview_pixmap(
        self,
        session: DocumentSession,
        eq_id: str,
        page_number: int,
    ) -> Optional[QPixmap]:
        try:
            page = session.doc.load_page(page_number - 1)
            hit = self._find_equation_number_rect_on_page(page, eq_id)
            if hit is None:
                return None

            # Crop a horizontal band that starts slightly before the equation
            # number and extends rightward across the formula. This keeps the
            # preview focused on the equation rather than the whole page.
            clip = fitz.Rect(
                max(0, hit.x0 - 25),
                max(0, hit.y0 - 35),
                min(page.rect.width, hit.x1 + 560),
                min(page.rect.height, hit.y1 + 65),
            )
            pix = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), clip=clip, alpha=False)
            image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
            preview = QPixmap.fromImage(image)
            if preview.width() > 720:
                preview = preview.scaledToWidth(720, Qt.SmoothTransformation)
            return preview
        except Exception as exc:
            print(f"[PDF Reader] Equation preview failed for {eq_id}: {exc}", flush=True)
            return None

    def _show_equation_preview_from_hover(self) -> None:
        session = self.current_session()
        if session is None or self._hover_preview_request is None:
            self._hide_equation_preview()
            return

        page, scene_x, scene_y, global_pos = self._hover_preview_request

        eq_id, _details, _snippet_path, _source = self._clicked_equation_reference_with_ocr_fallback(
            session,
            page,
            scene_x,
            scene_y,
        )
        if not eq_id:
            self.equation_preview_label.hide()
            self._last_preview_key = None
            return

        page_num = self._find_equation_page(session, eq_id)
        if page_num is None:
            self.equation_preview_label.hide()
            self._last_preview_key = None
            return

        key = (eq_id, page_num)
        if key != self._last_preview_key or self.equation_preview_label.pixmap() is None:
            preview = self._render_equation_preview_pixmap(session, eq_id, page_num)
            if preview is None:
                self.equation_preview_label.hide()
                self._last_preview_key = None
                return
            self.equation_preview_label.setPixmap(preview)
            self._last_preview_key = key

        # Put the popup near the cursor, but keep it fully visible on screen.
        # Normally it appears to the right of the cursor. If the equation reference
        # is near the right side of the screen, show the preview to the left instead.
        preview_size = self.equation_preview_label.sizeHint()
        screen = QApplication.screenAt(global_pos) or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None

        offset = QPoint(18, 18)
        proposed = global_pos + offset

        if available is not None:
            if proposed.x() + preview_size.width() > available.right():
                proposed.setX(global_pos.x() - preview_size.width() - offset.x())

            if proposed.y() + preview_size.height() > available.bottom():
                proposed.setY(available.bottom() - preview_size.height())

            proposed.setX(max(available.left(), proposed.x()))
            proposed.setY(max(available.top(), proposed.y()))

        self.equation_preview_label.move(proposed)
        self.equation_preview_label.show()
        self._equation_preview_enabled = False

    def preview_equation_under_cursor(self) -> None:
        session = self.current_session()
        if session is None:
            return
        cursor_pos = self.view.viewport().mapFromGlobal(QCursor.pos())
        if not self.view.viewport().rect().contains(cursor_pos):
            return
        scene_pos = self.view.mapToScene(cursor_pos)
        page_pos = self.view.page_at_scene_pos(scene_pos.x(), scene_pos.y())
        if page_pos is None:
            return
        page, page_x, page_y = page_pos
        self._equation_preview_enabled = True
        self._hover_preview_request = (page, page_x, page_y, self.view.viewport().mapToGlobal(cursor_pos))
        self._show_equation_preview_from_hover()

    def _link_at_position(self, session: DocumentSession, page_number: int, page_x: float, page_y: float) -> Optional[dict]:
        try:
            pdf_page = session.doc.load_page(page_number - 1)
        except Exception as exc:
            self.statusBar().showMessage(f"Could not inspect PDF links on page {page_number}: {exc}", 6000)
            return None

        pdf_point = fitz.Point(page_x / RENDER_SCALE, page_y / RENDER_SCALE)
        for link in pdf_page.get_links():
            rect = link.get("from")
            if rect is not None and fitz.Rect(rect).contains(pdf_point):
                return link
        return None

    def _open_external_link(self, uri: str) -> None:
        url = QUrl.fromUserInput(uri)
        if not url.isValid():
            self.statusBar().showMessage(f"Invalid link: {uri}", 6000)
            return
        if QDesktopServices.openUrl(url):
            self.statusBar().showMessage(f"Opened link: {uri}", 6000)
        else:
            self.statusBar().showMessage(f"Could not open link: {uri}", 6000)

    def _open_internal_link(self, session: DocumentSession, link: dict) -> bool:
        target_index = link.get("page", -1)
        target_y = 0.0

        if not isinstance(target_index, int) or target_index < 0:
            name = link.get("name")
            uri = link.get("uri")
            try:
                if name:
                    target_index, _target_x, target_y = session.doc.resolve_link(name)
                elif uri:
                    target_index, _target_x, target_y = session.doc.resolve_link(uri)
            except Exception:
                target_index = -1

        target_page = target_index + 1 if isinstance(target_index, int) and target_index >= 0 else 0
        if not (1 <= target_page <= len(session.doc)):
            return False

        target = link.get("to")
        if target is not None and hasattr(target, "y"):
            target_y = float(target.y)

        self._show_page_for_session(session, target_page, push_history=True)
        if target_y > 0:
            self.view.scroll_to_page(target_page, target_y * RENDER_SCALE)
        self.statusBar().showMessage(f"Opened internal link to page {target_page}", 6000)
        return True

    def open_link_clicked(self, page: int, scene_x: float, scene_y: float) -> None:
        session = self.current_session()
        if session is None:
            return

        link = self._link_at_position(session, page, scene_x, scene_y)
        if link is None:
            self.statusBar().showMessage("No PDF link at this position", 3000)
            return

        kind = link.get("kind")
        uri = link.get("uri")
        if kind == fitz.LINK_URI and uri:
            self._open_external_link(str(uri))
            return

        if kind in {fitz.LINK_GOTO, fitz.LINK_NAMED} and self._open_internal_link(session, link):
            return

        if kind == getattr(fitz, "LINK_GOTOR", None) and uri:
            self._open_external_link(str(uri))
            return

        self.statusBar().showMessage("This PDF link type is not supported yet", 5000)

    def jump_if_reference_clicked(self, page: int, scene_x: float, scene_y: float) -> None:
        session = self.current_session()
        if session is None:
            return

        session.current_page = page
        self.view.show_debug_box(page, scene_x, scene_y, CLICK_BOX_W, CLICK_BOX_H)

        fig_id = self._figure_reference_at_position(session, page, scene_x, scene_y)
        if fig_id is not None and self._jump_to_figure_destination(session, fig_id, page):
            return

        eq_id, fig_id, raw_text, snippet_path, details, source = self._clicked_reference_with_ocr_fallback(
            session,
            page,
            scene_x,
            scene_y,
        )
        if fig_id is not None and self._jump_to_figure_destination(session, fig_id, page, snippet_path):
            return

        if not eq_id:
            msg = details
            if raw_text:
                msg = f"{source.upper()} text did not contain an equation or figure reference: {raw_text!r}"
            if snippet_path is not None:
                msg += f" | snippet: {snippet_path.name}"
            self.statusBar().showMessage(msg, 8000)
            return

        self.eq_input.setText(eq_id)
        destination = self._find_equation_destination(session, eq_id)
        if destination is None:
            suffix = f" | snippet: {snippet_path.name}" if snippet_path is not None else ""
            self.statusBar().showMessage(
                f"{source.upper()} found {eq_id}, but destination page was not found{suffix}",
                8000,
            )
            return

        self._remember_equation_source_page()
        suffix = f" | snippet: {snippet_path.name}" if snippet_path is not None else ""
        self.statusBar().showMessage(
            f"{source.upper()} found {eq_id}, jumping to page {destination.page}{suffix}",
            8000,
        )
        self._jump_to_equation_location(session, destination, push_history=True)

    def research_equation_reference_clicked(self, page: int, scene_x: float, scene_y: float) -> None:
        """Right-click an equation reference to forget its cached page and search again."""
        session = self.current_session()
        if session is None:
            return

        session.current_page = page
        self.view.show_debug_box(page, scene_x, scene_y, CLICK_BOX_W, CLICK_BOX_H)

        eq_id, details, snippet_path, source = self._clicked_equation_reference_with_ocr_fallback(
            session,
            page,
            scene_x,
            scene_y,
        )
        if not eq_id:
            msg = details
            if snippet_path is not None:
                msg += f" | snippet: {snippet_path.name}"
            self.statusBar().showMessage(msg, 8000)
            return

        old_page = session.equation_cache.pop(eq_id, None)
        self.eq_input.setText(eq_id)
        self._debug_equation_search(
            f"Equation search: right-click re-search for {eq_id}; "
            f"forgot cached page {old_page if old_page is not None else 'none'}"
        )

        page_num = self._find_equation_page(session, eq_id)
        if page_num is None:
            self.statusBar().showMessage(
                f"{source.upper()} found {eq_id}, cache was cleared, but destination page was not found"
                + (f" | snippet: {snippet_path.name}" if snippet_path else ""),
                8000,
            )
            return

        self._remember_equation_source_page()
        self.statusBar().showMessage(
            f"Re-searched {eq_id} after clearing cache; jumping to page {page_num}"
            + (f" | snippet: {snippet_path.name}" if snippet_path else ""),
            8000,
        )
        self._show_page_for_session(session, page_num, push_history=True)


    def jump_to_equation(self) -> None:
        session = self.current_session()
        if session is None:
            return

        text = self.eq_input.text().strip()
        if not text:
            return

        try:
            destination = self._find_equation_destination(session, text)
        except ValueError:
            QMessageBox.warning(self, "Invalid", "Invalid equation number")
            return

        if destination is None:
            QMessageBox.information(self, "Not found", "Equation not found")
            return

        self._remember_equation_source_page()
        self._jump_to_equation_location(session, destination, push_history=True)


    def _redraw_selection_lines(self) -> None:
        session = self.current_session()
        if session is None:
            return
        self.view.show_selection_lines(session.selection_lines)

    def add_selection_line_at_position(self, page: int, _scene_x: float, scene_y: float) -> None:
        session = self.current_session()
        pixmap = self.view.base_pixmap(page)
        if session is None or pixmap is None or pixmap.isNull():
            return

        scene_y = max(0.0, min(float(scene_y), float(pixmap.height())))
        session.current_page = page
        session.selection_lines.append(SelectionLine(page=page, y=scene_y))
        self._redraw_selection_lines()
        self.statusBar().showMessage(
            f"Added blue line {len(session.selection_lines)} on page {page}. "
            f"Press Shift+C to copy with equation context, Shift+V to copy without it, or Shift+X to clear all blue lines.",
            7000,
        )

    def clear_selection_lines(self) -> None:
        session = self.current_session()
        if session is None:
            return

        state = self._layout_change_view_state()
        count = len(session.selection_lines)
        self._suspend_view_updates = True
        try:
            session.selection_lines.clear()
            self._redraw_selection_lines()
            self._restore_layout_change_view_state(state)
        finally:
            self._suspend_view_updates = False
        self._save_active_session_view()
        self.statusBar().showMessage(
            f"Cleared {count} blue line{'s' if count != 1 else ''}.",
            5000,
        )

    @staticmethod
    def _selection_sort_key(line: SelectionLine) -> Tuple[int, float]:
        return (line.page, line.y)

    @staticmethod
    def _line_pdf_y(line: SelectionLine) -> float:
        return max(0.0, float(line.y) / RENDER_SCALE)

    def _selection_pdf_band_for_page(
        self,
        session: DocumentSession,
        start: SelectionLine,
        end: SelectionLine,
        page_number: int,
    ) -> Optional[Tuple[float, float]]:
        if not (start.page <= page_number <= end.page):
            return None

        page = session.doc.load_page(page_number - 1)
        y0 = 0.0
        y1 = float(page.rect.height)
        if page_number == start.page:
            y0 = min(float(page.rect.height), self._line_pdf_y(start))
        if page_number == end.page:
            y1 = max(0.0, self._line_pdf_y(end))
        if y1 < y0:
            y0, y1 = y1, y0
        return y0, y1

    def _extract_text_between_two_lines(
        self,
        session: DocumentSession,
        first: SelectionLine,
        second: SelectionLine,
    ) -> str:
        start, end = sorted([first, second], key=self._selection_sort_key)
        chunks: List[str] = []

        for page_number in range(start.page, end.page + 1):
            page = session.doc.load_page(page_number - 1)
            band = self._selection_pdf_band_for_page(session, start, end, page_number)
            if band is None:
                continue
            y0, y1 = band
            clip = fitz.Rect(0, y0, page.rect.width, y1)
            try:
                text = page.get_text("text", clip=clip, sort=True).strip()
            except Exception:
                text = ""
            if text:
                chunks.append(f"--- Page {page_number} ---\n{text}")

        return "\n\n".join(chunks).strip()

    @staticmethod
    def _plain_comment_text(comment_text: str) -> str:
        if "<html" in comment_text.lower() or "<span" in comment_text.lower() or "<p" in comment_text.lower():
            doc = QTextDocument()
            doc.setHtml(comment_text)
            return doc.toPlainText().strip()
        return comment_text.strip()

    def _comment_reference_text(self, session: DocumentSession, comment: PdfComment) -> str:
        if not (1 <= comment.page <= len(session.doc)):
            return ""
        page = session.doc.load_page(comment.page - 1)
        rect = fitz.Rect(comment.rect) & page.rect
        if rect.is_empty:
            return ""

        try:
            text = page.get_text("text", clip=rect, sort=True).strip()
        except Exception:
            text = ""
        if text:
            return text

        expanded = fitz.Rect(
            max(0.0, rect.x0 - 20.0),
            max(0.0, rect.y0 - 12.0),
            min(page.rect.width, rect.x1 + 260.0),
            min(page.rect.height, rect.y1 + 36.0),
        )
        try:
            return page.get_text("text", clip=expanded, sort=True).strip()
        except Exception:
            return ""

    def _comments_between_selection_lines(
        self,
        session: DocumentSession,
        first: SelectionLine,
        second: SelectionLine,
    ) -> List[Tuple[int, PdfComment, str]]:
        start, end = sorted([first, second], key=self._selection_sort_key)
        selected_comments: List[Tuple[int, PdfComment, str]] = []

        for index, comment in enumerate(session.comments, start=1):
            if not (1 <= comment.page <= len(session.doc)):
                continue
            band = self._selection_pdf_band_for_page(session, start, end, comment.page)
            if band is None:
                continue
            y0, y1 = band
            comment_rect = fitz.Rect(comment.rect)
            if comment_rect.y1 < y0 or comment_rect.y0 > y1:
                continue
            selected_comments.append((index, comment, self._comment_reference_text(session, comment)))

        selected_comments.sort(key=lambda item: (item[1].page, item[1].rect[1], item[1].rect[0]))
        return selected_comments

    def _export_comments_context(
        self,
        session: DocumentSession,
        first: SelectionLine,
        second: SelectionLine,
    ) -> str:
        blocks: List[str] = []
        for comment_number, (_index, comment, reference_text) in enumerate(
            self._comments_between_selection_lines(session, first, second),
            start=1,
        ):
            comment_text = self._plain_comment_text(comment.text)
            rect_label = (
                f"page {comment.page}, rect=({comment.rect[0]:.1f}, {comment.rect[1]:.1f}, "
                f"{comment.rect[2]:.1f}, {comment.rect[3]:.1f})"
            )
            block = [f"[Comment {comment_number} | {rect_label}]"]
            block.append("Text this comment refers to:")
            block.append(reference_text or "[No extractable text under the comment rectangle.]")
            block.append("Comment/question:")
            block.append(comment_text or "[Empty comment text.]")
            blocks.append("\n".join(block))

        return "\n\n".join(blocks)

    @staticmethod
    def _find_parenthesized_equation_refs(text: str) -> List[str]:
        refs: List[str] = []
        seen = set()
        for match in re.finditer(r"\(\s*\d+(?:\s*\.\s*\d+)+\s*\)", text):
            eq_id = ".".join(re.findall(r"\d+", match.group(0)))
            if eq_id not in seen:
                seen.add(eq_id)
                refs.append(eq_id)
        return refs

    def _equation_clip_for_export(self, page: fitz.Page, eq_id: str) -> Optional[fitz.Rect]:
        hit = self._find_equation_number_rect_on_page(page, eq_id)
        if hit is None:
            return None
        return fitz.Rect(
            max(0, hit.x0 - 45),
            max(0, hit.y0 - 45),
            min(page.rect.width, hit.x1 + 650),
            min(page.rect.height, hit.y1 + 90),
        )

    def _equation_location_for_export(
        self,
        session: DocumentSession,
        eq_id: str,
    ) -> Tuple[Optional[EquationLocation], Optional[str]]:
        if session.equation_lookup_mode == EQUATION_LOOKUP_PATTERN_INDEX:
            location = session.equation_pattern_index.get(eq_id)
            return location, "learned-pattern equation index"
        if session.equation_lookup_mode == EQUATION_LOOKUP_ROUGH_INDEX:
            location = session.equation_rough_index.get(eq_id) or session.equation_index.get(eq_id)
            return location, "rough equation index"

        page_number = self._find_equation_page(session, eq_id)
        if page_number is None:
            return None, None
        return EquationLocation(page=page_number, y=0.0), None

    def _export_equation_context(self, session: DocumentSession, eq_ids: List[str]) -> str:
        if not eq_ids:
            return ""

        blocks: List[str] = []

        for eq_id in eq_ids:
            location, index_label = self._equation_location_for_export(session, eq_id)
            if location is None:
                if index_label:
                    blocks.append(f"[Equation ({eq_id}) was referenced, but it is not in the {index_label}.]")
                else:
                    blocks.append(f"[Equation ({eq_id}) was referenced, but its page was not found.]")
                continue

            page_number = location.page
            page = session.doc.load_page(page_number - 1)
            clip = self._equation_clip_for_export(page, eq_id)
            if clip is None:
                blocks.append(f"[Equation ({eq_id}), page {page_number}: could not locate the equation number on the page.]")
                continue

            try:
                equation_text = page.get_text("text", clip=clip, sort=True).strip()
            except Exception:
                equation_text = ""

            block = [f"[Referenced equation ({eq_id}), page {page_number}]"]
            if equation_text:
                block.append("Extracted equation text and immediate surroundings:")
                block.append(equation_text)
            else:
                block.append("Extracted equation text: unavailable from the PDF text layer.")
            blocks.append("\n".join(block))

        return "\n\n".join(blocks)

    def _copy_between_selection_lines(self, include_equation_context: bool) -> None:
        session = self.current_session()
        if session is None:
            return
        if len(session.selection_lines) < 2:
            self.statusBar().showMessage("Add at least two blue lines with Shift+Click first", 6000)
            return

        first, second = session.selection_lines[-2], session.selection_lines[-1]
        body_text = self._extract_text_between_two_lines(session, first, second)
        if not body_text:
            self.statusBar().showMessage("No extractable text was found between the two most recent blue lines", 7000)
            return

        refs = self._find_parenthesized_equation_refs(body_text)
        equation_context = self._export_equation_context(session, refs) if include_equation_context else ""
        comments_context = self._export_comments_context(session, first, second)
        start, end = sorted([first, second], key=self._selection_sort_key)

        copied_parts = [
            "SECTION FROM PDF",
            f"Source PDF: {session.path}",
            f"Selection start: page {start.page}, rendered y={start.y:.1f}",
            f"Selection end: page {end.page}, rendered y={end.y:.1f}",
            "",
            "MAIN TEXT:",
            body_text,
        ]
        if include_equation_context and refs:
            copied_parts.extend([
                "",
                "EQUATIONS NEEDED FOR CONTEXT:",
                ", ".join(f"({ref})" for ref in refs),
                "",
                equation_context,
            ])
        if comments_context:
            copied_parts.extend([
                "",
                "COMMENTS ON SELECTED TEXT:",
                "Each comment is paired with the PDF text under the highlighted/commented rectangle so the question can be answered in context.",
                "",
                comments_context,
            ])

        clipboard_text = "\n".join(part for part in copied_parts if part is not None)
        QApplication.clipboard().setText(clipboard_text)
        comment_count = comments_context.count("[Comment ")
        equation_message = (
            f"{len(refs)} referenced equations and "
            if include_equation_context
            else "equation references skipped; "
        )
        self.statusBar().showMessage(
            f"Copied section between the two most recent blue lines; {equation_message}{comment_count} comments included",
            8000,
        )

    def copy_between_selection_lines(self) -> None:
        self._copy_between_selection_lines(include_equation_context=True)

    def copy_between_selection_lines_without_equations(self) -> None:
        self._copy_between_selection_lines(include_equation_context=False)

    def toggle_bookmarks_panel(self) -> None:
        visible = self.bookmarks_dock.isVisible()
        if visible:
            self._hide_bookmarks_panel()
        else:
            self._show_bookmarks_panel()

    def _hide_bookmarks_panel(self) -> None:
        state = self._layout_change_view_state()
        self._suspend_view_updates = True
        try:
            self.bookmarks_dock.hide()
            self._restore_layout_change_view_state(state)
        finally:
            self._suspend_view_updates = False
        QTimer.singleShot(0, lambda saved_state=state: self._restore_layout_change_view_state(saved_state))

    def _show_bookmarks_panel(self) -> None:
        state = self._layout_change_view_state()
        self._suspend_view_updates = True
        try:
            self._refresh_bookmarks_panel()
            self.bookmarks_dock.show()
            self.bookmarks_dock.raise_()
            self._restore_layout_change_view_state(state)
        finally:
            self._suspend_view_updates = False
        QTimer.singleShot(0, lambda saved_state=state: self._ensure_bookmarks_dock_width(saved_state))

    def _ensure_bookmarks_dock_width(self, state: Optional[ViewState] = None) -> None:
        state = state or self._layout_change_view_state()
        self._suspend_view_updates = True
        try:
            if self.bookmarks_dock.isVisible() and self.bookmarks_dock.width() < BOOKMARKS_DOCK_MIN_WIDTH:
                self.resizeDocks(
                    [self.bookmarks_dock],
                    [BOOKMARKS_DOCK_DEFAULT_WIDTH],
                    Qt.Horizontal,
                )
            self._restore_layout_change_view_state(state)
        finally:
            self._suspend_view_updates = False

    def _layout_change_view_state(self) -> Optional[ViewState]:
        session = self.current_session()
        if session is None or not self._view_has_session(session):
            return None
        return ViewState(
            page=session.current_page,
            scroll_x=self.view.horizontalScrollBar().value(),
            scroll_y=self.view.verticalScrollBar().value(),
            zoom=self.view.zoom_factor(),
        )

    def _restore_layout_change_view_state(self, state: Optional[ViewState]) -> None:
        session = self.current_session()
        if state is None or session is None or not self._view_has_session(session):
            return
        was_suspended = self._suspend_view_updates
        self._suspend_view_updates = True
        try:
            self.view.set_zoom_factor(state.zoom)
            self.view.horizontalScrollBar().setValue(state.scroll_x)
            self.view.verticalScrollBar().setValue(state.scroll_y)
            session.current_page = state.page
            self.page_input.blockSignals(True)
            self.page_input.setValue(state.page)
            self.page_input.blockSignals(False)
            self._update_status()
        finally:
            self._suspend_view_updates = was_suspended

    def add_bookmark(self) -> None:
        session = self.current_session()
        if session is None:
            return

        default_title = f"Page {session.current_page}"
        title, ok = QInputDialog.getText(
            self,
            "Add Bookmark",
            "Bookmark title:",
            text=default_title,
        )
        if not ok:
            return

        title = title.strip() or default_title
        session.user_bookmarks.append(UserBookmark(title=title, page=session.current_page))
        self._refresh_bookmarks_panel()
        self._save_session_state_now()
        self.statusBar().showMessage(f"Added bookmark '{title}' on page {session.current_page}", 4000)

    def _get_pdf_toc(self, session: DocumentSession) -> List[list]:
        try:
            return session.doc.get_toc(simple=True)
        except Exception:
            return []

    def _refresh_bookmarks_panel(self) -> None:
        self.bookmarks_tree.clear()
        session = self.current_session()
        if session is None:
            return

        pdf_root = QTreeWidgetItem(["PDF bookmarks"])
        user_root = QTreeWidgetItem(["Added bookmarks"])
        self.bookmarks_tree.addTopLevelItem(pdf_root)
        self.bookmarks_tree.addTopLevelItem(user_root)

        toc = self._get_pdf_toc(session)
        stack: List[QTreeWidgetItem] = [pdf_root]

        for entry in toc:
            if len(entry) < 3:
                continue
            level, title, page = entry[0], str(entry[1]), int(entry[2])
            level = max(1, level)

            while len(stack) > level:
                stack.pop()

            parent = stack[-1]
            item = QTreeWidgetItem([title])
            item.setData(0, Qt.UserRole, page)
            parent.addChild(item)
            stack.append(item)

        for bm in session.user_bookmarks:
            item = QTreeWidgetItem([f"{bm.title} (p. {bm.page})"])
            item.setData(0, Qt.UserRole, bm.page)
            user_root.addChild(item)

        pdf_root.setExpanded(True)
        user_root.setExpanded(True)
        self.bookmarks_tree.expandToDepth(1)

    def _bookmark_item_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        session = self.current_session()
        if session is None:
            return

        page = item.data(0, Qt.UserRole)
        if page is None:
            return

        try:
            page_num = int(page)
        except Exception:
            return

        self._show_page_for_session(session, page_num, push_history=True)

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

    @classmethod
    def _deserialize_view_states(cls, entries: object) -> List[ViewState]:
        if not isinstance(entries, list):
            return []
        out: List[ViewState] = []
        for entry in entries[:3]:
            if not isinstance(entry, dict):
                continue
            try:
                out.append(cls._deserialize_view_state(entry))
            except Exception:
                continue
        return out

    @staticmethod
    def _serialize_selection_lines(lines: List[SelectionLine]) -> List[dict]:
        return [{"page": line.page, "y": line.y} for line in lines]

    @staticmethod
    def _deserialize_selection_lines(entries: List[dict]) -> List[SelectionLine]:
        out: List[SelectionLine] = []
        for entry in entries:
            try:
                page = max(1, int(entry.get("page", 1)))
                y = float(entry.get("y", 0.0))
                out.append(SelectionLine(page=page, y=y))
            except Exception:
                continue
        return out

    @staticmethod
    def _serialize_covers(covers: List[PdfCover]) -> List[dict]:
        return [
            {
                "page": cover.page,
                "rect": list(cover.rect),
                "color": cover.color,
            }
            for cover in covers
        ]

    @staticmethod
    def _deserialize_covers(entries: object) -> List[PdfCover]:
        if not isinstance(entries, list):
            return []
        out: List[PdfCover] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                page = max(1, int(entry.get("page", 1)))
                raw_rect = entry.get("rect", [])
                if not isinstance(raw_rect, list) or len(raw_rect) != 4:
                    continue
                rect = tuple(float(value) for value in raw_rect)
                color = str(entry.get("color", "#FFFFFF"))
            except Exception:
                continue
            out.append(PdfCover(page=page, rect=rect, color=color))
        return out

    @staticmethod
    def _serialize_user_bookmarks(bookmarks: List[UserBookmark]) -> List[dict]:
        return [{"title": bm.title, "page": bm.page} for bm in bookmarks]

    @staticmethod
    def _deserialize_user_bookmarks(entries: List[dict]) -> List[UserBookmark]:
        out: List[UserBookmark] = []
        for entry in entries:
            try:
                title = str(entry.get("title", "")).strip() or "Bookmark"
                page = max(1, int(entry.get("page", 1)))
                out.append(UserBookmark(title=title, page=page))
            except Exception:
                continue
        return out

    @staticmethod
    def _serialize_equation_index(index: Dict[str, EquationLocation]) -> Dict[str, dict]:
        return {
            eq_id: {"page": location.page, "y": location.y}
            for eq_id, location in sorted(index.items())
        }

    @staticmethod
    def _deserialize_equation_index(entries: dict) -> Dict[str, EquationLocation]:
        out: Dict[str, EquationLocation] = {}
        if not isinstance(entries, dict):
            return out
        for raw_eq_id, raw_location in entries.items():
            if not isinstance(raw_location, dict):
                continue
            try:
                eq_id = PdfReaderWindow._normalize_equation_id(str(raw_eq_id))
                page = max(1, int(raw_location.get("page", 1)))
                y = max(0.0, float(raw_location.get("y", 0.0)))
                out[eq_id] = EquationLocation(page=page, y=y)
            except Exception:
                continue
        return out

    @staticmethod
    def _serialize_manual_equation_samples(samples: List[EquationCandidate]) -> List[dict]:
        return [
            {
                "eq_id": sample.eq_id,
                "raw_text": sample.raw_text,
                "page": sample.location.page,
                "y": sample.location.y,
            }
            for sample in samples[-MAX_MANUAL_EQUATION_SAMPLES:]
        ]

    @staticmethod
    def _deserialize_manual_equation_samples(entries: object) -> List[EquationCandidate]:
        if not isinstance(entries, list):
            return []

        out: List[EquationCandidate] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                eq_id = PdfReaderWindow._normalize_equation_id(str(entry.get("eq_id", "")))
                raw_text = str(entry.get("raw_text", "")).strip() or eq_id
                page = max(1, int(entry.get("page", 1)))
                y = max(0.0, float(entry.get("y", 0.0)))
            except Exception:
                continue
            if len(eq_id.split(".")) not in {2, 3}:
                continue
            out.append(
                EquationCandidate(
                    eq_id=eq_id,
                    raw_text=raw_text,
                    location=EquationLocation(page=page, y=y),
                )
            )
        return out[-MAX_MANUAL_EQUATION_SAMPLES:]

    @staticmethod
    def _deserialize_equation_patterns(entries: dict) -> Dict[int, str]:
        out: Dict[int, str] = {}
        if not isinstance(entries, dict):
            return out
        for raw_key, raw_value in entries.items():
            try:
                key = int(raw_key)
                value = str(raw_value)
            except Exception:
                continue
            if key in {2, 3} and value:
                out[key] = value
        return out

    @staticmethod
    def _deserialize_equation_format(entry: object) -> Dict[str, object]:
        if not isinstance(entry, dict):
            return {}

        mode = str(entry.get("numbering_mode", EQUATION_MODE_SECTION))
        if mode not in {EQUATION_MODE_SECTION, EQUATION_MODE_CHAPTER}:
            mode = EQUATION_MODE_SECTION

        out: Dict[str, object] = {
            "version": int(entry.get("version", EQUATION_FORMAT_VERSION) or EQUATION_FORMAT_VERSION),
            "numbering_mode": mode,
            "source": str(entry.get("source", "")) or "saved",
        }
        for key in ("part_counts", "patterns", "examples"):
            value = entry.get(key)
            if isinstance(value, dict):
                out[key] = dict(value)
        try:
            out["sample_count"] = max(0, int(entry.get("sample_count", 0)))
        except Exception:
            out["sample_count"] = 0
        return out

    def _save_session_state_now(self) -> None:
        try:
            session_data = self._build_session_persistence()
            chapters = self.default_chapter_starts or {}
            write_a_txt(self.a_txt_path, chapters, session_data, self.a_txt_preserved_lines)
        except Exception as exc:
            QMessageBox.warning(self, "Save warning", f"Could not save session state to a.txt:\n{exc}")

    def _session_persistence_entry(self, session: DocumentSession) -> dict:
        state = session.saved_view_state or self._default_state_for_session(session)
        viewer_states = session.viewer_states or [state]
        active_view_index = max(0, min(session.active_view_index, len(viewer_states) - 1))
        return {
            "path": session.path,
            "file_identity": self._pdf_identity_for_path(session.path) or {},
            "view_state": self._serialize_view_state(state),
            "viewer_states": [self._serialize_view_state(v) for v in viewer_states[:3]],
            "active_view_index": active_view_index,
            "history_back": [self._serialize_view_state(v) for v in session.history_back],
            "history_forward": [self._serialize_view_state(v) for v in session.history_forward],
            "user_bookmarks": self._serialize_user_bookmarks(session.user_bookmarks),
            "selection_lines": self._serialize_selection_lines(session.selection_lines),
            "covers": self._serialize_covers(session.covers),
            "last_equation_return_state": (
                self._serialize_view_state(session.last_equation_return_state)
                if session.last_equation_return_state is not None
                else None
            ),
            "equation_numbering_mode": session.equation_numbering_mode,
            "equation_format": dict(session.equation_format),
            "equation_lookup_mode": session.equation_lookup_mode,
            "equation_index_signature": session.equation_index_signature,
            "equation_index": self._serialize_equation_index(session.equation_index),
            "equation_rough_index": self._serialize_equation_index(session.equation_rough_index),
            "equation_pattern_index": self._serialize_equation_index(session.equation_pattern_index),
            "equation_index_patterns": {
                str(part_count): pattern
                for part_count, pattern in sorted(session.equation_index_patterns.items())
            },
            "manual_equation_samples": self._serialize_manual_equation_samples(
                session.manual_equation_samples
            ),
        }

    def _build_session_persistence(self) -> SessionPersistence:
        self._save_active_session_view()

        docs_data: List[dict] = []
        for session in self.sessions:
            entry = self._session_persistence_entry(session)
            docs_data.append(entry)
            self.recent_document_entries[self._path_key(session.path)] = entry

        recent_entries: List[dict] = []
        for path in self.recent_documents:
            entry = self.recent_document_entries.get(self._path_key(path))
            if entry:
                recent_entries.append(self._strip_comment_metadata(entry))
            else:
                recent_entries.append({"path": path})

        active_index = max(0, self.current_index) if self.sessions else 0
        return SessionPersistence(
            active_index=active_index,
            documents=docs_data,
            recent_documents=recent_entries,
            equation_macros=dict(self.equation_macros),
            shortcut_overrides=dict(self.shortcut_overrides),
        )

    def _load_pdf_from_persistence_entry(
        self,
        doc_entry: dict,
        switch_to: bool,
        update_recent: bool,
    ) -> bool:
        path = doc_entry.get("path")
        if not path:
            return False
        resolved_path = self._resolve_document_path(path, doc_entry)
        if not resolved_path:
            return False
        if resolved_path != path:
            self._replace_recent_document_path(path, resolved_path, doc_entry)
            doc_entry = dict(doc_entry)
            doc_entry["path"] = resolved_path
            identity = self._pdf_identity_for_path(resolved_path)
            if identity:
                doc_entry["file_identity"] = identity
            path = resolved_path

        try:
            restored_state = self._deserialize_view_state(doc_entry.get("view_state", {}))
            restored_viewer_states = self._deserialize_view_states(doc_entry.get("viewer_states", []))
            if not restored_viewer_states:
                restored_viewer_states = [restored_state]
            try:
                restored_active_view_index = max(0, int(doc_entry.get("active_view_index", 0)))
            except Exception:
                restored_active_view_index = 0
            restored_back = [
                self._deserialize_view_state(v)
                for v in doc_entry.get("history_back", [])
            ]
            restored_forward = [
                self._deserialize_view_state(v)
                for v in doc_entry.get("history_forward", [])
            ]
            restored_user_bookmarks = self._deserialize_user_bookmarks(
                doc_entry.get("user_bookmarks", [])
            )
            restored_selection_lines = self._deserialize_selection_lines(
                doc_entry.get("selection_lines", [])
            )
            restored_comments = []
            restored_covers = self._deserialize_covers(
                doc_entry.get("covers", [])
            )
            raw_return_state = doc_entry.get("last_equation_return_state")
            restored_last_equation_return_state = (
                self._deserialize_view_state(raw_return_state)
                if isinstance(raw_return_state, dict)
                else None
            )
            restored_equation_numbering_mode = str(
                doc_entry.get("equation_numbering_mode", EQUATION_MODE_SECTION)
            )
            if restored_equation_numbering_mode not in {EQUATION_MODE_SECTION, EQUATION_MODE_CHAPTER}:
                restored_equation_numbering_mode = EQUATION_MODE_SECTION
            restored_equation_format = self._deserialize_equation_format(
                doc_entry.get("equation_format", {})
            )
            saved_format_mode = str(restored_equation_format.get("numbering_mode", ""))
            if saved_format_mode in {EQUATION_MODE_SECTION, EQUATION_MODE_CHAPTER}:
                restored_equation_numbering_mode = saved_format_mode
            restored_equation_lookup_mode = str(
                doc_entry.get("equation_lookup_mode", EQUATION_LOOKUP_SCAN)
            )
            if restored_equation_lookup_mode == "index":
                restored_equation_lookup_mode = EQUATION_LOOKUP_ROUGH_INDEX
            if restored_equation_lookup_mode not in EQUATION_LOOKUP_MODES:
                restored_equation_lookup_mode = EQUATION_LOOKUP_SCAN
            restored_equation_index = self._deserialize_equation_index(
                doc_entry.get("equation_index", {})
            )
            restored_equation_rough_index = self._deserialize_equation_index(
                doc_entry.get("equation_rough_index", {})
            ) or dict(restored_equation_index)
            restored_equation_pattern_index = self._deserialize_equation_index(
                doc_entry.get("equation_pattern_index", {})
            )
            restored_equation_index_patterns = self._deserialize_equation_patterns(
                doc_entry.get("equation_index_patterns", {})
            )
            restored_manual_equation_samples = self._deserialize_manual_equation_samples(
                doc_entry.get("manual_equation_samples", [])
            )
            raw_signature = doc_entry.get("equation_index_signature")
            restored_equation_index_signature = raw_signature if isinstance(raw_signature, dict) else None
        except Exception:
            restored_state = ViewState(page=1, scroll_x=0, scroll_y=0, zoom=1.0)
            restored_viewer_states = [restored_state]
            restored_active_view_index = 0
            restored_back = []
            restored_forward = []
            restored_user_bookmarks = []
            restored_selection_lines = []
            restored_comments = []
            restored_covers = []
            restored_last_equation_return_state = None
            restored_equation_numbering_mode = EQUATION_MODE_SECTION
            restored_equation_lookup_mode = EQUATION_LOOKUP_SCAN
            restored_equation_index = {}
            restored_equation_rough_index = {}
            restored_equation_pattern_index = {}
            restored_equation_index_patterns = {}
            restored_equation_format = {}
            restored_manual_equation_samples = []
            restored_equation_index_signature = None

        self.load_pdf(
            path,
            switch_to=switch_to,
            restored_state=restored_state,
            restored_back=restored_back,
            restored_forward=restored_forward,
            restored_viewer_states=restored_viewer_states,
            restored_active_view_index=restored_active_view_index,
            restored_user_bookmarks=restored_user_bookmarks,
            restored_selection_lines=restored_selection_lines,
            restored_comments=restored_comments,
            restored_covers=restored_covers,
            restored_last_equation_return_state=restored_last_equation_return_state,
            restored_equation_numbering_mode=restored_equation_numbering_mode,
            restored_equation_index=restored_equation_index,
            restored_equation_rough_index=restored_equation_rough_index,
            restored_equation_pattern_index=restored_equation_pattern_index,
            restored_equation_index_patterns=restored_equation_index_patterns,
            restored_equation_format=restored_equation_format,
            restored_manual_equation_samples=restored_manual_equation_samples,
            restored_equation_index_signature=restored_equation_index_signature,
            restored_equation_lookup_mode=restored_equation_lookup_mode,
            update_recent=update_recent,
        )
        return True

    def _restore_startup_session(self) -> None:
        restored_any = False

        if self.startup_session_data:
            for doc_entry in self.startup_session_data.documents:
                path = doc_entry.get("path")
                if not path:
                    continue
                resolved_path = self._resolve_document_path(path, doc_entry)
                if not resolved_path:
                    continue
                if resolved_path != path:
                    self._replace_recent_document_path(path, resolved_path, doc_entry)
                    doc_entry = dict(doc_entry)
                    doc_entry["path"] = resolved_path
                    identity = self._pdf_identity_for_path(resolved_path)
                    if identity:
                        doc_entry["file_identity"] = identity
                    path = resolved_path

                try:
                    restored_state = self._deserialize_view_state(doc_entry.get("view_state", {}))
                    restored_viewer_states = self._deserialize_view_states(doc_entry.get("viewer_states", []))
                    if not restored_viewer_states:
                        restored_viewer_states = [restored_state]
                    try:
                        restored_active_view_index = max(0, int(doc_entry.get("active_view_index", 0)))
                    except Exception:
                        restored_active_view_index = 0
                    restored_back = [
                        self._deserialize_view_state(v)
                        for v in doc_entry.get("history_back", [])
                    ]
                    restored_forward = [
                        self._deserialize_view_state(v)
                        for v in doc_entry.get("history_forward", [])
                    ]
                    restored_user_bookmarks = self._deserialize_user_bookmarks(
                        doc_entry.get("user_bookmarks", [])
                    )
                    restored_selection_lines = self._deserialize_selection_lines(
                        doc_entry.get("selection_lines", [])
                    )
                    restored_comments = []
                    restored_covers = self._deserialize_covers(
                        doc_entry.get("covers", [])
                    )
                    raw_return_state = doc_entry.get("last_equation_return_state")
                    restored_last_equation_return_state = (
                        self._deserialize_view_state(raw_return_state)
                        if isinstance(raw_return_state, dict)
                        else None
                    )
                    restored_equation_numbering_mode = str(
                        doc_entry.get("equation_numbering_mode", EQUATION_MODE_SECTION)
                    )
                    if restored_equation_numbering_mode not in {EQUATION_MODE_SECTION, EQUATION_MODE_CHAPTER}:
                        restored_equation_numbering_mode = EQUATION_MODE_SECTION
                    restored_equation_format = self._deserialize_equation_format(
                        doc_entry.get("equation_format", {})
                    )
                    saved_format_mode = str(restored_equation_format.get("numbering_mode", ""))
                    if saved_format_mode in {EQUATION_MODE_SECTION, EQUATION_MODE_CHAPTER}:
                        restored_equation_numbering_mode = saved_format_mode
                    restored_equation_lookup_mode = str(
                        doc_entry.get("equation_lookup_mode", EQUATION_LOOKUP_SCAN)
                    )
                    if restored_equation_lookup_mode == "index":
                        restored_equation_lookup_mode = EQUATION_LOOKUP_ROUGH_INDEX
                    if restored_equation_lookup_mode not in EQUATION_LOOKUP_MODES:
                        restored_equation_lookup_mode = EQUATION_LOOKUP_SCAN
                    restored_equation_index = self._deserialize_equation_index(
                        doc_entry.get("equation_index", {})
                    )
                    restored_equation_rough_index = self._deserialize_equation_index(
                        doc_entry.get("equation_rough_index", {})
                    ) or dict(restored_equation_index)
                    restored_equation_pattern_index = self._deserialize_equation_index(
                        doc_entry.get("equation_pattern_index", {})
                    )
                    restored_equation_index_patterns = self._deserialize_equation_patterns(
                        doc_entry.get("equation_index_patterns", {})
                    )
                    restored_manual_equation_samples = self._deserialize_manual_equation_samples(
                        doc_entry.get("manual_equation_samples", [])
                    )
                    raw_signature = doc_entry.get("equation_index_signature")
                    restored_equation_index_signature = raw_signature if isinstance(raw_signature, dict) else None
                except Exception:
                    restored_state = ViewState(page=1, scroll_x=0, scroll_y=0, zoom=1.0)
                    restored_viewer_states = [restored_state]
                    restored_active_view_index = 0
                    restored_back = []
                    restored_forward = []
                    restored_user_bookmarks = []
                    restored_selection_lines = []
                    restored_comments = []
                    restored_covers = []
                    restored_last_equation_return_state = None
                    restored_equation_numbering_mode = EQUATION_MODE_SECTION
                    restored_equation_lookup_mode = EQUATION_LOOKUP_SCAN
                    restored_equation_index = {}
                    restored_equation_rough_index = {}
                    restored_equation_pattern_index = {}
                    restored_equation_index_patterns = {}
                    restored_equation_format = {}
                    restored_manual_equation_samples = []
                    restored_equation_index_signature = None

                self.load_pdf(
                    path,
                    switch_to=False,
                    restored_state=restored_state,
                    restored_back=restored_back,
                    restored_forward=restored_forward,
                    restored_viewer_states=restored_viewer_states,
                    restored_active_view_index=restored_active_view_index,
                    restored_user_bookmarks=restored_user_bookmarks,
                    restored_selection_lines=restored_selection_lines,
                    restored_comments=restored_comments,
                    restored_covers=restored_covers,
                    restored_last_equation_return_state=restored_last_equation_return_state,
                    restored_equation_numbering_mode=restored_equation_numbering_mode,
                    restored_equation_index=restored_equation_index,
                    restored_equation_rough_index=restored_equation_rough_index,
                    restored_equation_pattern_index=restored_equation_pattern_index,
                    restored_equation_index_patterns=restored_equation_index_patterns,
                    restored_equation_format=restored_equation_format,
                    restored_manual_equation_samples=restored_manual_equation_samples,
                    restored_equation_index_signature=restored_equation_index_signature,
                    restored_equation_lookup_mode=restored_equation_lookup_mode,
                    update_recent=False,
                )
                restored_any = True

            if restored_any and self.sessions:
                idx = min(max(0, self.startup_session_data.active_index), len(self.sessions) - 1)
                self.doc_tabs.setCurrentIndex(idx)
                self._switch_to_document(idx)

        if not restored_any:
            self._set_no_document_ui()

    def _sessions_with_pending_pdf_changes(self) -> List[DocumentSession]:
        return [
            session
            for session in self.sessions
            if session.comments
            or session.covers
            or session.user_bookmarks
            or session.saved_comment_edits
            or session.saved_comment_removals
        ]

    def _confirm_close_with_pending_pdf_changes(self) -> bool:
        pending = self._sessions_with_pending_pdf_changes()
        if not pending:
            return True

        comment_count = sum(len(session.comments) for session in pending)
        cover_count = sum(len(session.covers) for session in pending)
        bookmark_count = sum(len(session.user_bookmarks) for session in pending)
        edit_count = sum(len(session.saved_comment_edits) for session in pending)
        removal_count = sum(len(session.saved_comment_removals) for session in pending)
        document_count = len(pending)

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle("Unsaved PDF Changes")
        dialog.setText(
            f"There are pending PDF changes in {document_count} document(s): "
            f"{comment_count} new comment(s), {edit_count} edited comment(s), "
            f"{removal_count} removed comment(s), {cover_count} rectangle(s), "
            f"and {bookmark_count} bookmark(s)."
        )
        dialog.setInformativeText("Save them into the PDF before quitting?")
        save_button = dialog.addButton("Save Changes", QMessageBox.AcceptRole)
        discard_button = dialog.addButton("Discard", QMessageBox.DestructiveRole)
        cancel_button = dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(save_button)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked == cancel_button:
            return False
        if clicked == discard_button:
            return True
        if clicked != save_button:
            return False

        for session in list(pending):
            if not (
                session.comments
                or session.covers
                or session.user_bookmarks
                or session.saved_comment_edits
                or session.saved_comment_removals
            ):
                continue
            try:
                index = self.sessions.index(session)
            except ValueError:
                continue
            self.doc_tabs.setCurrentIndex(index)
            if self.current_index != index:
                self._switch_to_document(index)
            self.save_current_document_changes()
            if (
                session.comments
                or session.covers
                or session.user_bookmarks
                or session.saved_comment_edits
                or session.saved_comment_removals
            ):
                return False
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_close_with_pending_pdf_changes():
            event.ignore()
            return

        self._save_session_state_now()

        for session in self.sessions:
            try:
                session.doc.close()
            except Exception:
                pass

        super().closeEvent(event)
