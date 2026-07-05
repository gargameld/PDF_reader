import re

from PySide6.QtCore import Qt

# ============================================================
# Shortcuts and tweakable constants
# ============================================================

SHORTCUT_OPEN = "Meta+O"
SHORTCUT_CLOSE_DOCUMENT = "Meta+W"
SHORTCUT_BACK = "Meta+Left"
SHORTCUT_FORWARD = "Meta+Right"
SHORTCUT_PREVIOUS_PAGE = "Alt+Up"
SHORTCUT_NEXT_PAGE = "Alt+Down"
SHORTCUT_ZOOM_IN = "Meta++"
SHORTCUT_ZOOM_OUT = "Meta+-"
SHORTCUT_JUMP_TO_EQUATION = "Meta+J"
SHORTCUT_TOGGLE_BOOKMARKS = "Meta+B"
SHORTCUT_ADD_BOOKMARK = "Meta+Shift+B"
SHORTCUT_RETURN_TO_EQUATION_SOURCE = "Meta+1"
SHORTCUT_COPY_BETWEEN_LINES = "Shift+C"
SHORTCUT_CLEAR_SELECTION_LINES = "Shift+X"
SHORTCUT_PREVIEW_EQUATION = "Shift+P"
SHORTCUT_OPEN_TEXT_SEARCH = "Shift+F"
SHORTCUT_SPLIT_VIEWER = "Shift+W"
SHORTCUT_CLOSE_VIEWER = "Shift+Q"
SHORTCUT_RETURN_TO_TEXT_SEARCH_SOURCE = "Ctrl+2"
SHORTCUT_SCAN_EQUATION_INDEX = "Ctrl+D"
SHORTCUT_LEARN_EQUATION_RECTANGLE = "Ctrl+R"
SHORTCUT_CREATE_COMMENT = "Ctrl+T"
SHORTCUT_TOGGLE_EQUATION_LOOKUP = "Ctrl+G"
SHORTCUT_TOGGLE_LAST_DOCUMENTS = ("Ctrl+`", "Ctrl+~")

SHORTCUT_DEFINITIONS = (
    ("menu", "Menu", "Meta+M"),
    ("open_pdfs", "Open PDFs", SHORTCUT_OPEN),
    ("toggle_last_documents", "Toggle Last Documents", SHORTCUT_TOGGLE_LAST_DOCUMENTS[0]),
    ("close_document", "Close Document", SHORTCUT_CLOSE_DOCUMENT),
    ("back", "Back", SHORTCUT_BACK),
    ("forward", "Forward", SHORTCUT_FORWARD),
    ("previous_page", "Previous Page", SHORTCUT_PREVIOUS_PAGE),
    ("next_page", "Next Page", SHORTCUT_NEXT_PAGE),
    ("zoom_in", "Zoom In", SHORTCUT_ZOOM_IN),
    ("zoom_out", "Zoom Out", SHORTCUT_ZOOM_OUT),
    ("jump_to_equation", "Jump to Equation", SHORTCUT_JUMP_TO_EQUATION),
    ("toggle_bookmarks", "Toggle Bookmarks", SHORTCUT_TOGGLE_BOOKMARKS),
    ("add_bookmark", "Add Bookmark", SHORTCUT_ADD_BOOKMARK),
    ("return_to_equation_source", "Return to Equation Source", SHORTCUT_RETURN_TO_EQUATION_SOURCE),
    ("copy_between_lines", "Copy Between Blue Lines", SHORTCUT_COPY_BETWEEN_LINES),
    ("clear_selection_lines", "Clear Blue Lines", SHORTCUT_CLEAR_SELECTION_LINES),
    ("preview_equation", "Preview Equation Under Cursor", SHORTCUT_PREVIEW_EQUATION),
    ("open_text_search", "Search Text", SHORTCUT_OPEN_TEXT_SEARCH),
    ("split_viewer", "Split Viewer", SHORTCUT_SPLIT_VIEWER),
    ("close_viewer", "Close Viewer", SHORTCUT_CLOSE_VIEWER),
    ("return_to_text_search_source", "Return to Text Search Source", SHORTCUT_RETURN_TO_TEXT_SEARCH_SOURCE),
    ("scan_equation_index", "Scan Equation Index", SHORTCUT_SCAN_EQUATION_INDEX),
    ("learn_equation_rectangle", "Learn Equation Rectangle", SHORTCUT_LEARN_EQUATION_RECTANGLE),
    ("create_comment", "Create Comment", SHORTCUT_CREATE_COMMENT),
    ("toggle_equation_lookup", "Toggle Equation Lookup", SHORTCUT_TOGGLE_EQUATION_LOOKUP),
)

KEYBOARD_LAYOUT_FALLBACKS = {
    # Hebrew keyboard layout characters in the same physical positions as US
    # QWERTY letters. This keeps shortcuts working when the active input
    # source is Hebrew and Qt reports the translated character instead of
    # Qt.Key_A through Qt.Key_Z.
    "/": "Q",
    "'": "W",
    "ק": "E",
    "ר": "R",
    "א": "T",
    "ט": "Y",
    "ו": "U",
    "ן": "I",
    "ם": "O",
    "פ": "P",
    "ש": "A",
    "ד": "S",
    "ג": "D",
    "כ": "F",
    "ע": "G",
    "י": "H",
    "ח": "J",
    "ל": "K",
    "ך": "L",
    "ז": "Z",
    "ס": "X",
    "ב": "C",
    "ה": "V",
    "נ": "B",
    "מ": "N",
    "צ": "M",
}

SHORTCUT_LETTER_KEYS = {
    "A": Qt.Key_A,
    "B": Qt.Key_B,
    "C": Qt.Key_C,
    "D": Qt.Key_D,
    "E": Qt.Key_E,
    "F": Qt.Key_F,
    "G": Qt.Key_G,
    "H": Qt.Key_H,
    "I": Qt.Key_I,
    "J": Qt.Key_J,
    "K": Qt.Key_K,
    "L": Qt.Key_L,
    "M": Qt.Key_M,
    "N": Qt.Key_N,
    "O": Qt.Key_O,
    "P": Qt.Key_P,
    "Q": Qt.Key_Q,
    "R": Qt.Key_R,
    "S": Qt.Key_S,
    "T": Qt.Key_T,
    "U": Qt.Key_U,
    "V": Qt.Key_V,
    "W": Qt.Key_W,
    "X": Qt.Key_X,
    "Y": Qt.Key_Y,
    "Z": Qt.Key_Z,
}

SHORTCUT_KEY_CODES = {
    **SHORTCUT_LETTER_KEYS,
    "1": Qt.Key_1,
    "2": Qt.Key_2,
    "`": Qt.Key_QuoteLeft,
    "~": Qt.Key_AsciiTilde,
    "-": Qt.Key_Minus,
    "+": Qt.Key_Plus,
    "LEFT": Qt.Key_Left,
    "RIGHT": Qt.Key_Right,
    "UP": Qt.Key_Up,
    "DOWN": Qt.Key_Down,
}

MACOS_NATIVE_VIRTUAL_KEYS = {
    "A": 0,
    "S": 1,
    "D": 2,
    "F": 3,
    "H": 4,
    "G": 5,
    "Z": 6,
    "X": 7,
    "C": 8,
    "V": 9,
    "B": 11,
    "Q": 12,
    "W": 13,
    "E": 14,
    "R": 15,
    "Y": 16,
    "T": 17,
    "O": 31,
    "U": 32,
    "I": 34,
    "P": 35,
    "L": 37,
    "J": 38,
    "K": 40,
    "N": 45,
    "M": 46,
    "`": 50,
    "1": 18,
    "2": 19,
    "-": 27,
    "+": 24,
}

CLICK_BOX_W = 60
CLICK_BOX_H = 25
PDF_TEXT_CLICK_RADIUS = 8.0
PDF_TEXT_LINE_Y_TOLERANCE = 3.0
PDF_TEXT_NEARBY_X_MARGIN = 85.0
PDF_TEXT_NEARBY_Y_MARGIN = 15.0
DEBUG_BOX_DURATION_MS = 700
MAX_HISTORY = 200
RENDER_SCALE = 1.8
PAGE_SPACING = 24
MOUSE_ZOOM_STEP = 1.2
LAZY_RENDER_RADIUS = 1

EQUATION_REF_RE = re.compile(r"\(?\d+(?:[.\s,\-]+\d+)+\)?")
TEXT_LAYER_EQUATION_REF_RE = re.compile(r"\(?\d+(?:[.,\-]\d+)+\)?")
FIGURE_REF_RE = re.compile(r"\b(?:fig\.?|figure)\s+(\d+(?:\.\d+)+)\b", re.IGNORECASE)
SESSION_MARKER = "[SESSION_STATE]"
EQUATION_MODE_SECTION = "section"
EQUATION_MODE_CHAPTER = "chapter"
EQUATION_LOOKUP_SCAN = "scan"
EQUATION_LOOKUP_ROUGH_INDEX = "rough_index"
EQUATION_LOOKUP_PATTERN_INDEX = "pattern_index"
EQUATION_LOOKUP_INDEX = EQUATION_LOOKUP_ROUGH_INDEX
EQUATION_LOOKUP_MODES = (
    EQUATION_LOOKUP_SCAN,
    EQUATION_LOOKUP_ROUGH_INDEX,
    EQUATION_LOOKUP_PATTERN_INDEX,
)
EQUATION_FORMAT_VERSION = 1
EQUATION_INDEX_ALGORITHM_VERSION = 2
MAX_MANUAL_EQUATION_SAMPLES = 10
RECTANGLE_SELECT_HOLD_MS = 450
RECTANGLE_SELECT_MIN_SIZE = 8.0
SUPPORTED_DOCUMENT_BUILDER_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".tif",
    ".tiff",
    ".webp",
}
