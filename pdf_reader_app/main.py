import sys
from pathlib import Path

from .application import PdfReaderApplication
from .window import PdfReaderWindow

def main() -> int:
    app = PdfReaderApplication(sys.argv)
    window = PdfReaderWindow()
    app.file_open_requested.connect(lambda path: window.load_pdf(path, switch_to=True))

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
