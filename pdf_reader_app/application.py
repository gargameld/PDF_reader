from pathlib import Path

from PySide6.QtCore import QEvent, Signal
from PySide6.QtWidgets import QApplication

class PdfReaderApplication(QApplication):
    file_open_requested = Signal(str)

    def event(self, event) -> bool:
        if event.type() == QEvent.FileOpen:
            path = event.file()
            if path and Path(path).suffix.lower() == ".pdf":
                self.file_open_requested.emit(path)
                return True
        return super().event(event)

