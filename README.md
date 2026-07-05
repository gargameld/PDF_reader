# PDF Reader

A PySide6/PyMuPDF desktop PDF reader focused on fast textbook navigation:

- Open multiple PDFs in tabs.
- Jump between equation references and equation definitions.
- Build and use equation indexes for large books.
- Search text across a PDF.
- Add bookmarks, comments, and selection lines.
- Create a new PDF from selected PDF pages and images.
- Browse Google Drive PDFs/Docs and lazily download cloud documents when opened.

## Requirements

- Python 3.11 or newer
- PyMuPDF
- PySide6

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

On Windows, use:

```powershell
py -m pip install -r requirements.txt
```

## Run From Source

```bash
python3 PDF_reader.py
```

On Windows:

```powershell
py PDF_reader.py
```

## Equation Lookup

The reader can detect equation references from the PDF text layer near the click coordinate, then jump to the matching equation. This works best for searchable PDFs with a real text layer.

For scanned/image-only PDFs, text-layer lookup will not work. The OCR fallback currently uses Apple's Swift/Vision framework on macOS only.

Use `Ctrl+D` to scan/rebuild the equation index for the current book.

## Google Drive

Users can click `Sign In Drive` in the app menu and sign in with their own Google account. The app uses one bundled OAuth client ID, but each user authorizes and sees only their own Drive files.

Cloud documents are loaded from a Google Drive folder named `PDF Reader`. The app can create that folder and upload local PDFs into it from the `Upload Cloud` button, or by right-clicking a local document in the menu and choosing `Upload to Cloud`. You can also place PDFs, Google Docs, Sheets, or Slides there to show them in the app.

For development, `google_oauth_client.json` next to the app or `PDF_READER_GOOGLE_CLIENT_ID` / `PDF_READER_GOOGLE_CLIENT_SECRET` environment variables override the bundled client.

The app requests Drive read access plus file access for documents it creates/uploads. Cloud documents are listed from Drive metadata; the PDF file is cached locally only after the document is opened.

## Windows Compatibility

The main application uses cross-platform libraries:

- PySide6 for the UI
- PyMuPDF for PDF rendering, text extraction, and search

The application can run on Windows as long as the PDF is searchable or the workflow does not require the macOS-only OCR fallback.

To build a portable Windows app folder on Windows 10 or Windows 11:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt pyinstaller
py -m PyInstaller --noconfirm --clean "PDF Reader Windows.spec"
```

The output folder will be:

```text
dist\PDF Reader
```

For installer instructions, see [WINDOWS_BUILD.md](WINDOWS_BUILD.md).

## Project Structure

```text
PDF_reader.py                 # Thin launcher
pdf_reader_app/
  application.py              # QApplication subclass and file-open events
  constants.py                # Shortcuts, regexes, tuning constants
  main.py                     # App startup
  mime.py                     # Drag/drop path extraction
  models.py                   # Dataclasses and domain state
  session.py                  # a.txt/session/chapter/TOC helpers
  widgets.py                  # PdfView and dialogs
  window.py                   # Main window orchestration
```

## Notes

Local state files, generated snippets, build outputs, and personal PDF-derived artifacts are intentionally ignored by git.
