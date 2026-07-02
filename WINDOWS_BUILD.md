# Windows Build

Build this on Windows 10 or Windows 11. PyInstaller does not cross-build a
Windows executable correctly from macOS.

## Build the portable app folder

Install Python 3.11 or newer, then open PowerShell in this project folder:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt pyinstaller
py -m PyInstaller --noconfirm --clean "PDF Reader Windows.spec"
```

The built app folder will be:

```text
dist\PDF Reader
```

You can zip that folder and send it as a portable app, or make an installer.

## Build the installer

Install Inno Setup, then compile:

```powershell
iscc installer\PDFReader.iss
```

The installer will be created at:

```text
installer\Output\PDFReaderSetup.exe
```

## PDF defaults on Windows

The installer registers PDF Reader as an app that can open `.pdf` files, so it
should appear in Windows "Open with" and Default apps. Windows 10 and 11 do not
allow normal installers to silently force themselves as the default PDF reader;
the user needs to choose it in Settings.

## OCR note

Clicking a searchable PDF now tries the PDF text layer first, which works on
Windows through PyMuPDF. The built-in OCR fallback still uses Apple's
Swift/Vision framework on macOS, so scanned/image-only pages still need a
separate Windows OCR backend if text-layer extraction fails.
