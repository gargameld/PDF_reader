# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['PDF_reader.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PDF Reader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PDF Reader',
)
app = BUNDLE(
    coll,
    name='PDF Reader.app',
    icon=None,
    bundle_identifier='com.yotam.pdfreader',
    info_plist={
        'CFBundleName': 'PDF Reader',
        'CFBundleDisplayName': 'PDF Reader',
        'CFBundleIdentifier': 'com.yotam.pdfreader',
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'PDF Document',
                'CFBundleTypeRole': 'Viewer',
                'CFBundleTypeExtensions': ['pdf'],
                'CFBundleTypeMIMETypes': ['application/pdf'],
                'LSHandlerRank': 'Owner',
            },
        ],
    },
)
