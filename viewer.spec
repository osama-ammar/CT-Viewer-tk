# volume_viewer.spec
block_cipher = None

a = Analysis(
    ['viewer.py'],  # Your main script
    pathex=[],
    binaries=[],
    datas=[],  # Add data files if needed (e.g., icons, configs)
    hiddenimports=[
        'pydicom',
        'numpy',
        'PIL',
        'pyvista',
        'nrrd',
        'sv_ttk',
        'utilities'  # If you have a custom utilities.py
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VolumeViewer',  # Name of the .exe
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress executable (recommended)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for no console window
    icon='icon.ico',  # Optional: Add an .ico file for the app icon
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)