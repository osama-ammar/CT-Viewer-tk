import PyInstaller.__main__

# python -m PyInstaller --onefile --windowed --icon="icon.ico" --add-data "utilities.py;."  --hidden-import "vtkmodules.all" --hidden-import "PyQt5.sip" --hidden-import "matplotlib.backends.backend_qt5agg" --name "CTViewer" main.py

# Configuration
app_name = "CTViewer"
main_script = "main.py"
icon_file = "icon.ico"
additional_data = [
    ("utilities.py", "."),  # (source, destination)
    ("assets/*", "assets"),  # Example for additional assets
]
hidden_imports = [
    "vtkmodules.all",
    "PyQt5.sip",
    "matplotlib.backends.backend_qt5agg",
]

# Build the PyInstaller command
params = [
    main_script,
    '--onefile',
    '--windowed',
    f'--icon={icon_file}',
    '--clean',
    '--noconfirm',
    f'--name={app_name}',
]

# Add additional data files
for src, dst in additional_data:
    params.append(f'--add-data={src};{dst}')

# Add hidden imports
for imp in hidden_imports:
    params.append(f'--hidden-import={imp}')

# Run PyInstaller
PyInstaller.__main__.run(params)