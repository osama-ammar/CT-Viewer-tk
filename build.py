
# python -m PyInstaller --onefile --windowed --icon="icon.ico" --add-data "utilities.py;."  --hidden-import "vtkmodules.all" --hidden-import "PyQt5.sip" --hidden-import "matplotlib.backends.backend_qt5agg" --name "CTViewer" main.py
