import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtSvg import QSvgWidget
from PIL import Image
import pydicom
import os
import pyvista as pv
import nrrd
import utilities


class VolumeViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Volume Viewer")
        self.setGeometry(100, 100, 1200, 900)

        # Initialize variables
        self.volume = None
        self.image = None
        self.current_slice_index = 0
        self.window_level = 400
        self.window_width = 2000
        self.view_mode = "axial"
        self.volume_type = "npy"
        self.unique_labels = None

        # Set up the main window
        self.setup_ui()

        # Apply custom styles
        self.apply_styles()

    def setup_ui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Control frame
        self.control_frame = QWidget()
        self.control_layout = QHBoxLayout(self.control_frame)
        self.main_layout.addWidget(self.control_frame)

        # Buttons for opening files (with SVG icons)
        self.open_volume_button = QPushButton("Open npy Volume")
        self.open_volume_button.setIcon(QIcon("icons/folder.svg"))  # SVG icon
        self.open_volume_button.clicked.connect(self.open_volume)
        self.control_layout.addWidget(self.open_volume_button)

        self.open_nrrd_button = QPushButton("Open nrrd/nii")
        self.open_nrrd_button.setIcon(QIcon("icons/folder.svg"))
        self.open_nrrd_button.clicked.connect(self.open_nrrd)
        self.control_layout.addWidget(self.open_nrrd_button)

        self.open_image_button = QPushButton("Open npy Image")
        self.open_image_button.setIcon(QIcon("icons/folder.svg"))
        self.open_image_button.clicked.connect(self.open_image)
        self.control_layout.addWidget(self.open_image_button)

        self.open_dicom_button = QPushButton("Open dicom")
        self.open_dicom_button.setIcon(QIcon("icons/folder.svg"))
        self.open_dicom_button.clicked.connect(self.open_dicom_case)
        self.control_layout.addWidget(self.open_dicom_button)

        self.export_stl_button = QPushButton("Export as STL")
        self.export_stl_button.setIcon(QIcon("icons/export.svg"))
        self.export_stl_button.clicked.connect(self.export_stl_vtk)
        self.control_layout.addWidget(self.export_stl_button)

        # Radio buttons for view modes
        self.view_mode_group = QButtonGroup()
        self.axial_radio = QRadioButton("Axial")
        self.axial_radio.setChecked(True)
        self.view_mode_group.addButton(self.axial_radio)
        self.control_layout.addWidget(self.axial_radio)

        self.sagittal_radio = QRadioButton("Sagittal")
        self.view_mode_group.addButton(self.sagittal_radio)
        self.control_layout.addWidget(self.sagittal_radio)

        self.coronal_radio = QRadioButton("Coronal")
        self.view_mode_group.addButton(self.coronal_radio)
        self.control_layout.addWidget(self.coronal_radio)

        self.show_3d_button = QPushButton("Show 3D")
        self.show_3d_button.setIcon(QIcon("icons/3d.svg"))
        self.show_3d_button.clicked.connect(self.open_3d_view)
        self.control_layout.addWidget(self.show_3d_button)

        # Display frame
        self.display_frame = QWidget()
        self.display_layout = QHBoxLayout(self.display_frame)
        self.main_layout.addWidget(self.display_frame)

        # Canvas for displaying images
        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        self.display_layout.addWidget(self.canvas, stretch=1)

        # Sliders frame
        self.slider_frame = QWidget()
        self.slider_layout = QVBoxLayout(self.slider_frame)
        self.display_layout.addWidget(self.slider_frame)

        # Sliders for WL, WW, and slice navigation
        self.wl_slider = QSlider(Qt.Vertical)
        self.wl_slider.setRange(-1000, 4000)
        self.wl_slider.setValue(self.window_level)
        self.wl_slider.valueChanged.connect(self.update_wl)
        self.slider_layout.addWidget(self.wl_slider)

        self.wl_label = QLabel(f"WL: {self.window_level}")
        self.slider_layout.addWidget(self.wl_label)

        self.ww_slider = QSlider(Qt.Vertical)
        self.ww_slider.setRange(1, 4000)
        self.ww_slider.setValue(self.window_width)
        self.ww_slider.valueChanged.connect(self.update_ww)
        self.slider_layout.addWidget(self.ww_slider)

        self.ww_label = QLabel(f"WW: {self.window_width}")
        self.slider_layout.addWidget(self.ww_label)

        self.slice_slider = QSlider(Qt.Vertical)
        self.slice_slider.setRange(0, 1)
        self.slice_slider.valueChanged.connect(self.update_slice)
        self.slider_layout.addWidget(self.slice_slider)

        self.slice_label = QLabel("Slice: 0")
        self.slider_layout.addWidget(self.slice_label)

        # Pixel value label
        self.pixel_value_label = QLabel("Pixel Value: ")
        self.main_layout.addWidget(self.pixel_value_label)

        # Connect radio buttons to update view
        self.axial_radio.toggled.connect(lambda: self.set_view_mode("axial"))
        self.sagittal_radio.toggled.connect(lambda: self.set_view_mode("sagittal"))
        self.coronal_radio.toggled.connect(lambda: self.set_view_mode("coronal"))

    def apply_styles(self):
        # Apply a modern dark theme using QSS
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d2d;
            }
            QWidget {
                color: #ffffff;
                font-size: 14px;
            }
            QPushButton {
                background-color: #444;
                color: #fff;
                border: 1px solid #555;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #666;
            }
            QSlider::groove:vertical {
                background: #444;
                width: 8px;
                border-radius: 4px;
            }
            QSlider::handle:vertical {
                background: #888;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QLabel {
                color: #fff;
            }
            QRadioButton {
                color: #fff;
            }
        """)

    def set_view_mode(self, mode):
        self.view_mode = mode
        self.update_slice_slider_range()
        self.update_slice(self.current_slice_index)

    def open_volume(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open npy Volume", "", "Numpy files (*.npy);;All Files (*)")
        if file_path:
            self.volume = np.load(file_path)
            self.current_slice_index = 0
            self.unique_labels = np.unique(self.volume)
            self.update_sliders()
            self.update_slice_slider_range()
            self.slice_slider.setValue(0)
            self.update_slice(0)

    def open_nrrd(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open nrrd/nii", "", "All Files (*)")
        if file_path:
            if file_path.endswith("nrrd"):
                self.volume_type = "nrrd"
                self.volume, _ = nrrd.read(file_path)
            elif ".nii" in file_path:
                self.volume, _ = utilities.load_nifti(file_path)
            self.volume = np.transpose(self.volume, (2, 1, 0))
            self.current_slice_index = 0
            self.unique_labels = np.unique(self.volume)
            self.update_sliders()
            self.update_slice_slider_range()
            self.slice_slider.setValue(0)
            self.update_slice(0)

    def open_dicom_case(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open dicom", "", "Dicom files (*.dcm)")
        if file_path:
            slices = []
            self.volume_type = "dicom"
            dicom_folder = os.path.dirname(os.path.abspath(file_path))
            for filename in os.listdir(dicom_folder):
                filepath = os.path.join(dicom_folder, filename)
                try:
                    dicom_file = pydicom.dcmread(filepath)
                    slices.append(dicom_file)
                except Exception as e:
                    print(f"Could not read {filepath}: {e}")
            slices.sort(key=lambda s: int(s.InstanceNumber))
            self.volume = np.stack([s.pixel_array for s in slices])
            self.current_slice_index = 0
            self.update_sliders()
            self.update_slice_slider_range()
            self.slice_slider.setValue(0)
            self.update_slice(0)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open npy Image", "", "Numpy files (*.npy);;All Files (*)")
        if file_path:
            self.image = np.load(file_path)
            self.current_slice_index = 0
            self.slice_slider.setRange(0, 1)
            self.slice_slider.setValue(0)
            self.update_slice(0)

    def update_sliders(self):
        if self.volume is not None:
            self.wl_slider.setRange(int(self.volume.min()), int(self.volume.max()))
            self.wl_slider.setValue(self.window_level)
            self.ww_slider.setRange(1, int(self.volume.max()))
            self.ww_slider.setValue(self.window_width)

    def update_slice_slider_range(self):
        if self.volume is not None:
            if self.view_mode == "axial":
                self.slice_slider.setRange(0, self.volume.shape[0] - 1)
            elif self.view_mode == "sagittal":
                self.slice_slider.setRange(0, self.volume.shape[1] - 1)
            elif self.view_mode == "coronal":
                self.slice_slider.setRange(0, self.volume.shape[2] - 1)

    def update_slice(self, value):
        self.current_slice_index = value
        self.slice_label.setText(f"Slice: {self.current_slice_index}")

        if self.image is not None:
            slice_to_show = self.image
        else:
            if self.view_mode == "axial":
                slice_to_show = self.volume[self.current_slice_index, :, :]
            elif self.view_mode == "sagittal":
                slice_to_show = self.volume[:, :, self.current_slice_index]
                slice_to_show = slice_to_show if self.volume_type == "dicom" else np.flipud(slice_to_show)
            elif self.view_mode == "coronal":
                slice_to_show = self.volume[:, self.current_slice_index, :]
                slice_to_show = slice_to_show if self.volume_type == "dicom" else np.flipud(slice_to_show)

        # Apply window level and width
        window_min = self.window_level - self.window_width / 2
        window_max = self.window_level + self.window_width / 2
        if window_min >= window_max:
            window_min = 0
            window_max = 255

        slice_to_show = np.clip(slice_to_show, window_min, window_max)
        slice_to_show = 255 * (slice_to_show - window_min) / (window_max - window_min)

        # Convert to QImage
        height, width = slice_to_show.shape
        qimage = QImage(slice_to_show.astype(np.uint8), width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # Scale pixmap to fit canvas
        canvas_size = self.canvas.size()
        scaled_pixmap = pixmap.scaled(canvas_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.canvas.setPixmap(scaled_pixmap)

    def update_wl(self, value):
        self.window_level = value
        self.wl_label.setText(f"WL: {self.window_level}")
        self.update_slice(self.current_slice_index)

    def update_ww(self, value):
        self.window_width = value
        self.ww_label.setText(f"WW: {self.window_width}")
        self.update_slice(self.current_slice_index)

    def export_stl_vtk(self):
        if self.volume is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Export as STL", "", "STL files (*.stl);;All Files (*)")
            if file_path:
                utilities.export_volume_as_stl_vtk(self.volume, file_path, self.window_level, self.window_width)

    def open_3d_view(self):
        if self.volume is not None:
            plotter = utilities.npy_to_pyvista(self.volume)
            plotter.show()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 12))  # Set a modern font
    viewer = VolumeViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()