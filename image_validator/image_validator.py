import os
import numpy as np
import vtk
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QFileDialog, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from pathlib import Path
import json
from panoramic_curve_utils import *

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.image_files = []
        self.current_image_index = 0
        self.panorama_positions=(None,None)

        # VTK setup
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # Image viewer setup
        self.image_viewer = vtk.vtkImageViewer2()
        self.image_viewer.SetRenderWindow(self.vtk_widget.GetRenderWindow())
        self.image_viewer.SetupInteractor(self.vtk_widget.GetRenderWindow().GetInteractor())


        # UI setup
        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(20, 20, 20, 20)

        # Title label
        self.title_label = QLabel("Image Validator")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #2E86C1;")
        self.layout.addWidget(self.title_label)

        # VTK widget
        self.layout.addWidget(self.vtk_widget)

        # Select folder button
        self.select_folder_button = QPushButton("Select Folder")
        self.select_folder_button.setFont(QFont("Arial", 12))
        self.select_folder_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3498DB;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            """
        )
        self.select_folder_button.clicked.connect(self.select_folder)
        self.layout.addWidget(self.select_folder_button)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                background: #D5D8DC;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498DB;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #3498DB;
                border-radius: 4px;
            }
            """
        )
        self.slider.valueChanged.connect(self.update_image)
        self.layout.addWidget(self.slider)

        # Save button
        self.save_button = QPushButton("Save Info")
        self.save_button.setFont(QFont("Arial", 12))
        self.save_button.setStyleSheet(
            """
            QPushButton {
                background-color: #27AE60;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            """
        )
        self.save_button.clicked.connect(self.save_current_image_name)
        self.layout.addWidget(self.save_button)

        # Set layout
        self.setLayout(self.layout)
        self.setWindowTitle("VTK Image Viewer")
        self.setStyleSheet("background-color: #F2F3F4;")  # Set background color for the main window


    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.npy'))]
            if self.image_files:
                self.slider.setMaximum(len(self.image_files) - 1)
                self.load_image(self.image_files[0])

    def load_image(self, image_path):
        #try:
        if image_path.lower().endswith('.npy'):
            # Load .npy file
            image_array = np.load(image_path)
            self.load_numpy_image(image_array)
        else:
            # Load .png or .jpg file
            reader = vtk.vtkPNGReader() if image_path.lower().endswith('.png') else vtk.vtkJPEGReader()
            reader.SetFileName(image_path)
            reader.Update()
            self.image_viewer.SetInputConnection(reader.GetOutputPort())
            self.image_viewer.Render()
        # except Exception as e:
        #     print(e)

            

    def normalize(self,image,mode="mask"):
        # the output is an image in the range [0, 1]
        # fixing clipping range ... better in metal cases
        if mode == "mask":
            min_val=np.min(image)
            max_val=np.max(image)
        else:
            min_val=-1000
            max_val = 4059
            
        
        image = np.clip(image,min_val,max_val)

        image = (image - min_val )/ (max_val-min_val)
        return image

    def load_numpy_image(self, image_array):
        # Convert NumPy array to VTK image
        #print(image_array.shape,np.unique(image_array))
        if len(image_array.shape) >2:
            image_array = np.amax(image_array,1)
            
        target_maxilla_z,target_mandible_z = get_max_z_from_cor_mask(image_array,"imae name ...")
        print(f"modified  target : {target_maxilla_z} - {target_mandible_z}")
        self.panorama_positions=(target_maxilla_z,target_mandible_z)
        current_image_path = self.image_files[self.current_image_index]
        current_image_name = os.path.basename(current_image_path)
        #print(current_image_name)
        
        image_array= self.normalize(image_array,mode="mask")*255
        #print(np.unique(image_array))
        
        image_array[target_maxilla_z,:]=255
        image_array[target_mandible_z,:]=255
    
        
        height, width = image_array.shape
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(width, height, 1)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # Copy data from NumPy array to VTK image
        for y in range(height):
            for x in range(width):
                vtk_image.SetScalarComponentFromDouble(x, y, 0, 0, image_array[y, x])

        # Set the VTK image to the viewer
        self.image_viewer.SetInputData(vtk_image)
        self.image_viewer.Render()
        #self.add_line_overlay(50,50,200,50)

    def add_line_overlay(self, x1, y1, x2, y2):
        """ Adds an overlay line on the image viewer """
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(x1, y1, 0)  # Start point
        line_source.SetPoint2(x2, y2, 0)  # End point

        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputConnection(line_source.GetOutputPort())

        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(1, 0, 0)  # Red color
        line_actor.GetProperty().SetLineWidth(12)    # Line thickness

        self.renderer.AddActor(line_actor)

    def update_image(self, value):
        if self.image_files:
            self.current_image_index = value
            self.load_image(self.image_files[self.current_image_index])

    def save_current_image_name(self):
        if self.image_files:
            current_image_path = self.image_files[self.current_image_index]
            current_image_path = Path(current_image_path)
            current_image_name = current_image_path.name.split(".")[0]

            save_path=os.path.join(current_image_path.parent,f"{current_image_name}.json")
            #print(current_image_path,save_path)
            
            json_data = {
                "name":current_image_name,
                "maxilla_z":float(self.panorama_positions[0]),
                "mandibble_z":float(self.panorama_positions[1])}
            
            if save_path:
                with open(save_path, "w") as file:
                    json.dump(json_data,file,indent=2)
                print(f"Saved current image name: {current_image_name} to {save_path}")


if __name__ == "__main__":
    app = QApplication([])
    window = ImageViewer()
    window.resize(1000, 800)  # Set initial window size
    window.show()
    app.exec_()
