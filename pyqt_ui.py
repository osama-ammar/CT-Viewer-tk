import os
import sys
import vtk
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QPushButton
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class DICOMViewer(QMainWindow):
    def __init__(self):
        super(DICOMViewer, self).__init__()

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Create a button to open the DICOM folder
        self.open_folder_button = QPushButton('Open DICOM Folder', self.central_widget)
        self.open_folder_button.clicked.connect(self.openDICOMFolder)

        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.vtk_widget.setMinimumSize(800, 600)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.open_folder_button)
        self.layout.addWidget(self.vtk_widget)

        self.ren = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.ren)

        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Initialize variables for DICOM image series
        self.dicom_reader = None
        self.image_actor = None
        self.image_plane_widget = None

        self.show()

    def openDICOMFolder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select DICOM Folder')
        if folder_path:
            dicom_files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
            dicom_files.sort()

            # Initialize DICOM reader
            self.dicom_reader = vtk.vtkDICOMImageReader()
            self.dicom_reader.SetDirectoryName(folder_path)
            self.dicom_reader.Update()

            # Create a vtkImageActor for displaying the DICOM images
            self.image_actor = vtk.vtkImageActor()
            self.image_actor.SetInputData(self.dicom_reader.GetOutput())

            # Create a vtkImagePlaneWidget for interactive slicing
            self.image_plane_widget = vtk.vtkImagePlaneWidget()
            self.image_plane_widget.SetInteractor(self.iren)
            self.image_plane_widget.SetInputConnection(self.dicom_reader.GetOutputPort())
            self.image_plane_widget.SetPlaneOrientationToXAxes()
            self.image_plane_widget.SetSliceIndex(len(dicom_files) // 2)
            self.image_plane_widget.DisplayTextOn()
            self.image_plane_widget.SetPicker(vtk.vtkCellPicker())

            # Add actors to the renderer
            self.ren.AddActor(self.image_actor)
            self.ren.AddActor(self.image_plane_widget.GetPlaneProperty())
            self.ren.AddActor(self.image_plane_widget.GetSelectedPlaneProperty())

            # Set up the camera to display the entire image
            self.ren.ResetCamera()

            # Initialize the interactor
            self.iren.Initialize()

            # Start the rendering loop
            self.iren.Start()

def main():
    app = QApplication(sys.argv)
    window = DICOMViewer()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
