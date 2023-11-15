import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from stl import mesh  # Make sure to install the `numpy-stl` library
import vtk
print (vtk.__version__)
from vtk.util import numpy_support


class VolumeViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Numpy Volume Viewer")

        # Set the initial dimensions of the window
        self.root.geometry("900x800")  # Adjust the dimensions as needed
        
        # Configure a dark background
        self.root.configure(bg='#333333')        
        
        
        # Initialize variables
        self.volume = None
        self_image = None 
        self.current_slice_index = 0
        self.window_level = 400  # Initial value, adjust as needed
        self.window_width = 1080  # Initial value, adjust as needed

        # Add a button to open a 3D Numpy volume file
        self.open_volume_button = tk.Button(root, text="Open Volume", command=self.open_volume, bg='#555555', fg='white')
        self.open_volume_button.pack(side=tk.TOP, padx=10, pady=10)


        # Add a button to open a Numpy image file
        self.open_image_button = tk.Button(root, text="Open Image", command=self.open_image, bg='#555555', fg='white')
        self.open_image_button.pack(side=tk.TOP, padx=10, pady=10)

        # Add a button to export the volume as an STL file
        self.export_stl_button = tk.Button(root, text="Export as STL", command=self.export_stl_vtk, bg='#555555', fg='white')
        self.export_stl_button.pack(side=tk.LEFT, padx=10, pady=10)


        # Create Tkinter Canvas
        self.canvas = tk.Canvas(root, bg='#222222')  # Set canvas background color
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Label to display pixel values
        self.pixel_value_label = tk.Label(root, text="Pixel Value: ")
        self.pixel_value_label.pack(side=tk.BOTTOM, padx=10, pady=10)
        

        # Add sliders for adjusting window level and window width
        self.wl_scale = tk.Scale(root, from_=-1000, to=6000, orient=tk.VERTICAL, label="WL", command=self.update_wl,length=300)
        self.ww_scale = tk.Scale(root, from_=1, to=6000, orient=tk.VERTICAL, label="WW", command=self.update_ww,length=300)
        self.ww_scale.pack(side=tk.RIGHT, padx=10, pady=10)
        self.wl_scale.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Add a slider for navigating through slices
        self.slice_slider = tk.Scale(root, from_=0, to=1, orient=tk.VERTICAL, resolution=1, command=self.update_slice,length=300)
        self.slice_slider.pack(side=tk.LEFT, padx=10, pady=10)

        # Bind mouse motion event to update pixel values
        self.canvas.bind("<Motion>", self.update_pixel_values)
        
    def open_volume(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        self.image=None
        if file_path:
            self.volume = np.load(file_path)
            self.current_slice_index = 0
            self.slice_slider.config(from_=0, to=len(self.volume) - 1, state=tk.NORMAL)
            self.slice_slider.set(0)
            self.update_slice(0)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        #self.volume=None
        if file_path:
            self.image = np.load(file_path)
            self.current_slice_index = 0
            self.slice_slider.config(from_=0, to=1, state=tk.NORMAL)
            self.slice_slider.set(0)
            self.update_slice(0)

    def normalize_volume(self,npy_volume):
        # Adjust window level and window width
        normalized_volume=npy_volume.copy()
        window_min = self.window_level - self.window_width / 2
        window_max = self.window_level + self.window_width / 2
        for i, image in enumerate(normalized_volume) :
            # Clip values to the window level and width
            image = np.clip(image, window_min, window_max)
            image=image/window_max
            normalized_volume[i]=image
            
        return normalized_volume
        

    def update_slice(self, val):
        self.current_slice_index = int(self.slice_slider.get())
        
        if self.image!=None:
            slice_to_show = self.image
        else :
            slice_to_show = self.volume[self.current_slice_index]
            
        # Adjust window level and window width
        window_min = self.window_level - self.window_width / 2
        window_max = self.window_level + self.window_width / 2

        # Clip values to the window level and width
        slice_to_show = np.clip(slice_to_show, window_min, window_max)
        slice_to_show=(slice_to_show/window_max)*255


        # Convert NumPy array to PIL Image
        image = Image.fromarray(slice_to_show)

        # Convert PIL Image to PhotoImage
        photo_image = ImageTk.PhotoImage(image)

        # Update the Canvas with the new PhotoImage
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
        self.canvas.photo_image = photo_image  # Prevent the PhotoImage from being garbage collected


    def export_stl_vtk(self):
        if self.volume is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])

            if file_path:
                self.export_volume_as_stl_vtk(file_path)

    def export_volume_as_stl_vtk(self, file_path):
        if self.volume is not None:
            # Create a VTK image data
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(self.volume.shape[::-1])
            vtk_image.SetSpacing(1, 1, 1)
            vtk_image.SetOrigin(0, 0, 0)

            # Copy the NumPy array to VTK image data
            normalized_volume=self.normalize_volume(self.volume)
            vtk_array = numpy_support.numpy_to_vtk(normalized_volume.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            vtk_image.GetPointData().SetScalars(vtk_array)

            # threshold = vtk.vtkThreshold()
            # threshold.SetInputData(vtk_image)
            # threshold.ThresholdByLower(0)  # Adjust the threshold as needed
            # threshold.Update()
            # contour.SetInputData(threshold.GetOutput())

            # Convert VTK image data to a VTK PolyData
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(vtk_image)
            contour.ComputeNormalsOn()
            contour.SetValue(0, 0.5)

            # # Apply decimation to reduce the number of triangles
            # decimation = vtk.vtkDecimatePro()
            # decimation.SetInputConnection(contour.GetOutputPort())
            # decimation.SetTargetReduction(0.9)  # Adjust the reduction factor as needed

            # Write the STL file
            stl_writer = vtk.vtkSTLWriter()
            stl_writer.SetFileName(file_path)
            stl_writer.SetInputConnection(contour.GetOutputPort())
            stl_writer.Write()
            
            print(f"Volume exported as STL : {file_path}")


    def update_wl(self, val):
        self.window_level = int(self.wl_scale.get())
        self.update_slice(self.current_slice_index)

    def update_ww(self, val):
        self.window_width = int(self.ww_scale.get())
        self.update_slice(self.current_slice_index)

    def update_pixel_values(self, event):
        x, y = event.x, event.y

        if self.volume is not None:
            pixel_value = self.volume[self.current_slice_index][int(y), int(x)]
            self.pixel_value_label.config(text=f"Pixel Value: {pixel_value} , pixel Location :{int(y), int(x)}")


if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg='#333333')  # Set the overall background color
    volume_viewer = VolumeViewer(root)
    root.mainloop()