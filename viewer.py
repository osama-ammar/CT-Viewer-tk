import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import vtk
from vtkmodules.util import numpy_support #very tricky issue (using vtkmodules.util insted of  vtk.util ) both will work but when converting to exe vtk.util will not work
import pydicom
import os 
import pyvista as pv

class VolumeViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Volume Viewer")

        # Set the initial dimensions of the window
        self.root.geometry("1000x900")  # Adjust the dimensions as needed
        
        # Configure a dark background
        self.root.configure(bg='#333333')        
        
        
        # Initialize variables
        self.volume = None
        self.image = None 
        self.current_slice_index = 0
        self.window_level = 400  # Initial value, adjust as needed
        self.window_width = 2000  # Initial value, adjust as needed
        self.view_mode = tk.StringVar(value="axial")  # Default to axial view
        self.volume_type = "npy"
        self.pyvista_mesh = None
        
        
        # Create a frame for buttons
        self.button_frame = tk.Frame(root, bg='#333333')
        self.button_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Add buttons for opening volume and image (horizontal arrangement)
        self.open_volume_button = tk.Button(self.button_frame, text="Open npy Volume", command=self.open_volume, bg='#555555', fg='white')
        self.open_volume_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.open_image_button = tk.Button(self.button_frame, text="Open npy Image", command=self.open_image, bg='#555555', fg='white')
        self.open_image_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.open_dicom_button = tk.Button(self.button_frame, text="Open dicom", command=self.open_dicom_case, bg='#555555', fg='white')
        self.open_dicom_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.export_stl_button = tk.Button(self.button_frame, text="Export as STL", command=self.export_stl_vtk, bg='#555555', fg='white')
        self.export_stl_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add radio buttons for view modes (axial, sagittal, coronal)
        tk.Radiobutton(self.button_frame, text="Axial", variable=self.view_mode, value="axial", command=self.update_view, bg='#333333', fg='white').pack(side=tk.LEFT, padx=5, pady=5)
        tk.Radiobutton(self.button_frame, text="Sagittal", variable=self.view_mode, value="sagittal", command=self.update_view, bg='#333333', fg='white').pack(side=tk.LEFT, padx=5, pady=5)
        tk.Radiobutton(self.button_frame, text="Coronal", variable=self.view_mode, value="coronal", command=self.update_view, bg='#333333', fg='white').pack(side=tk.LEFT, padx=5, pady=5)

        self.show_3d = tk.Button(self.button_frame, text="Show 3D", command=self.open_3d_view, bg='#555555', fg='white')
        self.show_3d.pack(side=tk.LEFT, padx=5, pady=5)


        # Create Tkinter Canvas
        self.canvas = tk.Canvas(root, bg='#222222')  # Set canvas background color
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        # Label to display pixel values
        self.pixel_value_label = tk.Label(root, text="Pixel Value: ", bg='#333333', fg='white')
        self.pixel_value_label.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Add sliders for adjusting window level and window width
        self.wl_scale = tk.Scale(root, from_=-1000, to=4000, orient=tk.VERTICAL, label="WL", command=self.update_wl, length=400)
        self.ww_scale = tk.Scale(root, from_=1, to=4000, orient=tk.VERTICAL, label="WW", command=self.update_ww, length=400)
        self.ww_scale.pack(side=tk.RIGHT, padx=10, pady=10)
        self.wl_scale.pack(side=tk.RIGHT, padx=10, pady=10)

        # Add a slider for navigating through slices
        self.slice_slider = tk.Scale(root, from_=0, to=1, orient=tk.VERTICAL, resolution=1, command=self.update_slice, length=400)
        self.slice_slider.pack(side=tk.LEFT, padx=10, pady=10)

        # Bind mouse wheel event to update the displayed slice
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

        # Bind mouse motion event to update pixel values
        self.canvas.bind("<Motion>", self.update_pixel_values)
        

        
    def open_volume(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy"), ("All Files", "*.*")])
        self.image=None
        if file_path:
            self.volume = np.load(file_path)
            self.current_slice_index = 0
            
            # Set specific window level and window width for the volume
            # Update the window level and width sliders
            self.wl_scale.set(self.window_level)
            self.ww_scale.set(self.window_width)
            
            self.slice_slider.config(from_=0, to=len(self.volume) - 1, state=tk.NORMAL)
            self.slice_slider.set(0)
            self.update_slice(0)

    def open_dicom_case(self):
        
        file_path = filedialog.askopenfilename(filetypes=[("Dicom files", "*.dcm")])
        self.image=None
        if file_path:
            # List to hold the slices
            slices = []
            self.volume_type = "dicom"
            dicom_folder = os.path.dirname(os.path.abspath(file_path))
            # Loop through all files in the given directory
            for filename in os.listdir(dicom_folder):
                # Construct the full file path
                filepath = os.path.join(dicom_folder, filename)
                
                # Try to read the file as a DICOM file
                try:
                    dicom_file = pydicom.dcmread(filepath)
                    slices.append(dicom_file)
                except Exception as e:
                    print(f"Could not read {filepath}: {e}")

            # Sort the slices by the InstanceNumber (or another relevant attribute)
            slices.sort(key=lambda s: int(s.InstanceNumber))

            # Create a 3D NumPy array from the DICOM slices
            self.volume =np.stack([s.pixel_array for s in slices])
            self.current_slice_index = 0
            
            # Set specific window level and window width for the volume
            # Update the window level and width sliders
            self.wl_scale.set(self.window_level)
            self.ww_scale.set(self.window_width)
            
            self.slice_slider.config(from_=0, to=len(self.volume) - 1, state=tk.NORMAL)
            self.slice_slider.set(0)
            self.update_slice(0)
            print(self.volume.shape)
        
        

    
        # later : return dicom info to be displayed laterwith the volume

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        #self.volume=None
        if file_path:
            self.image = np.load(file_path)
            self.current_slice_index = 0
            self.slice_slider.config(from_=0, to=1, state=tk.NORMAL)
            self.slice_slider.set(0)
            self.update_slice(0)

    # TODO : make normalization compatible wih different pixels ranges (0 to 1) | (0 to 255) |(-1000 to 5000)
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
        
    def update_view(self):
        """Update the view based on the selected view mode."""
        self.update_slice(self.current_slice_index)
        
    def update_slice(self, val):
        self.current_slice_index = int(self.slice_slider.get())
        
        if self.image!=None:
            slice_to_show = self.image
        else :

            # Extract the appropriate slice based on the view mode
            if self.view_mode.get() == "axial":
                slice_to_show = self.volume[self.current_slice_index, :, :]
                self.slice_slider.config(from_=0, to=self.volume.shape[0] - 1, state=tk.NORMAL)
                
            elif self.view_mode.get() == "sagittal":
                slice_to_show = self.volume[:, :, self.current_slice_index]
                self.slice_slider.config(from_=0, to=self.volume.shape[1] - 1, state=tk.NORMAL)
                slice_to_show = np.flipud(slice_to_show) if self.volume_type == "npy" else slice_to_show
                
            elif self.view_mode.get() == "coronal":
                slice_to_show = self.volume[:, self.current_slice_index, :]
                # adjust slice slider because coronal images number is different
                self.slice_slider.config(from_=0, to=self.volume.shape[2] - 1, state=tk.NORMAL)
                slice_to_show = np.flipud(slice_to_show)  if self.volume_type == "npy" else slice_to_show
                
            
            
        # Adjust window level and window width
        window_min = self.window_level - self.window_width / 2
        window_max = self.window_level + self.window_width / 2

        # Ensure window_min is not greater than window_max (fix any bad configurations)
        if window_min >= window_max:
            window_min = 0
            window_max = 255
            
        # Clip values to the window level and width
        slice_to_show = np.clip(slice_to_show, window_min, window_max)
        slice_to_show = 255 * (slice_to_show - window_min) / (window_max - window_min)


        # Convert NumPy array to PIL Image
        image = Image.fromarray(slice_to_show)

        # Convert PIL Image to PhotoImage
        photo_image = ImageTk.PhotoImage(image)

        # Update the Canvas with the new PhotoImage
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
        self.canvas.photo_image = photo_image  # Prevent the PhotoImage from being garbage collected


    def update_wl(self, val):
        self.window_level = int(self.wl_scale.get())
        self.update_slice(self.current_slice_index)

    def update_ww(self, val):
        self.window_width = int(self.ww_scale.get())
        self.update_slice(self.current_slice_index)

    def update_pixel_values(self, event):
        x, y = event.x, event.y

        # TODO: this sfould be modified to account for other orthogonal views
        if self.volume is not None:
            pixel_value = self.volume[self.current_slice_index][int(y), int(x)]
            self.pixel_value_label.config(text=f"Pixel Value: {pixel_value} , pixel Location :{int(y), int(x)}")

    def on_mousewheel(self, event):
        
        # Determine the direction of the mouse wheel scroll
        delta = event.delta
        # Update the current slice index based on the mouse wheel direction
        if delta > 0:  # Scrolling up
            self.current_slice_index+=1
        else:  # Scrolling down
            self.current_slice_index -=1
        # Update the displayed slice
        self.slice_slider.set(self.current_slice_index)
        self.update_slice(self.current_slice_index)

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

            # Convert VTK image data to a VTK PolyData
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(vtk_image)
            contour.ComputeNormalsOn()
            contour.SetValue(0, 0.5)

            # Write the STL file
            stl_writer = vtk.vtkSTLWriter()
            stl_writer.SetFileName(file_path)
            stl_writer.SetInputConnection(contour.GetOutputPort())
            stl_writer.Write()
            print(f"Volume exported as STL : {file_path}")


    def open_3d_view(self):

        # Adjust window level and window width
        window_min = self.window_level - self.window_width / 2
        window_max = self.window_level + self.window_width / 2

        # Ensure window_min is not greater than window_max (fix any bad configurations)
        if window_min >= window_max:
            window_min = 0
            window_max = 255
            
        # Clip values to the window level and width
        volume_3d = np.clip(self.volume, window_min, window_max)
        volume_3d = 255 * (volume_3d - window_min) / (window_max - window_min)


        # pyvista
        normalized_volume=self.normalize_volume(self.volume)
        volume = pv.wrap(normalized_volume)
        # Step 3: Plotting the volume
        plotter = pv.Plotter()
        # Using volume rendering with a custom color map
        opacity = [0.0, 0.05, 0.1, 0.4, 0.8, 1.0] 
        plotter.add_volume(volume,cmap="magma",opacity="sigmoid_9",show_scalar_bar=False,shade=True)

        # Configure lighting for a cinematic effect
        light = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0))
        light.intensity = 0.8  # Stronger lighting for cinematic shadows
        light.specular = 0.5  # Higher specular for a shiny effect
        plotter.add_light(light)

        # Configure a secondary light for depth
        secondary_light = pv.Light(position=(-1, -1, -1), focal_point=(0, 0, 0))
        secondary_light.intensity = 0.4  # Softer secondary light to fill shadows
        plotter.add_light(secondary_light)

        # Render with custom settings
        plotter.camera_position = 'iso'  # Iso view for better depth

        # Step 4: Show the plot
        plotter.show()
 
def main():
    root = tk.Tk()
    root.configure(bg='#333333')  # Set the overall background color
    volume_viewer = VolumeViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()  