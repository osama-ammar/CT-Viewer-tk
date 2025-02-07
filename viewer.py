import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import pydicom
import os 
import pyvista as pv
import nrrd
import utilities


class VolumeViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Volume Viewer")
        self.root.geometry("1000x900")  # Adjust the dimensions as needed
        self.root.configure(bg='#333333')        
        
        # Initialize variables for viewer 1
        self.volume = None
        self.image = None 
        self.current_slice_index = 0
        self.window_level = 400  # Initial value, adjust as needed
        self.window_width = 2000  # Initial value, adjust as needed
        self.view_mode = tk.StringVar(value="axial")  # Default to axial view
        self.volume_type = "npy"
        self.pyvista_mesh = None
        self.unique_labels =None
        
        # Initialize variables for viewer 2
        self.volume_2 = None
        self.image_2 = None 
        self.current_slice_index_2 = 0
        self.window_level_2 = 400  # Initial value, adjust as needed
        self.window_width_2 = 2000  # Initial value, adjust as needed
        self.view_mode_2 = tk.StringVar(value="axial")  # Default to axial view
        self.volume_type_2 = "npy"
        self.pyvista_mesh_2 = None
        self.unique_labels_2 =None
        
        self.create_viewer_1()
        self.create_viewer_2()

        # Label to display pixel values
        self.pixel_value_label = tk.Label(root, text="Pixel Value: ", bg='#333333', fg='white')
        self.pixel_value_label.pack(side=tk.BOTTOM, padx=10, pady=10)   
        
        
    def create_viewer_1(self):
        # frames 
        viewer1_frame = tk.Frame(self.root, bg='#333333')
        viewer1_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X)

        button_frame = tk.Frame(viewer1_frame, bg='#333333')
        button_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)

        radio_buttons_frame = tk.Frame(viewer1_frame, bg='#333333')
        radio_buttons_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)

        image_frame = tk.Frame(viewer1_frame, bg='#333333')
        image_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)

        brightness_frame = tk.Frame(viewer1_frame, bg='#333333')
        brightness_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)
        
        # image canvas
        self.canvas = tk.Canvas(image_frame, bg='#222222')  # Set canvas background color
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        # Bind mouse wheel event to update the displayed slice
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        # Bind mouse motion event to update pixel values
        self.canvas.bind("<Motion>", self.update_pixel_values)
        
        # Add buttons for opening volume and image (horizontal arrangement)
        self.open_volume_button = tk.Button(button_frame, text="Open npy Vol", command=self.open_volume, bg='#555555', fg='white')
        self.open_nrrd_button = tk.Button(button_frame, text="Open nrrd", command=self.open_nrrd, bg='#555555', fg='white')
        self.open_image_button = tk.Button(button_frame, text="Open npy Image", command=self.open_image, bg='#555555', fg='white')
        self.open_dicom_button = tk.Button(button_frame, text="Open dicom", command=self.open_dicom_case, bg='#555555', fg='white')
        self.export_stl_button = tk.Button(button_frame, text="Export STL", command=self.export_stl_vtk, bg='#555555', fg='white')
        self.show_3d = tk.Button(button_frame, text="Show 3D", command=self.open_3d_view, bg='#555555', fg='white')
        
        # buttons positions
        self.open_volume_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.open_nrrd_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.open_image_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.open_dicom_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.export_stl_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.show_3d.pack(side=tk.LEFT, padx=5, pady=5)

        self.axial_radio_button=tk.Radiobutton(radio_buttons_frame, text="Axial", variable=self.view_mode, value="axial", command=self.update_view, bg='#333333', fg='white')
        self.sagittal_radio_button=tk.Radiobutton(radio_buttons_frame, text="Sagittal", variable=self.view_mode, value="sagittal", command=self.update_view, bg='#333333', fg='white')
        self.coronal_radio_button=tk.Radiobutton(radio_buttons_frame, text="Coronal", variable=self.view_mode, value="coronal", command=self.update_view, bg='#333333', fg='white')
        self.axial_radio_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.coronal_radio_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.sagittal_radio_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.slice_slider = tk.Scale(image_frame, from_=0, to=1, orient=tk.VERTICAL, resolution=1, command=self.update_slice, length=300)
        self.slice_slider.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.wl_scale = tk.Scale(viewer1_frame , from_=-1000, to=4000, orient=tk.HORIZONTAL, label="WL", command=self.update_wl, length=300)
        self.ww_scale = tk.Scale(viewer1_frame , from_=1, to=4000, orient=tk.HORIZONTAL, label="WW", command=self.update_ww, length=300)

        self.ww_scale.pack(side=tk.BOTTOM, fill=None, padx=10, pady=10)
        self.wl_scale.pack(side=tk.BOTTOM, fill=None, padx=10, pady=10)
        
        
    def create_viewer_2(self):
        # frames 
        viewer2_frame = tk.Frame(self.root, bg='#333333')
        viewer2_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.X)

        button_frame = tk.Frame(viewer2_frame, bg='#333333')
        button_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)

        radio_buttons_frame = tk.Frame(viewer2_frame, bg='#333333')
        radio_buttons_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)

        image_frame = tk.Frame(viewer2_frame, bg='#333333')
        image_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)

        brightness_frame = tk.Frame(viewer2_frame, bg='#333333')
        brightness_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)
        
        # image canvas
        self.canvas_2 = tk.Canvas(image_frame, bg='#222222')  # Set canvas background color
        self.canvas_2.pack(side=tk.LEFT, padx=10, pady=10)
        # Bind mouse wheel event to update the displayed slice
        self.canvas_2.bind("<MouseWheel>", self.on_mousewheel)
        # Bind mouse motion event to update pixel values
        self.canvas_2.bind("<Motion>", self.update_pixel_values)
        
        # Add buttons for opening volume and image (horizontal arrangement)
        self.open_volume_button_2 = tk.Button(button_frame, text="Open npy Vol", command=self.open_volume_2, bg='#555555', fg='white')
        self.open_nrrd_button_2 = tk.Button(button_frame, text="Open nrrd", command=self.open_nrrd_2, bg='#555555', fg='white')
        self.open_image_button_2 = tk.Button(button_frame, text="Open npy Image", command=self.open_image_2, bg='#555555', fg='white')
        self.open_dicom_button_2 = tk.Button(button_frame, text="Open dicom", command=self.open_dicom_case_2, bg='#555555', fg='white')
        self.export_stl_button_2 = tk.Button(button_frame, text="Export STL", command=self.export_stl_vtk_2, bg='#555555', fg='white')
        self.show_3d_2 = tk.Button(button_frame, text="Show 3D", command=self.open_3d_view_2, bg='#555555', fg='white')
        
        self.open_volume_button_2.pack(side=tk.LEFT, padx=5, pady=5)
        self.open_nrrd_button_2.pack(side=tk.LEFT, padx=5, pady=5)
        self.open_image_button_2.pack(side=tk.LEFT, padx=5, pady=5)
        self.open_dicom_button_2.pack(side=tk.LEFT, padx=5, pady=5)
        self.export_stl_button_2.pack(side=tk.LEFT, padx=5, pady=5)
        self.show_3d_2.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.axial_radio_button_2=tk.Radiobutton(radio_buttons_frame, text="Axial", variable=self.view_mode_2, value="axial", command=self.update_view_2, bg='#333333', fg='white').pack(side=tk.LEFT, padx=5, pady=5)
        self.sagittal_radio_button_2=tk.Radiobutton(radio_buttons_frame, text="Sagittal", variable=self.view_mode_2, value="sagittal", command=self.update_view_2, bg='#333333', fg='white').pack(side=tk.LEFT, padx=5, pady=5)
        self.coronal_radio_button_2=tk.Radiobutton(radio_buttons_frame, text="Coronal", variable=self.view_mode_2, value="coronal", command=self.update_view_2, bg='#333333', fg='white').pack(side=tk.LEFT, padx=5, pady=5)

        self.slice_slider_2 = tk.Scale(image_frame, from_=0, to=1, orient=tk.VERTICAL, resolution=1, command=self.update_slice_2, length=300)
        self.slice_slider_2.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.wl_scale_2 = tk.Scale(viewer2_frame , from_=-1000, to=4000, orient=tk.HORIZONTAL, label="WL", command=self.update_wl_2, length=300)
        self.wl_scale_2.pack(side=tk.BOTTOM, fill=None, padx=10, pady=10)
        self.ww_scale_2 = tk.Scale(viewer2_frame , from_=1, to=4000, orient=tk.HORIZONTAL, label="WW", command=self.update_ww_2, length=300)
        self.ww_scale_2.pack(side=tk.BOTTOM, fill=None, padx=10, pady=10)


        
    def open_volume(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy"), ("All Files", "*.*")])
        self.image=None
        if file_path:
            self.volume = np.load(file_path)
            self.current_slice_index = 0
            
            self.unique_labels = np.unique(self.volume)
            self.wl_scale.config(from_=min(self.unique_labels), to=max(self.unique_labels)-1, state=tk.NORMAL)
            self.wl_scale.config(from_=0, to=max(self.unique_labels) - 1, state=tk.NORMAL)
            self.wl_scale.set(max(self.unique_labels))
            self.ww_scale.set(max(self.unique_labels))

            self.slice_slider.config(from_=0, to=len(self.volume) - 1, state=tk.NORMAL)
            self.slice_slider.set(0)
            self.update_slice(0)

    def open_volume_2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy"), ("All Files", "*.*")])
        self.image_2=None
        if file_path:
            self.volume_2 = np.load(file_path)
            self.current_slice_index_2 = 0
            
            self.unique_labels_2 = np.unique(self.volume_2)
            self.wl_scale_2.config(from_=min(self.unique_labels_2), to=max(self.unique_labels_2)-1, state=tk.NORMAL)
            self.wl_scale_2.config(from_=0, to=max(self.unique_labels_2) - 1, state=tk.NORMAL)
            self.wl_scale_2.set(max(self.unique_labels_2))
            self.ww_scale_2.set(max(self.unique_labels_2))

            self.slice_slider_2.config(from_=0, to=len(self.volume_2) - 1, state=tk.NORMAL)
            self.slice_slider_2.set(0)
            self.update_slice_2(0)
            
    def open_nrrd(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy" ), ("All Files", "*.*")])
        self.image=None
        if file_path:
            self.volume , header= nrrd.read(file_path)
            self.volume = np.transpose(self.volume, (2, 1, 0))
            self.current_slice_index = 0
            
            self.unique_labels = np.unique(self.volume)
            self.wl_scale.config(from_=min(self.unique_labels), to=max(self.unique_labels)-1, state=tk.NORMAL)
            self.wl_scale.config(from_=0, to=max(self.unique_labels) - 1, state=tk.NORMAL)
            self.wl_scale.set(max(self.unique_labels))
            self.ww_scale.set(max(self.unique_labels))

            self.slice_slider.config(from_=0, to=len(self.volume) - 1, state=tk.NORMAL)
            self.slice_slider.set(0)
            self.update_slice(0)

            
    def open_nrrd_2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy" ), ("All Files", "*.*")])
        self.image_2=None
        if file_path:
            self.volume_2 , header= nrrd.read(file_path)
            self.volume_2 = np.transpose(self.volume, (2, 1, 0))
            self.current_slice_index_2 = 0
            
            self.unique_labels_2 = np.unique(self.volume_2)
            self.wl_scale_2.config(from_=min(self.unique_labels_2), to=max(self.unique_labels_2)-1, state=tk.NORMAL)
            self.wl_scale_2.config(from_=0, to=max(self.unique_labels_2) - 1, state=tk.NORMAL)
            self.wl_scale_2.set(max(self.unique_labels_2))
            self.ww_scale_2.set(max(self.unique_labels_2))

            self.slice_slider_2.config(from_=0, to=len(self.volume_2) - 1, state=tk.NORMAL)
            self.slice_slider_2.set(0)
            self.update_slice_2(0)
            
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

    def open_dicom_case_2(self):
        
        file_path = filedialog.askopenfilename(filetypes=[("Dicom files", "*.dcm")])
        self.image_2=None
        if file_path:
            # List to hold the slices
            slices = []
            self.volume_type_2 = "dicom"
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
            self.volume_2 =np.stack([s.pixel_array for s in slices])
            self.current_slice_index_2 = 0
            
            # Set specific window level and window width for the volume
            # Update the window level and width sliders
            self.wl_scale_2.set(self.window_level_2)
            self.ww_scale_2.set(self.window_width_2)
            
            self.slice_slider_2.config(from_=0, to=len(self.volume_2) - 1, state=tk.NORMAL)
            self.slice_slider_2.set(0)
            self.update_slice_2(0)
            print(self.volume.shape_2)
        
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

        
    def open_image_2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        #self.volume=None
        if file_path:
            self.image_2 = np.load(file_path)
            self.current_slice_index_2 = 0
            self.slice_slider_2.config(from_=0, to=1, state=tk.NORMAL)
            self.slice_slider_2.set(0)
            self.update_slice_2(0)
            
    def update_view(self):
        """Update the view based on the selected view mode."""
        self.update_slice(self.current_slice_index)
    
    def update_view_2(self):
        """Update the view based on the selected view mode."""
        self.update_slice_2(self.current_slice_index_2)
        
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

        
    def update_slice_2(self, val):
        self.current_slice_index_2 = int(self.slice_slider_2.get())
        if self.image_2!=None:
            slice_to_show = self.image_2
        else :
            # Extract the appropriate slice based on the view mode
            if self.view_mode_2.get() == "axial":
                slice_to_show = self.volume_2[self.current_slice_index_2, :, :]
                self.slice_slider_2.config(from_=0, to=self.volume_2.shape[0] - 1, state=tk.NORMAL)
                
            elif self.view_mode_2.get() == "sagittal":
                slice_to_show = self.volume_2[:, :, self.current_slice_index_2]
                self.slice_slider_2.config(from_=0, to=self.volume_2.shape[1] - 1, state=tk.NORMAL)
                slice_to_show = np.flipud(slice_to_show) if self.volume_type_2 == "npy" else slice_to_show
                
            elif self.view_mode_2.get() == "coronal":
                slice_to_show = self.volume_2[:, self.current_slice_index_2, :]
                # adjust slice slider because coronal images number is different
                self.slice_slider_2.config(from_=0, to=self.volume_2.shape[2] - 1, state=tk.NORMAL)
                slice_to_show = np.flipud(slice_to_show)  if self.volume_type_2 == "npy" else slice_to_show
                
        # Adjust window level and window width
        window_min = self.window_level_2 - self.window_width_2 / 2
        window_max = self.window_level_2 + self.window_width_2 / 2

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
        self.canvas_2.config(width=image.width, height=image.height)
        self.canvas_2.create_image(0, 0, anchor=tk.NW, image=photo_image)
        self.canvas_2.photo_image = photo_image  # Prevent the PhotoImage from being garbage collected



    def update_wl(self, val):
        self.window_level = int(self.wl_scale.get())
        self.update_slice(self.current_slice_index)

    def update_ww(self, val):
        self.window_width = int(self.ww_scale.get())
        self.update_slice(self.current_slice_index)

    def update_wl_2(self, val):
        self.window_level_2 = int(self.wl_scale_2.get())
        self.update_slice_2(self.current_slice_index_2)

    def update_ww_2(self, val):
        self.window_width_2 = int(self.ww_scale_2.get())
        self.update_slice_2(self.current_slice_index_2)


    def update_pixel_values(self, event):
        x, y = event.x, event.y

        # TODO: this sfould be modified to account for other orthogonal views + there's a problem here with mouse position and x , y
        if self.volume is not None:
            try:
                pixel_value = self.volume[self.current_slice_index][int(y)+1][int(x)+1]
            except IndexError:
                pixel_value = 0
                
            self.pixel_value_label.config(text=f"Pixel Value: {pixel_value} , pixel Location :{int(y), int(x)}")


    def update_pixel_values_2(self, event):
        x, y = event.x, event.y

        # TODO: this sfould be modified to account for other orthogonal views + there's a problem here with mouse position and x , y
        if self.volume_2 is not None:
            try:
                pixel_value = self.volume_2[self.current_slice_index_2][int(y)+1][int(x)+1]
            except IndexError:
                pixel_value = 0
                
            self.pixel_value_label_2.config(text=f"Pixel Value: {pixel_value} , pixel Location :{int(y), int(x)}")


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


    def on_mousewheel_2(self, event):
        
        # Determine the direction of the mouse wheel scroll
        delta = event.delta
        # Update the current slice index based on the mouse wheel direction
        if delta > 0:  # Scrolling up
            self.current_slice_index_2+=1
        else:  # Scrolling down
            self.current_slice_index_2 -=1
        # Update the displayed slice
        self.slice_slider.set(self.current_slice_index_2)
        self.update_slice(self.current_slice_index_2)


    def export_stl_vtk(self):
        if self.volume is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])

            if file_path:
                utilities.export_volume_as_stl_vtk(self.volume,file_path,self.window_level,self.window_width)



    def export_stl_vtk_2(self):
        if self.volume_2 is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])

            if file_path:
                utilities.export_volume_as_stl_vtk(self.volume_2,file_path,self.window_level_2,self.window_width_2)

    def open_3d_view(self):
        #plotter = utilities.open_3d_view(self.volume,self.window_level,self.window_width)
        plotter = utilities.npy_to_pyvista(self.volume)
        plotter.show()

    def open_3d_view_2(self):
        #plotter = utilities.open_3d_view(self.volume,self.window_level,self.window_width)
        plotter = utilities.npy_to_pyvista(self.volume_2)
        plotter.show()
        
def main():
    root = tk.Tk()
    root.configure(bg='#333333')  # Set the overall background color
    volume_viewer = VolumeViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()  