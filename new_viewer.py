import tkinter as tk
from tkinter import filedialog, ttk
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
        self.root.geometry("1200x900")
        self.root.configure(bg='#333333')

        # Initialize variables
        self.volume = None
        self.image = None
        self.current_slice_index = 0
        self.window_level = 400
        self.window_width = 2000
        self.view_mode = tk.StringVar(value="axial")
        self.volume_type = "npy"
        self.pyvista_mesh = None
        self.unique_labels = None
        self.resized_image = None  # To store the resized image
        
        self.slice_index = tk.StringVar()
        self.slice_index.set("slice: 0")
        self.wl_index= tk.StringVar()
        self.wl_index.set("WL: 0") 
        self.ww_index= tk.StringVar()
        self.ww_index.set("WW: 0")
        
        
        # Create a main frame for better organization
        self.main_frame = tk.Frame(root, bg='#333333')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a frame for buttons and controls
        self.control_frame = tk.Frame(self.main_frame, bg='#333333')
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Add buttons for opening volume and image
        self.open_volume_button = ttk.Button(self.control_frame, text="Open npy Volume", command=self.open_volume)
        self.open_volume_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.open_nrrd_button = ttk.Button(self.control_frame, text="Open nrrd/nii", command=self.open_nrrd)
        self.open_nrrd_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.open_image_button = ttk.Button(self.control_frame, text="Open npy Image", command=self.open_image)
        self.open_image_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.open_dicom_button = ttk.Button(self.control_frame, text="Open dicom", command=self.open_dicom_case)
        self.open_dicom_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.export_stl_button = ttk.Button(self.control_frame, text="Export as STL", command=self.export_stl_vtk)
        self.export_stl_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add radio buttons for view modes
        self.view_mode_frame = tk.Frame(self.control_frame, bg='#333333')
        self.view_mode_frame.pack(side=tk.LEFT, padx=10, pady=5)

        ttk.Radiobutton(self.view_mode_frame, text="Axial", variable=self.view_mode, value="axial", command=self.update_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.view_mode_frame, text="Sagittal", variable=self.view_mode, value="sagittal", command=self.update_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.view_mode_frame, text="Coronal", variable=self.view_mode, value="coronal", command=self.update_view).pack(side=tk.LEFT, padx=5)

        self.show_3d_button = ttk.Button(self.control_frame, text="Show 3D", command=self.open_3d_view)
        self.show_3d_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Create a frame for the canvas and sliders
        self.display_frame = tk.Frame(self.main_frame, bg='#333333')
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create Tkinter Canvas
        self.canvas = tk.Canvas(self.display_frame, bg='#222222')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a frame for sliders
        self.slider_frame = tk.Frame(self.display_frame, bg='#333333')
        self.slider_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Add sliders for adjusting window level and window width
        self.wl_scale = ttk.Scale(self.slider_frame, from_=-1000, to=4000, orient=tk.VERTICAL, length=400, command=self.update_wl)
        self.wl_scale.pack(side=tk.RIGHT, padx=5, pady=5)
        self.wl_label = ttk.Label(self.slider_frame, text="WL", textvariable=self.wl_index, background='#333333', foreground='white')
        self.wl_label.pack(side=tk.RIGHT, padx=5, pady=5)

        self.ww_scale = ttk.Scale(self.slider_frame, from_=1, to=4000, orient=tk.VERTICAL, length=400, command=self.update_ww)
        self.ww_scale.pack(side=tk.RIGHT, padx=5, pady=5)
        self.ww_label = ttk.Label(self.slider_frame, text="WW", textvariable=self.ww_index, background='#333333', foreground='white')
        self.ww_label.pack(side=tk.RIGHT, padx=5, pady=5)

        # Add a slider for navigating through slices
        self.slice_slider = ttk.Scale(self.slider_frame, from_=0, to=1, orient=tk.VERTICAL, length=400, command=self.update_slice)
        self.slice_slider.pack(side=tk.RIGHT, padx=5, pady=5)
        self.slice_label = ttk.Label(self.slider_frame, text="Slice", textvariable=self.slice_index, background='#333333', foreground='white')
        self.slice_label.pack(side=tk.RIGHT, padx=5, pady=5)

        # Label to display pixel values
        self.pixel_value_label = ttk.Label(self.main_frame, text="Pixel Value: ", background='#333333', foreground='white')
        self.pixel_value_label.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Bind mouse wheel event to update the displayed slice
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

        # Bind mouse motion event to update pixel values
        self.canvas.bind("<Motion>", self.update_pixel_values)
        

    def open_volume(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy"), ("All Files", "*.*")])
        self.image = None
        if file_path:
            self.volume = np.load(file_path)
            self.current_slice_index = 0
            self.unique_labels = np.unique(self.volume)
            
            # Update sliders
            self.wl_scale.config(from_=min(self.unique_labels), to=max(self.unique_labels), state=tk.NORMAL)
            self.wl_scale.set(max(self.unique_labels) // 2)  # Set WL to mid-range
            self.ww_scale.config(from_=1, to=max(self.unique_labels), state=tk.NORMAL)
            self.ww_scale.set(max(self.unique_labels))  # Set WW to max
            
            # Update slice slider based on the view mode
            self.update_slice_slider_range()
            self.slice_slider.set(0)
            
            # Display the initial slice
            self.update_slice(0)

    def open_nrrd(self):
        file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
        self.image = None
        if file_path and file_path.endswith("nrrd"):
            self.volume_type = "nrrd"
            self.volume, header = nrrd.read(file_path)

        if file_path and ".nii" in file_path:
            self.volume, spacing = utilities.load_nifti(file_path)
            
        self.volume = np.transpose(self.volume, (2, 1, 0))
        print(self.volume.shape)
        self.current_slice_index = 0
        self.unique_labels = np.unique(self.volume)
        
        # Update sliders
        self.wl_scale.config(from_=min(self.unique_labels), to=max(self.unique_labels), state=tk.NORMAL)
        self.wl_scale.set(max(self.unique_labels) // 2)  # Set WL to mid-range
        self.ww_scale.config(from_=1, to=max(self.unique_labels), state=tk.NORMAL)
        self.ww_scale.set(max(self.unique_labels))  # Set WW to max
        
        # Update slice slider based on the view mode
        self.update_slice_slider_range()
        self.slice_slider.set(0)
        
        # Display the initial slice
        self.update_slice(0)

    def open_dicom_case(self):
        file_path = filedialog.askopenfilename(filetypes=[("Dicom files", "*.dcm")])
        self.image = None
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
            
            # Update sliders
            self.wl_scale.config(from_=self.volume.min(), to=self.volume.max(), state=tk.NORMAL)
            self.wl_scale.set(self.window_level) # Set WL to default
            self.ww_scale.config(from_=1, to=self.volume.max(), state=tk.NORMAL)
            self.ww_scale.set(self.window_width)  # Set WW to default
            
            # Update slice slider based on the view mode
            self.update_slice_slider_range()
            self.slice_slider.set(0)
            
            # Display the initial slice
            self.update_slice(0)
            print(self.volume.shape)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        if file_path:
            self.image = np.load(file_path)
            self.current_slice_index = 0
            self.slice_slider.config(from_=0, to=1, state=tk.NORMAL)
            self.slice_slider.set(0)
            
            # Resize the image only once when loaded
            self.resized_image = self.resize_image_to_canvas(self.image)
            self.update_slice(0)
            
    def update_view(self):
        """Update the view based on the selected view mode."""
        self.update_slice_slider_range()
        self.update_slice(self.current_slice_index)

    def update_slice_slider_range(self):
        if self.volume is not None:
            if self.view_mode.get() == "axial":
                self.slice_slider.config(from_=0, to=self.volume.shape[0] - 1, state=tk.NORMAL)
            elif self.view_mode.get() == "sagittal":
                self.slice_slider.config(from_=0, to=self.volume.shape[2] - 1, state=tk.NORMAL)
            elif self.view_mode.get() == "coronal":
                self.slice_slider.config(from_=0, to=self.volume.shape[1] - 1, state=tk.NORMAL)
                    
    def update_slice(self, val):
        self.current_slice_index = int(self.slice_slider.get())
        self.slice_index.set(f"slice: { self.current_slice_index}") 
        #print(f"slice: { self.current_slice_index}") 
        #print(self.volume.shape)
        
        if self.image is not None:
            slice_to_show = self.image
        else:
            if self.view_mode.get() == "axial":
                slice_to_show = self.volume[self.current_slice_index, :, :]
                
                
            elif self.view_mode.get() == "sagittal":
                slice_to_show = self.volume[:,: ,self.current_slice_index]
                slice_to_show = slice_to_show if self.volume_type == "dicom" else np.flipud(slice_to_show)

                
            elif self.view_mode.get() == "coronal":
                slice_to_show = self.volume[:, self.current_slice_index,:]
                slice_to_show = slice_to_show if self.volume_type == "dicom" else np.flipud(slice_to_show)


        # Apply window level and width
        window_min = self.window_level - self.window_width / 2
        window_max = self.window_level + self.window_width / 2
        if window_min >= window_max:
            window_min = 0
            window_max = 255

        # Clip and scale the pixel values
        slice_to_show = np.clip(slice_to_show, window_min, window_max)
        slice_to_show = 255 * (slice_to_show - window_min) / (window_max - window_min)

        # Convert to PIL Image
        image = Image.fromarray(slice_to_show.astype(np.uint8))

        # Resize the image to fit the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 0 and canvas_height > 0:  # Ensure canvas dimensions are valid
            image_ratio = image.width / image.height
            canvas_ratio = canvas_width / canvas_height

            if canvas_ratio > image_ratio:
                # Canvas is wider than the image
                new_height = canvas_height
                new_width = int(image_ratio * new_height)
            else:
                # Canvas is taller than the image
                new_width = canvas_width
                new_height = int(new_width / image_ratio)

            # Resize the image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and display
        photo_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=photo_image)
        self.canvas.photo_image = photo_image  # Prevent garbage collection
        
    def update_wl(self, val):
        self.window_level = int(self.wl_scale.get())
        self.wl_index.set(f"wl: { self.window_level}") 
        self.update_slice(self.current_slice_index)

    def update_ww(self, val):
        self.window_width = int(self.ww_scale.get())
        self.ww_index.set(f"ww: { self.window_width}") 
        self.update_slice(self.current_slice_index)

    def on_mousewheel(self, event):
        delta = event.delta
        if delta > 0:  # Scrolling up
            self.current_slice_index += 1
        else:  # Scrolling down
            self.current_slice_index -= 1
        
        # Ensure the slice index stays within bounds
        if self.volume is not None:
            if self.view_mode.get() == "axial":
                max_slices = self.volume.shape[0] - 1
            elif self.view_mode.get() == "sagittal":
                max_slices = self.volume.shape[1] - 1
            elif self.view_mode.get() == "coronal":
                max_slices = self.volume.shape[2] - 1
            self.current_slice_index = max(0, min(self.current_slice_index, max_slices))
        
        self.slice_slider.set(self.current_slice_index)
        self.update_slice(self.current_slice_index)

    def update_pixel_values(self, event):
        x, y = event.x, event.y
        if self.volume is not None:
            try:
                if self.view_mode.get() == "axial":
                    pixel_value = self.volume[self.current_slice_index, int(y), int(x)]
                elif self.view_mode.get() == "sagittal":
                    pixel_value = self.volume[int(y), self.current_slice_index, int(x)]
                elif self.view_mode.get() == "coronal":
                    pixel_value = self.volume[int(y), int(x), self.current_slice_index]
            except IndexError:
                pixel_value = 0
            self.pixel_value_label.config(text=f"Pixel Value: {pixel_value}, Location: ({int(y)}, {int(x)})")

    def export_stl_vtk(self):
        if self.volume is not None:
            # file_path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])
            # if file_path:
            #     utilities.export_volume_as_stl_vtk(self.volume, file_path, self.window_level, self.window_width)
            utilities.npy_to_colored_mesh(self.volume)

    def open_3d_view(self):
        plotter = utilities.npy_to_pyvista(self.volume)
        plotter.show()

    def resize_image_to_canvas(self, slice_to_show):
        """Resize the image to fit the canvas while maintaining aspect ratio."""
        # Apply window level and width
        window_min = self.window_level - self.window_width / 2
        window_max = self.window_level + self.window_width / 2
        if window_min >= window_max:
            window_min = 0
            window_max = 255

        # Clip and scale the pixel values
        slice_to_show = np.clip(slice_to_show, window_min, window_max)
        slice_to_show = 255 * (slice_to_show - window_min) / (window_max - window_min)

        # Convert to PIL Image
        image = Image.fromarray(slice_to_show.astype(np.uint8))

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 0 and canvas_height > 0:  # Ensure canvas dimensions are valid
            image_ratio = image.width / image.height
            canvas_ratio = canvas_width / canvas_height

            if canvas_ratio > image_ratio:
                # Canvas is wider than the image
                new_height = canvas_height
                new_width = int(image_ratio * new_height)
            else:
                # Canvas is taller than the image
                new_width = canvas_width
                new_height = int(new_width / image_ratio)

            # Resize the image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image

def main():
    root = tk.Tk()
    root.configure(bg='#333333')
    volume_viewer = VolumeViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()