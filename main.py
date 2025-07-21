import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
import pydicom
import os 
import pyvista as pv
import nrrd
import utilities
import time
from functools import lru_cache
import sv_ttk  # For Sun Valley theme
from concurrent.futures import ThreadPoolExecutor

class VolumeViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Volume Viewer")
        self.root.geometry("1400x900")
        
        # Set theme
        sv_ttk.set_theme("dark")
        
        # Initialize variables
        self.volume = None
        self.mask = None
        self.image = None 
        self.current_slice_index = 0
        self.window_level = 400
        self.window_width = 2000
        self.view_mode = tk.StringVar(value="axial")
        self.volume_type = "npy"
        self.pyvista_mesh = None
        self.unique_labels = None
        self._last_canvas_size = (0, 0)
        self._last_wheel_time = 0
        self._mask_colors_cache = None
        
        self.file_name = None
        self.main_folder_path = None

        
        # Create UI elements
        self.create_ui()
        
    def create_ui(self):
        """Initialize all UI components"""
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel for series list
        self.left_panel = ttk.Frame(self.main_container, width=200)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        
        # Create control panel on the right
        self.control_panel = ttk.Frame(self.main_container, width=250)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Create viewer panel
        self.viewer_panel = ttk.Frame(self.main_container)
        self.viewer_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add left panel components
        self.create_left_panel_components()
            
        # Add buttons to control panel
        self.create_buttons()
        
        # Add view mode selector
        self.create_view_mode_selector()
        
        # Add sliders
        self.create_sliders()
        
        # Create viewer area
        self.create_viewer()
        
        # Status bar
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        
        self.pixel_value_label = ttk.Label(self.status_bar, text="Pixel Value: ")
        self.pixel_value_label.pack(side=tk.LEFT)
        
        self.case_name_label = ttk.Label(self.status_bar, text="   Case: ")
        self.case_name_label.pack(side=tk.LEFT)
        
    def create_buttons(self):
        """Create control buttons"""
        button_frame = ttk.LabelFrame(self.control_panel, text="File Operations", padding=10)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        buttons = [
            ("Open Volume", self.open_volume),
            ("Open npy Image", self.open_image),
            ("Open dicom", self.open_dicom_case),
            ("Export as STL", self.export_stl_vtk),
            ("Blend mask", self.show_mask_overlay),
            ("Show 3D", self.open_3d_view)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(button_frame, text=text, command=command)
            btn.pack(fill=tk.X, pady=2)
            
    # customize this function according to structure of your dataset
    def on_series_select(self, event):
        """Handle selection of a series from the listbox"""
        selection = self.series_listbox.curselection()
        if selection:
            selected_folder = self.series_listbox.get(selection[0])
            full_path = os.path.join(self.current_folder_path, selected_folder)
            # Here you can add code to load the selected series
            print(f"Selected folder: {full_path}")  # Replace with your loading logic
            self.main_folder_path = full_path
            input_volume_path = os.path.join(self.current_folder_path, selected_folder,f"{selected_folder}_CT.nrrd")
            lower_teeth_mask_path = os.path.join(self.current_folder_path, selected_folder,f"{selected_folder}_Lower.nrrd")
            mand_mask_path = os.path.join(self.current_folder_path, selected_folder,f"{selected_folder}_Mand.nrrd")
            maxilla_mask_path = os.path.join(self.current_folder_path, selected_folder,f"{selected_folder}_Max.nrrd")
            upper_teeth_mask_path = os.path.join(self.current_folder_path, selected_folder,f"{selected_folder}_Upper.nrrd")
            
            # clearing any previous cache
            self._clear_volume_cache()
            self.image = None
            
            with ThreadPoolExecutor() as executor:
                future_volume = executor.submit(utilities.get_npy_from_nrrd_npy_path, input_volume_path)
                future_mask   = executor.submit(utilities.images_and_masks_to_npy, *(mand_mask_path, maxilla_mask_path, upper_teeth_mask_path, lower_teeth_mask_path, selected_folder))

                self.volume = future_volume.result()
                self.mask = future_mask.result()
            
            #self.volume = utilities.get_npy_from_nrrd_npy_path(input_volume_path)
            #input_volume_shape = self.volume.shape
            #self.mask = utilities.images_and_masks_to_npy( mand_mask_path,input_volume_shape, maxilla_mask_path, upper_teeth_mask_path, lower_teeth_mask_path, case_id=selected_folder)
            
            self._initialize_volume_settings()
            self._mask_colors_cache = None  # Clear cached colors
            self.update_slice(self.current_slice_index)
            self.file_name=selected_folder
            
    def create_left_panel_components(self):
        """Create components for the left panel"""
        # Button to select folder
        self.select_folder_btn = ttk.Button(
            self.left_panel, 
            text="Select Folder", 
            command=self.load_subfolders
        )
        self.select_folder_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Listbox to display subfolders
        self.series_listbox = tk.Listbox(
            self.left_panel,
            bg='#333333',
            fg='white',
            selectbackground='#555555',
            selectforeground='white'
        )
        self.series_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Bind double-click event
        self.series_listbox.bind("<Double-Button-1>", self.on_series_select)
        
    
    def create_view_mode_selector(self):
        """Create view mode radio buttons"""
        mode_frame = ttk.LabelFrame(self.control_panel, text="View Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        for mode in ["Axial", "Sagittal", "Coronal"]:
            rb = ttk.Radiobutton(
                mode_frame, 
                text=mode, 
                variable=self.view_mode,
                value=mode.lower(), 
                command=self.update_view
            )
            rb.pack(anchor=tk.W, pady=2)
    
    def create_sliders(self):
        """Create control sliders"""
        sliders_frame = ttk.LabelFrame(self.control_panel, text="Adjustments", padding=10)
        sliders_frame.pack(fill=tk.BOTH, expand=True)
        
        # Slice slider
        self.slice_label = ttk.Label(sliders_frame, text=f"Slice {self.current_slice_index}")
        self.slice_label.pack(anchor=tk.W)
        self.slice_slider = ttk.Scale(
            sliders_frame, 
            from_=0, 
            to=1, 
            orient=tk.VERTICAL,
            command=self.update_slice
        )
        self.slice_slider.pack(fill=tk.Y, expand=True, pady=5)
        
        # Window level/width sliders
        wl_frame = ttk.Frame(sliders_frame)
        wl_frame.pack(fill=tk.X, pady=5)
        
        self.window_level_label  = ttk.Label(wl_frame, text=f"Window Level: {self.window_level}")
        self.window_level_label.pack(anchor=tk.W)
        self.wl_scale = ttk.Scale(
            wl_frame,
            from_=-1000,
            to=4000,
            orient=tk.HORIZONTAL,
            command=self.update_wl
        )
        self.wl_scale.pack(fill=tk.X)
        
        ww_frame = ttk.Frame(sliders_frame)
        ww_frame.pack(fill=tk.X, pady=5)
        
        self.window_width_label=ttk.Label(ww_frame, text=f"Window Width: {self.window_width}")
        self.window_width_label.pack(anchor=tk.W)
        self.ww_scale = ttk.Scale(
            ww_frame,
            from_=1,
            to=4000,
            orient=tk.HORIZONTAL,
            command=self.update_ww
        )
        self.ww_scale.pack(fill=tk.X)
    
    def create_viewer(self):
        """Create the image viewer canvas"""
        viewer_frame = ttk.Frame(self.viewer_panel)
        viewer_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with border
        self.canvas = tk.Canvas(viewer_frame, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Motion>", self.update_pixel_values)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def on_canvas_resize(self, event):
        """Handle canvas resize events"""
        if hasattr(self, 'volume') and self.volume is not None:
            self.update_slice(self.current_slice_index)

    def load_subfolders(self):
        """Load subfolders of selected directory into listbox"""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.series_listbox.delete(0, tk.END)  # Clear current items
            subfolders = [f for f in os.listdir(folder_path) 
                        if os.path.isdir(os.path.join(folder_path, f))]
            for folder in sorted(subfolders):
                self.series_listbox.insert(tk.END, folder)
            self.current_folder_path = folder_path  # Store the current folder path

    def open_volume(self):
        """Load an nrrd or nifti file"""
        file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
        self.file_name= file_path.split("/")[-1]
        self.case_name_label.config(text=f"   {self.file_name}")
        
        if file_path:
            self._clear_volume_cache()
            self.image = None
            self.volume = utilities.get_npy_from_nrrd_npy_path(file_path)
            self._initialize_volume_settings()
            print(f"loading volime {self.file_name} , shape : {self.volume.shape}")

            
    def show_mask_overlay(self):
        """Load and display mask overlay"""
        file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
        if file_path:
            self.mask = utilities.get_npy_from_nrrd_npy_path(file_path)
            self._mask_colors_cache = None  # Clear cached colors
            self.update_slice(self.current_slice_index)
            
    def _initialize_volume_settings(self):
        """Common initialization for volume settings"""
        self.current_slice_index = 0
        self.unique_labels = np.unique(self.volume)
        
        max_val = max(self.unique_labels) if len(self.unique_labels) > 0 else 1
        self.wl_scale.config(from_=0, to=max_val - 1)
        self.wl_scale.set(max_val)
        self.ww_scale.set(max_val)

        self.slice_slider.config(from_=0, to=self.volume.shape[0] - 1)
        self.slice_slider.set(0)
        self.update_slice(0)

    def _clear_volume_cache(self):
        """Clear cached data when loading new volume"""
        self._mask_colors_cache = None
        if hasattr(self.canvas, 'photo_image'):
            self.canvas.photo_image = None

    def open_dicom_case(self):
        """Load DICOM series"""
        file_path = filedialog.askopenfilename(filetypes=[("Dicom files", "*.dcm")])
        if file_path:
            self._clear_volume_cache()
            self.image = None
            self.volume_type = "dicom"
            
            slices = []
            dicom_folder = os.path.dirname(os.path.abspath(file_path))
            
            for filename in os.listdir(dicom_folder):
                filepath = os.path.join(dicom_folder, filename)
                try:
                    slices.append(pydicom.dcmread(filepath))
                except Exception:
                    continue

            slices.sort(key=lambda s: int(s.InstanceNumber))
            self.volume = np.stack([s.pixel_array for s in slices])
            
            self.current_slice_index = 0
            self.wl_scale.set(self.window_level)
            self.ww_scale.set(self.window_width)
            
            self.slice_slider.config(from_=0, to=len(self.volume) - 1)
            self.slice_slider.set(0)
            self.update_slice(0)

    def open_image(self):
        """Load a 2D numpy image"""
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        if file_path:
            self._clear_volume_cache()
            self.image = np.load(file_path)
            self.current_slice_index = 0
            self.slice_slider.config(from_=0, to=1)
            self.slice_slider.set(0)
            self.update_slice(0)
            

            
    def update_view(self):
        """Update the view based on selected view mode"""
        self.update_slice(self.current_slice_index)
        
    def update_slice(self, val):
        """Update the displayed slice"""
        if isinstance(val, str):
            val = float(val)
        self.current_slice_index = int(val)
        
        if self.image is not None:
            slice_to_show = self.image
            self._display_slice(slice_to_show)
            return

        if self.volume is None:
            return

        # Get the appropriate slice based on view mode
        if self.view_mode.get() == "axial":
            slice_to_show = self.volume[self.current_slice_index, :, :]
            max_slices = self.volume.shape[0] - 1
        elif self.view_mode.get() == "sagittal":
            slice_to_show = self.volume[:, :, self.current_slice_index]
            max_slices = self.volume.shape[2] - 1
            if self.volume_type == "npy":
                slice_to_show = np.flipud(slice_to_show)
        elif self.view_mode.get() == "coronal":
            slice_to_show = self.volume[:, self.current_slice_index, :]
            max_slices = self.volume.shape[1] - 1
            if self.volume_type == "npy":
                slice_to_show = np.flipud(slice_to_show)

        self.slice_slider.config(to=max_slices)
        self.slice_label.config(text=f"Slice {self.current_slice_index}")
        self._display_slice(slice_to_show)

    @lru_cache(maxsize=32)
    def _get_windowed_slice(self, slice_data, window_level, window_width):
        """Apply windowing to slice data with caching"""
        window_min = window_level - window_width / 2
        window_max = window_level + window_width / 2
        
        if window_min >= window_max:
            window_min = 0
            window_max = 255
            
        clipped = np.clip(slice_data, window_min, window_max)
        return 255 * (clipped - window_min) / (window_max - window_min)

    def _display_slice(self, slice_to_show):
        """Display the processed slice on canvas"""
        if slice_to_show is None:
            return
        if self.volume is not None and self.mask is not None:
            if  self.volume.shape != self.mask.shape:
                print(f"volume and mask shapes are different , volume: {self.volume.shape} , mask: {self.mask.shape} ")
            
        # Apply windowing with caching
        windowed_slice = self._get_windowed_slice(
            tuple(slice_to_show.flatten()),  # Convert to tuple for hashability
            self.window_level,
            self.window_width
        ).reshape(slice_to_show.shape).astype(np.uint8)

        # Create base image
        image = Image.fromarray(windowed_slice)
        
        # Apply mask if available
        if self.mask is not None and hasattr(self, 'mask'):
            mask_slice = self._get_mask_slice()
            if mask_slice is not None:
                colored_mask = self._get_colored_mask(mask_slice)
                if colored_mask is not None:
                    mask_image = Image.fromarray(colored_mask, 'RGBA')
                    image = image.convert('RGBA')
                    image = Image.alpha_composite(image, mask_image)

        # Resize image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 0 and canvas_height > 0:
            if (canvas_width, canvas_height) != self._last_canvas_size:
                self._last_canvas_size = (canvas_width, canvas_height)
                
            image_ratio = image.width / image.height
            canvas_ratio = canvas_width / canvas_height
            
            if canvas_ratio > image_ratio:
                new_height = canvas_height
                new_width = int(image_ratio * new_height)
            else:
                new_width = canvas_width
                new_height = int(new_width / image_ratio)
                
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Display the image
        if hasattr(self.canvas, 'photo_image'):
            self.canvas.photo_image = None  # Clear old reference
            
        photo_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, 
                               anchor=tk.CENTER, image=photo_image)
        self.canvas.photo_image = photo_image  # Keep reference

    def _get_mask_slice(self):
        """Get the appropriate mask slice for current view"""
        if self.view_mode.get() == "axial":
            return self.mask[self.current_slice_index, :, :]
        elif self.view_mode.get() == "sagittal":
            mask_slice = self.mask[:, :, self.current_slice_index]
            return mask_slice if self.volume_type == "dicom" else np.flipud(mask_slice)
        elif self.view_mode.get() == "coronal":
            mask_slice = self.mask[:, self.current_slice_index, :]
            return mask_slice if self.volume_type == "dicom" else np.flipud(mask_slice)
        return None

    def _get_colored_mask(self, mask_slice):
        """Get colored mask with caching"""
        if self._mask_colors_cache is None:
            mask_labels = np.unique(self.mask).astype(np.uint8).tolist()
            self._mask_colors_cache = {
                label: color 
                for label, color in zip(
                    mask_labels, 
                    utilities.get_pathology_colors(mask_labels)
                )
            }
        
        colored_mask = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 4), dtype=np.uint8)
        mask_slice = mask_slice.astype(np.uint8)
        
        for label, color in self._mask_colors_cache.items():
            colored_mask[mask_slice == label] = color
            
        return colored_mask

    def update_wl(self, val):
        """Update window level"""
        self.window_level = int(float(val))
        self.window_level_label.config(text=f"Window Level: {self.window_level}")
        self._get_windowed_slice.cache_clear()  # Clear cached slices
        self.update_slice(self.current_slice_index)

    def update_ww(self, val):
        """Update window width"""
        self.window_width = int(float(val))
        self.window_width_label.config(text=f"Window Width: {self.window_width}")
        self._get_windowed_slice.cache_clear()  # Clear cached slices
        self.update_slice(self.current_slice_index)

    def update_pixel_values(self, event):
        """Update pixel value display"""
        if self.volume is None:
            return

        x, y = event.x, event.y
        try:
            # Adjust coordinates based on image scaling
            img_width = self.canvas.winfo_width()
            img_height = self.canvas.winfo_height()
            
            # Get the displayed image dimensions
            if hasattr(self.canvas, 'photo_image'):
                disp_width = self.canvas.photo_image.width()
                disp_height = self.canvas.photo_image.height()
                
                # Calculate actual image position
                x_img = int((x - (img_width - disp_width) // 2) * self.volume.shape[2] / disp_width)
                y_img = int((y - (img_height - disp_height) // 2) * self.volume.shape[1] / disp_height)
                
                if 0 <= x_img < self.volume.shape[2] and 0 <= y_img < self.volume.shape[1]:
                    if self.view_mode.get() == "axial":
                        pixel_value = self.volume[self.current_slice_index, y_img, x_img]
                    elif self.view_mode.get() == "sagittal":
                        pixel_value = self.volume[y_img, x_img, self.current_slice_index]
                    elif self.view_mode.get() == "coronal":
                        pixel_value = self.volume[y_img, self.current_slice_index, x_img]
                    
                    self.pixel_value_label.config(
                        text=f"Pixel Value: {pixel_value}, Location: ({x_img}, {y_img})"
                    )
                    return
        except (IndexError, AttributeError):
            pass
            
        self.pixel_value_label.config(text="Pixel Value: N/A")

    def on_mousewheel(self, event):
        """Handle mouse wheel events with throttling"""
        current_time = time.time()
        if current_time - self._last_wheel_time < 0.1:  # 100ms throttle
            return
        self._last_wheel_time = current_time
        
        delta = 1 if event.delta > 0 else -1
        new_index = self.current_slice_index + delta
        
        # Clamp to valid range
        max_slices = self.slice_slider.cget("to")
        new_index = max(0, min(new_index, max_slices))
        
        if new_index != self.current_slice_index:
            self.current_slice_index = new_index
            self.slice_slider.set(new_index)
            self.update_slice(new_index)

    def export_stl_vtk(self):
        """Export volume as STL"""
        if self.volume is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".stl", 
                filetypes=[("STL files", "*.stl")]
            )
            if file_path:
                utilities.export_volume_as_stl_vtk(
                    self.volume, file_path,
                    self.window_level, self.window_width
                )

    def open_3d_view(self):
        """Show 3D view of volume"""
        if self.volume is not None:
            plotter = utilities.npy_to_pyvista(self.volume)
            plotter.show()


def main():
    root = tk.Tk()
    volume_viewer = VolumeViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()