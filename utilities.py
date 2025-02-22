import numpy as np
import vtk
import pyvista as pv
from vtkmodules.util import numpy_support #very tricky issue (using vtkmodules.util insted of  vtk.util ) both will work but when converting to exe vtk.util will not work
import nibabel as nib
import matplotlib.pyplot as plt





# TODO : make normalization compatible wih different pixels ranges (0 to 1) | (0 to 255) |(-1000 to 5000)
def normalize_volume(npy_volume,window_level,window_width):
    # Adjust window level and window width
    normalized_volume=npy_volume.copy()
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2
    for i, image in enumerate(normalized_volume) :
        # Clip values to the window level and width
        image = np.clip(image, window_min, window_max)
        image=image/window_max
        normalized_volume[i]=image

    return normalized_volume

def fix_mask_labels(mask_array):
    # whae mask labels are not consistent like [1,44,54,55,90..] ----> we need to make them 1,2,3,4,5 for visualization
    for i , label in enumerate(np.unique(mask_array)):
        mask_array[mask_array==label]=i
    return mask_array    
    
    
    
    
def create_auto_label_lookup_table(label_values, colormap="tab10"):
    """
    Create a PyVista Lookup Table (LUT) for given label values with automatically assigned colors.
    Parameters:
    - label_values (list of int): Unique label values.
    - colormap (str): Matplotlib colormap name (default: "tab10").
    Returns:
    - pv.LookupTable: PyVista Lookup Table with assigned colors.
    """
    num_labels = len(label_values)
    cmap = plt.get_cmap("tab20", num_labels)  # Get distinct colors
    colors = (cmap(np.arange(num_labels))[:, :3] * 255).astype(np.uint8)  # Convert to RGB 0-255

    lut = pv.LookupTable()
    lut.SetNumberOfTableValues(num_labels)

    for i, color in enumerate(colors):
        lut.SetTableValue(i, *color/ 255 , 1.0)  # Normalize and set RGBA
        print(i,*color / 255)
    return lut


def npy_array_to_pyvista_data(npy_array , spacing=(1.0, 1.0, 1.0),origin = (0.0, 0.0, 0.0) ):
    
    # Create the UniformGrid
    grid = pv.ImageData()
    grid.dimensions =  np.array(npy_array.shape) + 1
    grid.spacing = spacing
    grid.origin = origin
    
    # Flatten the image array and add it to the grid's cell data
    grid.cell_data['image'] = npy_array.flatten(order="F")
    
    unique_voxel_values= np.unique(npy_array)
    
    return grid , unique_voxel_values



# Function to clip the volume and update the plot
def update_clipping(volume ,plotter, min_val=None, max_val=None):
    # Clip the volume to the range
    pass
    

from pyvista import examples
def npy_to_pyvista(volume,mode="madsk"):
    # Use PyVista's color map (or specify your own)
    
    plotter = pv.Plotter()
    
    if mode=="mask":
        # labels mask
        volume=fix_mask_labels(volume)
        #print(np.unique(volume))
        label_map , unique_voxel_values = npy_array_to_pyvista_data(volume)

        if label_map.cell_data:
            label_map = label_map.cell_data_to_point_data()

        custom_lut  = create_auto_label_lookup_table(unique_voxel_values)
        labels_mesh = label_map.contour_labeled(smoothing=True)
        _ = plotter.add_mesh(labels_mesh, cmap=custom_lut, show_scalar_bar=True)
        
        # very very important interaction (work when using labels_mesh pyvista data object )
        #plotter.add_mesh_slice(volume, assign_to_axis='z', interaction_event=vtk.vtkCommand.InteractionEvent)
        #plotter.add_volume_clip_plane(volume, normal='-x', cmap='magma')
        #plotter.add_mesh_clip_plane(label_map)
    else:
        volume=fix_mask_labels(volume)
        
        plotter = pv.Plotter()
        _ = plotter.add_volume(
            volume,
            cmap="tab20",
            opacity="linear",
            show_scalar_bar=True,
        )


    
    
    plotter.view_zx()
    plotter.camera.up = (0, 0, 1)
    plotter.camera.zoom(1.3)


    return plotter



def export_volume_as_stl_vtk(volume,file_path,window_level,window_width):
    
    if volume is not None:
        # Create a VTK image data
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(volume.shape[::-1])
        vtk_image.SetSpacing(1, 1, 1)
        vtk_image.SetOrigin(0, 0, 0)

        # Copy the NumPy array to VTK image data
        normalized_volume=normalize_volume(volume,window_level,window_width)
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
        

def createMIP(np_img, slices_num = 15):
    ''' create the mip image from original image, slice_num is the number of 
    slices for maximum intensity projection'''
    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i-slices_num)
        np_mip[i,:,:] = np.amax(np_img[start:i+1],0)
    return np_mip


def load_nifti(filepath):
    """
    Loads a NIfTI (.nii, .nii.gz) medical image file and extracts voxel data and spacing.

    Args:
        filepath (str): Path to the NIfTI file.

    Returns:
        tuple: (numpy.ndarray, voxel_spacing)
            - Image array (H, W, D) or (H, W, D, C) if multi-channel.
            - Voxel spacing in mm as a tuple (sx, sy, sz).
    """
    img = nib.load(filepath)
    data = np.array(img.get_fdata())  # Image data in NumPy format

    # Extract voxel spacing from the affine transformation matrix
    voxel_spacing = tuple(img.header.get_zooms())  # (sx, sy, sz)

    return data, voxel_spacing

def export_npy(npy_volume,save_path):
    pass

def export_nrrd(npy_volume,save_path):
    pass

def export_tiff(npy_volume,save_path):
    pass

def export_nifti(npy_volume,save_path):
    pass

def segmentation_volume_fuse(npy_volume ,npy_mask ):
    pass

