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



import numpy as np
import pyvista as pv
def create_auto_label_lookup_table(unique_labels):
    """Create a lookup table for labels with random colors in the range [0, 1]."""
    import random
    lut = {}
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        lut[label] = [random.random(), random.random(), random.random()]

    return lut


import numpy as np
import pyvista as pv

def npy_to_colored_mesh(volume, output_file="output.ply"):
    # Fix mask labels (if needed)
    volume = fix_mask_labels(volume)

    # Convert the volume to a PyVista dataset
    label_map, unique_voxel_values = npy_array_to_pyvista_data(volume)

    if label_map.cell_data:
        label_map = label_map.cell_data_to_point_data()

    # Create a lookup table for colors
    custom_lut = {
        1.0: [0.7752098806867631, 0.1306064383690987, 0.8437576388762167],
        2.0: [0.8008656386418762, 0.3294719428647084, 0.14044659884433752],
        3.0: [0.3387886440096143, 0.27614631931015277, 0.04894469244317201],
        4.0: [0.423270741111495, 0.03230702211236092, 0.8815410748921367],
        5.0: [0.09741415679782917, 0.18762531319554532, 0.9772314746248427],
        6.0: [0.9266788796126396, 0.7194148867701912, 0.16177991821960125],
        7.0: [0.2420561524569047, 0.579772339975456, 0.5752011694238277],
        8.0: [0.8340624185572977, 0.11644644811808558, 0.1780692798478587],
        9.0: [0.7591728627293507, 0.24907286650833804, 0.401979254267959],
        10.0: [0.5502268844974432, 0.8679952905581659, 0.14503719156594264],
        11.0: [0.05603794782167015, 0.38446552611622276, 0.4489681958476418],
        12.0: [0.5968552223947956, 0.3360718000121661, 0.5344166977655997],
        13.0: [0.12828977064102443, 0.8946793436768321, 0.05215896602597103],
        14.0: [0.9052368992946301, 0.7496235858280192, 0.2900096807357414],
        15.0: [0.5685592844808667, 0.5823765411904269, 0.19288958056930627],
        16.0: [0.8877469030377194, 0.21347936934119127, 0.18784477100160013],
        17.0: [0.17894755864544187, 0.7877600339617247, 0.6050779321116121],
    }

    # Extract meshes for each label
    combined_mesh = pv.PolyData()
    for label in unique_voxel_values:
        if label == 0:  # Skip background
            continue

        # Extract the mesh for the current label
        single_label_mesh = label_map.threshold([label - 0.5, label + 0.5])

        # Convert the mesh to PolyData if it's not already
        if not isinstance(single_label_mesh, pv.PolyData):
            single_label_mesh = single_label_mesh.extract_surface().triangulate()

        # Assign a color to the mesh based on the label
        color = custom_lut.get(label, [0.5, 0.5, 0.5])  # Default to gray if label not in custom_lut

        # Convert color to uint8 in the range [0, 255]
        color_uint8 = (np.array(color) * 255).astype(np.uint8)
        single_label_mesh["colors"] = np.tile(color_uint8, (single_label_mesh.n_points, 1))

        # Append the mesh to the combined mesh
        combined_mesh += single_label_mesh

    # Save the combined mesh with colors
    if output_file.endswith(".ply"):
        combined_mesh.save(output_file, texture="colors")
    elif output_file.endswith(".obj"):
        combined_mesh.save(output_file, texture="colors")
    else:
        raise ValueError("Unsupported file format. Use .ply or .obj.")

    print(f"Saved colored mesh to {output_file}")

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

