import numpy as np
import vtk
import pyvista as pv
from vtkmodules.util import numpy_support #very tricky issue (using vtkmodules.util insted of  vtk.util ) both will work but when converting to exe vtk.util will not work


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



# Function to clip the volume and update the plot
def update_clipping(volume ,plotter, min_val=None, max_val=None):
    # Clip the volume to the range
    pass
    
    
def npy_to_pyvista(volume):
    # Use PyVista's color map (or specify your own)
    plotter = pv.Plotter()
    _ = plotter.add_volume(
        volume,
        cmap="bone",
        opacity="sigmoid_9",
        show_scalar_bar=False,
    )

    plotter.add_slider_widget(
        lambda min_val : update_clipping(volume, plotter,min_val=min_val),  # Update min value
        [volume.min(), volume.max()],  # Slider range based on volume data
        value=volume.min(),  # Initial min value
        title="min",
        tube_width=0.002,
        slider_width = 0.002,
        style="classic",  # Modern style slider
        pointa=(0.02, 0.9),  # Position of one end of the slider
        pointb=(0.3, 0.9),  # Position of the other end of the slider
        
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


def createMIP(np_img, slices_num = 15):
    ''' create the mip image from original image, slice_num is the number of 
    slices for maximum intensity projection'''
    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i-slices_num)
        np_mip[i,:,:] = np.amax(np_img[start:i+1],0)
    return np_mip