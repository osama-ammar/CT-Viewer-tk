import numpy as np
import vtk
import pyvista as pv


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


def npy_to_pyvista(volume):
    # Use PyVista's color map (or specify your own)
    plotter = pv.Plotter()
    _ = plotter.add_volume(
        volume,
        cmap="bone",
        opacity="sigmoid_9",
        show_scalar_bar=False,
    )

    plotter.view_zx()
    plotter.camera.up = (0, 0, 1)
    plotter.camera.zoom(1.3)

    return plotter



def open_3d_view(volume,window_level,window_width):

    # Adjust window level and window width
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2

    # Ensure window_min is not greater than window_max (fix any bad configurations)
    if window_min >= window_max:
        window_min = 0
        window_max = 255
    # pyvista
    normalized_volume=normalize_volume(volume,window_level,window_width)
    volume = pv.wrap(normalized_volume)
    
    # Step 3: Plotting the volume
    plotter = pv.Plotter()
    
    # Render with custom settings
    plotter.camera_position = 'iso'  # Iso view for better depth
    
    return plotter
