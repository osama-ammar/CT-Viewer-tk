from time import time

import numpy as np
from nilearn import image
from skimage import draw, filters, exposure, measure
from scipy import ndimage

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import  html
from dash import  dcc
from dash_slicer import VolumeSlicer

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, update_title=None, external_stylesheets=external_stylesheets)
server = app.server


t1 = time()

# ------------- I/O and data massaging ---------------------------------------------------

# Add argparse for command-line argument parsing
import argparse

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the medical image viewer app.")
    parser.add_argument("volume_path", type=str, help="Path to the medical image volume.")
    return parser.parse_args()

# Parse command-line arguments
#args = parse_arguments()
#args.volume_path)

img = image.load_img("D:\\Code_store\\Radiosh-Tracker-1\\radiosh_app_project\\radiosh\\CT_Images\\assets\\radiopaedia_org_covid-19-pneumonia-7_85703_0-dcm.nii")
mat = img.affine
img = img.get_fdata()
#print(img.shape)
img = np.copy(np.moveaxis(img, 1, 2))[:, ::-1]
img = np.copy(np.moveaxis(img, 0, 2))[:, ::-1]
img = np.copy(np.moveaxis(img, 1, 2))[:, ::-1]
#print(img.shape)
spacing = abs(mat[2, 2]), abs(mat[1, 1]), abs(mat[0, 0])

# Create smoothed image and histogram
med_img = filters.median(img, footprint=np.ones((1, 3, 3), dtype=bool))
hi = exposure.histogram(med_img)

# Create mesh
verts, faces, _, _ = measure.marching_cubes(med_img, 200, step_size=5)
x, y, z = verts.T
i, j, k = faces.T
fig_mesh = go.Figure()
fig_mesh.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i))

# Create slicers
slicer1 = VolumeSlicer(app, img, axis=0, spacing=spacing, thumbnail=False)
slicer1.graph.figure.update_layout(dragmode="drawclosedpath", newshape_line_color="cyan", plot_bgcolor="rgb(0, 0, 0)")
slicer1.graph.config.update(modeBarButtonsToAdd=["drawclosedpath", "eraseshape",])


slicer2 = VolumeSlicer(app, img, axis=2, spacing=spacing, thumbnail=False)
slicer2.graph.figure.update_layout(dragmode="drawrect", newshape_line_color="cyan", plot_bgcolor="rgb(0, 0, 0)")
slicer2.graph.config.update(modeBarButtonsToAdd=["drawrect", "eraseshape",])

#########################################################################
#creat x-ray image figure


# Constants

from PIL import Image
import plotly.express as px
x_ray_image = Image.open("D:\\chest.jpg")
image_shape = x_ray_image.size
#print("Image shape:", image_shape)
x_ray_fig_px = px.imshow(x_ray_image, binary_string=True)
# Update layout to show the full size and enable annotations
x_ray_fig_px.update_layout(
    width=image_shape[0],  # Set width to image width
    height=image_shape[1],  # Set height to image height
    dragmode="drawrect",  # Enable rectangle annotation
    newshape=dict(line=dict(color="cyan")),  # Set annotation line color to cyan
)
annotations = x_ray_fig_px.layout.shapes

# Print the coordinates of each rectangle
for annotation in annotations:
    print(annotation)
    if annotation.type == 'rect':
        print("Rectangle coordinates ", annotation.x0, annotation.y0, annotation.x1, annotation.y1)

################################################################################################

histogram = dcc.Graph(
                    id="graph-histogram",
                    figure=px.bar(
                        x=hi[1],
                        y=hi[0],
                        labels={"x": "intensity", "y": "count"},
                        template="plotly_white",
                    ),
                    config={
                        "modeBarButtonsToAdd": [
                            "drawline",
                            "drawclosedpath",
                            "drawrect",
                            "eraseshape",
                        ]
                    },
                )

def path_to_coords(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point"""
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.array(indices_str, dtype=float)


def largest_connected_component(mask):
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (np.argmax(sizes) + 1)


t2 = time()
print("initial calculations", t2 - t1)

# ------------- Define App Layout ---------------------------------------------------
axial_card = dbc.Card(
    [
        dbc.CardHeader("Axial view of the lung"),
        dbc.CardBody([slicer1.graph, slicer1.slider, *slicer1.stores]),
        dbc.CardFooter(
            [
                html.H6(
                    [
                        "Step 1: Draw a rough outline that encompasses all ground glass occlusions across ",
                        html.Span(
                            "all axial slices",
                            id="tooltip-target-1",
                            className="tooltip-target",
                        ),
                        ".",
                    ]
                ),
                dbc.Tooltip(
                    "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                    target="tooltip-target-1",
                ),
            ]
        ),
    ]
)

saggital_card = dbc.Card(
    [
        dbc.CardHeader("Sagittal view of the lung"),
        dbc.CardBody([slicer2.graph, slicer2.slider, *slicer2.stores]),
        dbc.CardFooter(
            [
                html.H6(
                    [
                        "Step 2:\n\nDraw a rectangle to determine the ",
                        html.Span(
                            "min and max height ",
                            id="tooltip-target-2",
                            className="tooltip-target",
                        ),
                        "of the occlusion.",
                    ]
                ),
                dbc.Tooltip(
                    "Only the min and max height of the rectangle are used, the width is ignored",
                    target="tooltip-target-2",
                ),
            ]
        ),
    ]
)

histogram_card = dbc.Card(
    [
        dbc.CardHeader("Histogram of intensity values"),
        dbc.CardBody(
            [histogram ]
        ),
        dbc.CardFooter(
            [
                dbc.Toast(
                    [
                        html.P(
                            "Before you can select value ranges in this histogram, you need to define a region"
                            " of interest in the slicer views above (step 1 and 2)!",
                            className="mb-0",
                        )
                    ],
                    id="roi-warning",
                    header="Please select a volume of interest first",
                    icon="danger",
                    is_open=True,
                    dismissable=False,
                ),
                "Step 3: Select a range of values to segment the occlusion. Hover on slices to find the typical "
                "values of the occlusion.",
            ]
        ),
    ]
)

mesh_card = dbc.Card(
    [
        dbc.CardHeader("3D mesh representation of the image data and annotation"),
        dbc.CardBody([dcc.Graph(id="graph-helper", figure=fig_mesh)]),

        
    ]
)




image_card = dbc.Card(
    [
        dbc.CardHeader("X-Ray"),
        dbc.CardBody([
            dcc.Graph(figure=x_ray_fig_px,id='x-ray-graph',style={'width': '100%', 'height': '100%'})
            # Add other dbc components as needed
        ]),
        dbc.Button("Print Something", id="print-button", color="primary", className="mr-1", n_clicks=0),
        html.Div(id="print-output"),  # Placeholder for the output of the print function
        dbc.CardFooter(
            [
                html.H6(
                    [
                        "Step 1: Draw a rough outline that encompasses all ground glass occlusions across ",
                    ]
                ),
                dbc.Tooltip(
                    "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                    target="tooltip-target-1",
                ),
            ]
        ),
    ]
)


# Access the children of the image_card
children = image_card.children

# Initialize variables to store image size and annotations
image_size = None
annotations = None

# Iterate through the children to find the dcc.Graph component
for child in children:
    if isinstance(child, dcc.Graph):
        # Get the figure object of the dcc.Graph component
        figure = child.figure
        
        # Extract image size and annotations from the figure
        layout = figure.get('layout', {})
        image_size = layout.get('width'), layout.get('height')
        annotations = figure.get('layout', {}).get('annotations')

# Print image size and annotations
print("Image size:", image_size)
print("Annotations:", annotations)








# Define Modal
with open("data/assets/modal.md", "r") as f:
    howto_md = f.read()




app.layout = html.Div(
    [

        dbc.Container(
            [
                dbc.Row([dbc.Col(axial_card), dbc.Col(saggital_card)]),
                dbc.Row([dbc.Col(histogram_card), dbc.Col(mesh_card),]),
                dbc.Row([dbc.Col(image_card), ]),
                
            ],
            fluid=True,
        ),
        dcc.Store(id="annotations", data={}),
        dcc.Store(id="occlusion-surface", data={}),
    ],
)

t3 = time()
print("layout definition", t3 - t2)


# ------------- Define App Interactivity ---------------------------------------------------
@app.callback(
    [Output("graph-histogram", "figure"), Output("roi-warning", "is_open")],
    [Input("annotations", "data")],
)
def update_histo(annotations):
    print("slicer1.overlay_data:",slicer1.overlay_data)
    if (
        annotations is None
        or annotations.get("x") is None
        or annotations.get("z") is None
    ):
        return dash.no_update, dash.no_update
    # Horizontal mask for the xy plane (z-axis)
    path = path_to_coords(annotations["z"]["path"])
    rr, cc = draw.polygon(path[:, 1] / spacing[1], path[:, 0] / spacing[2])
    if len(rr) == 0 or len(cc) == 0:
        return dash.no_update, dash.no_update
    mask = np.zeros(img.shape[1:])
    mask[rr, cc] = 1
    mask = ndimage.binary_fill_holes(mask)
    # top and bottom, the top is a lower number than the bottom because y values
    # increase moving down the figure
    top, bottom = sorted([int(annotations["x"][c] / spacing[0]) for c in ["y0", "y1"]])
    intensities = med_img[top:bottom, mask].ravel()
    if len(intensities) == 0:
        return dash.no_update, dash.no_update
    hi = exposure.histogram(intensities)
    fig = px.bar(
        x=hi[1],
        y=hi[0],
        # Histogram
        labels={"x": "intensity", "y": "count"},
    )
    fig.update_layout(dragmode="select", title_font=dict(size=20, color="blue"))
    return fig, False


@app.callback(
    [
        Output("occlusion-surface", "data"),
        Output(slicer1.overlay_data.id, "data"),
        Output(slicer2.overlay_data.id, "data"),
    ],
    [Input("graph-histogram", "selectedData"), Input("annotations", "data")],
)
def update_segmentation_slices(selected, annotations):
    ctx = dash.callback_context
    # When shape annotations are changed, reset segmentation visualization
    if (
        ctx.triggered[0]["prop_id"] == "annotations.data"
        or annotations is None
        or annotations.get("x") is None
        or annotations.get("z") is None
    ):
        mask = np.zeros_like(med_img)
        overlay1 = slicer1.create_overlay_data(mask)
        overlay2 = slicer2.create_overlay_data(mask)
        return go.Mesh3d(), overlay1, overlay2
    elif selected is not None and "range" in selected:
        if len(selected["points"]) == 0:
            return dash.no_update
        v_min, v_max = selected["range"]["x"]
        t_start = time()
        # Horizontal mask
        path = path_to_coords(annotations["z"]["path"])
        rr, cc = draw.polygon(path[:, 1] / spacing[1], path[:, 0] / spacing[2])
        mask = np.zeros(img.shape[1:])
        mask[rr, cc] = 1
        mask = ndimage.binary_fill_holes(mask)
        # top and bottom, the top is a lower number than the bottom because y values
        # increase moving down the figure
        top, bottom = sorted(
            [int(annotations["x"][c] / spacing[0]) for c in ["y0", "y1"]]
        )
        img_mask = np.logical_and(med_img > v_min, med_img <= v_max)
        img_mask[:top] = False
        img_mask[bottom:] = False
        img_mask[top:bottom, np.logical_not(mask)] = False
        img_mask = largest_connected_component(img_mask)
        # img_mask_color = mask_to_color(img_mask)
        t_end = time()
        print("build the mask", t_end - t_start)
        t_start = time()
        # Update 3d viz
        verts, faces, _, _ = measure.marching_cubes(
            filters.median(img_mask, selem=np.ones((1, 7, 7))), 0.5, step_size=3
        )
        t_end = time()
        print("marching cubes", t_end - t_start)
        x, y, z = verts.T
        i, j, k = faces.T
        trace = go.Mesh3d(x=z, y=y, z=x, color="red", opacity=0.8, i=k, j=j, k=i)
        overlay1 = 0#slicer1.create_overlay_data(img_mask)
        overlay2 = slicer2.create_overlay_data(img_mask)
        # todo: do we need an output to trigger an update?
        return trace, overlay1, overlay2
    else:
        return (dash.no_update,) * 3


@app.callback(
    Output("annotations", "data"),
    [Input(slicer1.graph.id, "relayoutData"), Input(slicer2.graph.id, "relayoutData"),],
    [State("annotations", "data")],
)
def update_annotations(relayout1, relayout2, annotations):
    if relayout1 is not None and "shapes" in relayout1:
        if len(relayout1["shapes"]) >= 1:
            shape = relayout1["shapes"][-1]
            annotations["z"] = shape
        else:
            annotations.pop("z", None)
    elif relayout1 is not None and "shapes[2].path" in relayout1:
        annotations["z"]["path"] = relayout1["shapes[2].path"]

    if relayout2 is not None and "shapes" in relayout2:
        if len(relayout2["shapes"]) >= 1:
            shape = relayout2["shapes"][-1]
            annotations["x"] = shape
        else:
            annotations.pop("x", None)
    elif relayout2 is not None and (
        "shapes[2].y0" in relayout2 or "shapes[2].y1" in relayout2
    ):
        annotations["x"]["y0"] = relayout2["shapes[2].y0"]
        annotations["x"]["y1"] = relayout2["shapes[2].y1"]
    return annotations


app.clientside_callback(
    """
function(surf, fig){
        let fig_ = {...fig};
        fig_.data[1] = surf;
        return fig_;
    }
""",
    output=Output("graph-helper", "figure"),
    inputs=[Input("occlusion-surface", "data"),],
    state=[State("graph-helper", "figure"),],
)

# Define callback to update the Div with coordinates
@app.callback(
    Output("coordinate-output", "children"),
    [Input("x_ray_fig_px", "relayoutData")]
)
def update_coordinates(relayout_data):
    print("update_coordinatesupdate_coordinatesupdate_coordinatesupdate_coordinates")
    if "shapes" in relayout_data:
        shapes = relayout_data["shapes"]
        rectangles = [shape for shape in shapes if shape["type"] == "rect"]
        coordinates = [f"Rectangle {i+1}: ({rectangle['x0']}, {rectangle['y0']}) - ({rectangle['x1']}, {rectangle['y1']})" for i, rectangle in enumerate(rectangles)]
        print(coordinates)
        return [html.P(coord) for coord in coordinates]
    return []


@app.callback(
    Output("modal", "is_open"),
    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


def print_annotations(n_clicks, figure):
    if n_clicks > 0:
        annotations = figure.get('layout', {}).get('annotations')
        if annotations:
            print("Annotations:")
            for annotation in annotations:
                print(annotation)
        else:
            print("No annotations found.")

# Define callback to print something when the button is clicked
@app.callback(
    Output("print-output", "children"),
    [Input("print-button", "n_clicks")],
    [Input("x-ray-graph", "figure")]
)
def update_output(n_clicks, figure):
    if n_clicks > 0:
        print_annotations(n_clicks, figure)
        return "Annotations printed in the console."

# osama 
# TODO : clearing the steps
# make a very simple app that take image (chest) and get sigmentation overlayed in it ----just this

if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)