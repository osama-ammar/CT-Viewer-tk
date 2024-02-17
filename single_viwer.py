import dash
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html, Input, Output, no_update, callback
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go
import json
import numpy as np
from PIL import Image
import plotly.express as px
import datetime

"""
TODO :
--------
[x] - view image and mask
[ ] - view every part of the mask with different color 
[ ] - get good layout for portofolio
[ ] - clean code (remove unneeded code - organize and document)
[ ] - get the area of mask segments and view it 

"""

# pathes and configs
image_path = "D:\\chest-x-ray.jpeg"
onnx_model_path="D:\\Code_store\\CT-Viewer-tk\\unet-2v.onnx"




#getting image
x_ray_image = Image.open(image_path)
x_ray_image = np.array(x_ray_image.resize((512, 512)))
image_shape = x_ray_image.shape
print("Image shape:", image_shape)

# image in px
x_ray_fig_px = px.imshow(x_ray_image, binary_string=True)
# Update layout to show the full size and enable annotations
x_ray_fig_px.update_layout(
    width=image_shape[0],  # Set width to image width
    height=image_shape[1],  # Set height to image height
    dragmode="drawrect",  # Enable rectangle annotation
    newshape=dict(line=dict(color="cyan")),  # Set annotation line color to cyan
)

#img in fig
# Define your X-ray figure (replace this with your actual figure)
x_ray_figure = go.Figure()
x_ray_figure.add_layout_image(layer="below",sizing="stretch",source=image_path, 
    x=0,
    y=1,
    xref="x",
    yref="y",
    sizex=12,
    sizey=12,)

x_ray_figure.update_layout(
    width=image_shape[0] * 1.5,
    height=image_shape[0] * 1,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
    dragmode="drawrect",  # Enable rectangle annotation
    newshape=dict(line=dict(color="cyan")),  # Set annotation line color to cyan
)
##############################
# Define your Dash app
#############################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

config = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ]
}


image_card = dbc.Card(
    [
        dbc.CardHeader("X-Ray Image"),
        dbc.CardBody([
            dcc.Graph(
                id='input_image_id',  # Set an ID for the graph
                figure=x_ray_fig_px,
                responsive ='auto',
                style={'width': '100%', 'height': '100%'},
                config=config  # Enable shape editing
                    ),
                ]),
        dbc.Button("Show Mask", id="show-mask-button", color="primary", className="mr-1", n_clicks=0),
            
        dbc.CardFooter(
            [
                html.H6("Step 1: Draw a rough outline that encompasses all ground glass occlusions."),
                dbc.Tooltip(
                    "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                    target="x-ray-slider",  # Ensure this matches the ID of the target element
                ),
            ]
        ),
    ],
    style={'width': '100%', 'height': '100%'}  # Set card width to 100% and height to 100vh (viewport height)
)



# Define the mask card layout
mask_image_card = dbc.Card(
    [
        dbc.CardHeader("X-Ray mask"),
        dbc.CardBody([
            dcc.Graph(
                id='mask_image_id',  # Set an ID for the graph
                figure=x_ray_figure,
                responsive ='auto',
                config=config  # Enable shape editing
            ),

            dbc.Button("Print Annotations", id="print-button", color="primary", className="mr-1", n_clicks=0),
            
            html.Div(id="annotations-data", style={'display': 'none'}),  # Hidden div to store annotations data
            
            html.Div(id="print-output") 
            
            
        ]),
        dbc.CardFooter(
            [
                html.H6("Step 1: Draw a rough outline that encompasses all ground glass occlusions."),
                dbc.Tooltip(
                    "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                    target="x-ray-slider",  # Ensure this matches the ID of the target element
                ),
            ]
        ),
    ],
    style={'width': '100%', 'height': '100%'}  # Set card width to 100% and height to 100vh (viewport height)
)


upload_image_card=dbc.Card(
    [
        dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
            ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
            },
        # Allow multiple files to be uploaded
        multiple=True),
        html.Div(id='output-image-upload')
    ])


####################
# Define app layout
app.layout = html.Div(
    [
        dbc.Row([dbc.Col(image_card, width=5),dbc.Col(mask_image_card, width=5)]),
        dbc.Row([dbc.Col(upload_image_card, width=5),])
        
    ]
)


###########################
# helper functions 
###########################
def print_annotations(n_clicks, figure):
    if n_clicks > 0:
        annotations = figure.get('layout', {}).get('annotations')
        if annotations:
            print("Annotations:")
            for annotation in annotations:
                print(annotation)
        else:
            print("No annotations found.")

#prepare input image for model inference
def prepare_model_input(img_path):
    # Load the image and resize it
    input_image = Image.open(img_path)

    input_image = np.array(input_image.resize((512, 512)))
    input_image=np.transpose(input_image, (2, 0, 1))
    #print(input_image.shape)
    # Convert to NumPy array and normalize pixel values
    input_image = input_image.astype(np.float32) / 255.0
    # Adjust array values
    input_image -= 0.5
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    # Convert to single-channel image
    input_image = np.mean(input_image, axis=1, keepdims=True)
    #print(input_image.shape)
    return input_image


import onnxruntime
def model_inference(onnx_model_path , input_array):
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: input_array}
    ort_output = ort_session.run(None, ort_inputs)[0]
    #print(ort_output)
    return ort_output


def show_mask_on_image(image_path,onnx_model_path):
    input_image=prepare_model_input(image_path)
    output_mask = model_inference(onnx_model_path , input_image)
    #print(output_mask.shape)
    return output_mask



# def parse_uploaded_image(contents, filename, date):
#     return html.Div([
#         html.H5(filename),
#         html.H6(datetime.datetime.fromtimestamp(date)),

#         # HTML images accept base64 encoded strings in the same format
#         # that is supplied by the upload
#         html.Img(src=contents),
#         html.Hr(),
#         html.Div('Raw Content'),
#         html.Pre(contents[0:200] + '...', style={
#             'whiteSpace': 'pre-wrap',
#             'wordBreak': 'break-all'
#         })
#     ])

###########################
#Callbacks
# Define callback to print something when the button is clicked
@app.callback(
    Output("print-output", "children"),
    [Input("mask_image_id", "relayoutData")],
    prevent_initial_call=True,
)
def update_output(relayout_data):
    if "shapes" in relayout_data:
        
        output_json = json.dumps(relayout_data["shapes"], indent=2)
        if len(relayout_data["shapes"])>0:
            print(relayout_data["shapes"][0]['x0'])
        return output_json
    else:
        return no_update




# Callback to update mask overlay when button is clicked
@app.callback(
    Output('mask_image_id', 'figure'),
    [Input('show-mask-button', 'n_clicks')],
    [State('mask_image_id', 'figure')]
)
def update_mask_overlay(n_clicks,current_figure):
    if n_clicks > 0:
        # Update the figure data with mask overlay
        output_mask = show_mask_on_image(image_path,onnx_model_path)
        output_mask=output_mask.reshape((2, 512, 512))
        # Compute softmax along the appropriate axis
        output_mask = np.exp(output_mask) / np.sum(np.exp(output_mask), axis=0)
        # Find the index of the maximum value along the specified axis (e.g., axis=1 for channels)
        output_mask = np.argmax(output_mask, axis=0)

        alpha = output_mask*255 # Adjust transparency level
        combined_data = np.stack([x_ray_image[:, :, 0], x_ray_image[:, :, 1], x_ray_image[:, :, 2], alpha], axis=2)

        print(f"combined : {output_mask.shape} ,{combined_data.shape} ,{np.unique(combined_data)} ")

        updated_figure = px.imshow(combined_data,
                        zmin=0, zmax=255,
                        color_continuous_scale='gray',  # Example color scale
                        labels={'color': 'Heatmap Value'})

        return updated_figure
    else:
        return current_figure



# @callback(Output('input_image_id', 'figure'),
#               Input('upload-image', 'contents'),
#               State('upload-image', 'filename'),
#               State('upload-image', 'last_modified'))
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_uploaded_image(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children






###################
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)