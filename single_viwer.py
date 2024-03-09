
import onnxruntime
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
import base64
from PIL import Image
import io
from dash_bootstrap_templates import load_figure_template
load_figure_template('SUPERHERO')
"""
TODO :
--------
[x] - view image and mask
[x] - import images
[/] - understand plotly
[/] - view every part of the mask with different color 
[/] - get good layout for portofolio
[ ] - clean code (remove unneeded code - organize and document)
[ ] - get the area of mask segments and view it 

"""

# pathes and configs
image_path = "D:\\chest-x-ray.jpeg"
onnx_model_path = "D:\\Code_store\\CT-Viewer-tk\\unet-2v.onnx"


# getting image
x_ray_image = Image.open(image_path)
x_ray_image = np.array(x_ray_image)
image_shape = x_ray_image.shape


# image in px
x_ray_fig_px = px.imshow(x_ray_image, binary_string=True)
# Update layout to show the full size and enable annotations
x_ray_fig_px.update_layout(
    width=image_shape[0],  # Set width to image width
    height=image_shape[1],  # Set height to image height
    dragmode="drawrect",  # Enable rectangle annotation
    # Set annotation line color to cyan
    newshape=dict(line=dict(color="cyan")),
)

##############################
# Define your Dash app
#############################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

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
                responsive='auto',
                style={'width': '100%', 'height': '100%', 'margin': 'auto', 'display': 'block'},  # Center the image
                config=config  # Enable shape editing
            ),
        ]),
        ###############################################################################
        
        dcc.Dropdown(
            [
                {
                    "label": html.Span(['Lung'], style={'color': 'Black', 'font-size': 20}),
                    "value": "Lung",
                },
                {
                    "label": html.Span(['Chest'], style={'color': 'Black', 'font-size': 20}),
                    "value": "Chest",
                },
                {
                    "label": html.Span(['Bones'], style={'color': 'Black', 'font-size': 20}),
                    "value": "Bones",
                },
            ], value='Montreal'
        ),   
        ###############################################################################
        
        dbc.Button("Show Mask", id="show-mask-button",
                   color="primary", className="mr-1", n_clicks=0),
        ###############################################################################

        dbc.CardFooter(
            [
                html.H6(
                    "import an X-Ray image and press show mask"),
                dbc.Tooltip(
                    "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                    target="x-ray-slider",  # Ensure this matches the ID of the target element
                ),
            ]
        ),
        ###############################################################################
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '90%',
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
                
        
        
    ],
    # Set card width to 100% and height to 100vh (viewport height)
    style={'width': '100%', 'height': 'auto', 'margin': 'auto'}

)


# Define the mask card layout
mask_image_card = dbc.Card(
    [
        dbc.CardHeader("X-Ray mask"),
        dbc.CardBody([
            dcc.Graph(
                id='mask_image_id',  # Set an ID for the graph
                #figure=x_ray_figure,
                responsive='auto',
                config=config,  # Enable shape editing
                style={'width': '100%', 'height': '100%', 'margin': 'auto', 'display': 'block'}
            ),

            dbc.Button("Print Annotations", id="print-button",
                       color="primary", className="mr-1", n_clicks=0),

            # Hidden div to store annotations data
            html.Div(id="annotations-data", style={'display': 'none'}),

            html.Div(id="print-output")
        ##########################################################################
        ]),
        dbc.CardFooter(
            [
                html.H6(
                    "Chest X-Ray segmented image"),
                dbc.Tooltip(
                    "Use the slider to scroll vertically through the image and look for the ground glass occlusions.",
                    target="x-ray-slider",  # Ensure this matches the ID of the target element
                ),
            ]
        ),

        
    ],
    # Set card width to 100% and height to 100vh (viewport height)
    style={'width': '100%', 'height': 'auto', 'margin': 'auto'}
)




####################
# Define app layout
####################

app.layout = html.Div(
    [
        dbc.Row([dbc.Col(image_card, width=5), dbc.Col(mask_image_card, width=5)]),
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


def base64_to_array(base64_string):
    image_data = base64.b64decode(base64_string)
    image_data = Image.open(io.BytesIO(image_data))
    return np.array(image_data)


# prepare input image for model inference
def prepare_model_input(input_image):
    # Load the image and resize it
    #input_image = Image.open(img_path)
    #print(f"input_image 2 {input_image.shape}")
    #input_image=np.expand_dims(input_image, axis=2)

    input_image = np.transpose(input_image, (2, 0, 1))
    print(f"input_image 2 {input_image.shape}")
    # print(input_image.shape)
    # Convert to NumPy array and normalize pixel values
    input_image = input_image.astype(np.float32) / 255.0
    # Adjust array values
    input_image -= 0.5
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    # Convert to single-channel image
    input_image = np.mean(input_image, axis=1, keepdims=True)
    # print(input_image.shape)
    return input_image


def model_inference(onnx_model_path, input_array):
    ort_session = onnxruntime.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: input_array}
    ort_output = ort_session.run(None, ort_inputs)[0]
    # print(ort_output)
    return ort_output


def show_mask_on_image(input_image, onnx_model_path):
    input_image = prepare_model_input(input_image)
    output_mask = model_inference(onnx_model_path, input_image)
    # print(output_mask.shape)
    return output_mask



#############
# Callbacks
##############

# Define callback to print shapes(rectangles , lassos...) when the button is clicked
@app.callback(
    Output("print-output", "children"),
    [Input("mask_image_id", "relayoutData")],
    prevent_initial_call=True,
)
def update_output(relayout_data):
    if "shapes" in relayout_data:

        output_json = json.dumps(relayout_data["shapes"], indent=2)
        if len(relayout_data["shapes"]) > 0:
            print(relayout_data["shapes"][0]['x0'])
        return output_json
    else:
        return no_update


# Callback to update mask overlay when button is clicked
@app.callback(
    Output('mask_image_id', 'figure'),
    [Input('show-mask-button', 'n_clicks')],
    [State('input_image_id', 'figure')]
)
def update_mask_overlay(n_clicks, current_figure):
    if n_clicks > 0:

        print(
            f"current_figure['data'][0] : {current_figure['data'][0].keys()}")
        # TODO : Handling all images inputs (png , jpg , npy....)
        if 'z' in current_figure['data'][0].keys():
            # mostly one channel images in alist format
            input_image = np.array(
                current_figure['data'][0]['z']).reshape(512, 512)
            input_image = np.expand_dims(input_image, axis=2)
            print("upload_image ....", input_image.shape)
            input_image = np.repeat(input_image, 3, axis=2)
            output_mask = show_mask_on_image(input_image, onnx_model_path)

        else:
            print("using 64-encoded image ")
            # print(len(current_figure['data'][0]['source'][22:-1]+"=")/4)
            # to pad encoded string .. making it divisible by 4
            input_image = base64_to_array(
                current_figure['data'][0]['source'][22:-1]+"=")
            print("input_image ....", input_image.shape)
            # Update the figure data with mask overlay
            output_mask = show_mask_on_image(input_image, onnx_model_path)

        output_mask = output_mask.reshape((2, 512, 512))
        # Compute softmax along the appropriate axis
        output_mask = np.exp(output_mask) / np.sum(np.exp(output_mask), axis=0)
        # Find the index of the maximum value along the specified axis
        output_mask = np.argmax(output_mask, axis=0)
        alpha = output_mask*80  # Adjust trasnsparency level
        alpha[alpha == 0] = 255
        #changing only onne channel of pixels 0 for red , 1 for green ,  2 for blue
        # we zeroed 2 channels to make the third stand out
        input_image[alpha==80 ,0]=0 
        input_image[alpha==80 ,2]=0
        combined_data = np.stack(
            [input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2], alpha], axis=2)
        #print(f"combined : {x_ray_image.shape} ,{image_data.shape} , {output_mask.shape}  ")
        updated_figure = px.imshow(combined_data,
                                   zmin=0, zmax=255,
                                   color_continuous_scale='gray',  # Example color scale
                                   labels={'color': 'Heatmap Value'})

        return updated_figure
    else:
        return current_figure


@app.callback(
    Output('input_image_id', 'figure'),
    [Input('upload-image', 'contents')],
    [State('input_image_id', 'figure')]
)
def update_output(list_of_contents, current_figure):
    if list_of_contents is not None:
        _, content_string = list_of_contents[0].split(',')
        image_bytes = base64.b64decode(content_string)

        # Load the image data into a PIL Image object
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image.resize((512, 512)))
        #print(f"image.shape{image.shape}" )
        updated_figure = px.imshow(image,
                                   zmin=0, zmax=255,
                                   color_continuous_scale='gray',  # Example color scale
                                   labels={'color': 'Heatmap Value'})

        return updated_figure
    else:
        return current_figure


###################
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
