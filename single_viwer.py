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


#getting image
image_path = "D:\\chest-x-ray.jpeg"
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
    sizex=10,
    sizey=10,)

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


# Define the image card layout
image_card = dbc.Card(
    [
        dbc.CardHeader("X-Ray"),
        dbc.CardBody([
            dcc.Graph(
                id='x-ray-graph',  # Set an ID for the graph
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


mask_image_card = dbc.Card(
    [
        dbc.CardHeader("X-Ray"),
        dbc.CardBody([
            dcc.Graph(
                id='x-ray-mask',  # Set an ID for the graph
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


###########################
#Callbacks
# Define callback to print something when the button is clicked
@app.callback(
    Output("print-output", "children"),
    [Input("x-ray-graph", "relayoutData")],
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



onnx_model_path="D:\\Code_store\\CT-Viewer-tk\\unet-2v.onnx"
# Callback to update mask overlay when button is clicked
@app.callback(
    Output('x-ray-mask', 'figure'),
    [Input('show-mask-button', 'n_clicks')],
    [State('x-ray-mask', 'figure')]
    
)
def update_mask_overlay(n_clicks,current_figure):
    if n_clicks > 0:
        # Update the figure data with mask overlay
        output_mask = show_mask_on_image(image_path,onnx_model_path)
        output_mask = np.squeeze(output_mask,axis=0) # removing first dim (batch dim)
        output_mask = np.transpose(output_mask , (1,2,0)) #(512,512,2)--->(2,512,512)
        
        output_mask = np.mean(output_mask, axis=0, keepdims=True)
        output_mask = np.squeeze(output_mask,axis=0) # removing first dim (batch dim)
        
        
        print(f"generating mask 1...{output_mask.shape}.....")
        
        heatmap_data = go.Heatmap(z=output_mask, opacity=0.5, colorscale='Viridis')
        layout = go.Layout(title='X-ray with Mask Overlay')
        updated_figure = go.Figure(data=[heatmap_data], layout=layout)
        
        print("generating mask 2........")
        
        #relayout_data[""]
        # x_ray_fig_px.update_layout(images=[dict(source=output_mask,binary_string=True)])
        # print("generating mask 3........")
        
        # Add your mask overlay logic here
        # For example, if you have a pre-computed mask array, you can add it as a heatmap to the figure
        # updated_fig.add_heatmap(z=mask_array, opacity=0.5, colorscale='Viridis')
        return updated_figure
    else:
        return current_figure




####################
# Define app layout
app.layout = html.Div([
    dbc.Row([
        dbc.Col(image_card, width=5),  # Place the first image card in a column of width 6
        dbc.Col(mask_image_card, width=5)  # Place the second image card in a column of width 6
    ])
])




###################
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)