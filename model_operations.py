
import os
from onnx import load as load_onnx
from onnx.checker import check_model
import torch
from numpy.testing import assert_allclose as numpy_assert_allclose
import datetime

def load_weights(model, weights_path: str, device) -> None:
    """Load the model weights from disk"""

    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint,strict=False)
    model.to(device)
    # set the model to evaluation mode
    model.train(False)
    model.eval()
    return model


def export(
    model,
    path: str,
    dummy_input,
) -> None:
    """Export the model to ONNX format"""

    # send the dummy input to cpu
    dummy_input = dummy_input.to("cpu")

    input_names = ["input"]
    output_names = ["output"]

    """
    A dynamic axis is one whose size is not known at export time ,
    in the below case  the model can accept different and outputs batch sizes
    
    """
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        # the ONNX file will contain the model parameters as well as the model graph(so that you can retrain or fine tune model later) if false (model graph only will be exported)
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,  # Constant folding is a technique that can be used to optimize the ONNX graph by replacing constant expressions with their numerical values. This can make the ONNX graph
        opset_version=16,
    )
    # Checks
    model_onnx = load_onnx(path)  # load onnx model
    check_model(model_onnx)  # check onnx model
    print("Model exported to ONNX format.")


def onnx_export(model, weights_path: str, test_loader, img_size, model_type, num_epochs, lr) -> None:
    """Test and export the model on the test set"""

    # load the best model checkpoint from disk
    model = load_weights(weights_path)

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    onnx_path = os.path.join(
        weights_path,
        "%s-%d-%.4f-%s.onnx"
        % (model_type, num_epochs, lr, current_datetime),
    )

    dummy_input = next(iter(test_loader))["img"].view(
        -1, 2, img_size, img_size
    )

    export(model, onnx_path, dummy_input)

#overloaded
def onnx_export(model, weights_path, dummy_input,onnx_path) -> None:
    """Test and export the model on the test set"""

    # load the best model checkpoint from disk
    model = load_weights(model,weights_path,"cpu")

    export(model, onnx_path, dummy_input)


def load_models_and_compare(
    pt_model_path, onnx_model_path, input_data, model, img_ch, output_ch
):
    with torch.no_grad():

        pt_model = model
        checkpoint = torch.load(pt_model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint,strict=False)
        model.to("cpu")
        # delete checkpoint
        del checkpoint

        # Forward pass with PyTorch model
        pt_output = pt_model(input_data)

        # Forward pass with ONNX model
        import onnxruntime

        ort_session = onnxruntime.InferenceSession(
            onnx_model_path, providers=["CPUExecutionProvider"]
        )
        ort_inputs = {ort_session.get_inputs(
        )[0].name: input_data.cpu().numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        ###############################################
        # check that torch and onnx results are equal #
        ###############################################
        torch.testing.assert_close(
            pt_output, torch.from_numpy(ort_output), rtol=1e-03, atol=1e-05
        )
        numpy_assert_allclose(
            pt_output.cpu().numpy(), ort_output, rtol=1e-03, atol=1e-05
        )

        print(
            "The outputs of PyTorch and ONNX models are equal. Congratulations along way to go!"
        )


############################################
#adding a tail function to the model after making it
############################################

import torch
from torch.onnx import register_custom_op_symbolic
import onnxruntime

def my_custom_function(x):
    """
    A custom function that takes a tensor as input and returns the square of the tensor.
    """
    return x ** 2

def add_tail_to_model(my_custom_function,model):
    register_custom_op_symbolic("my_custom_function", my_custom_function)

    model = torch.nn.Linear(10, 10)
    x = torch.randn(10)

    torch.onnx.export(model, x, "model.onnx", verbose=True, custom_opset_version=13, opset_version=13, input_names=["x"], output_names=["y"])

    # Run the model using the ONNX Runtime
    sess = onnxruntime.InferenceSession("model.onnx")

    input_data = x.cpu().numpy()
    outputs = sess.run(None, {"x": input_data})

    # Check the output
    print(outputs["y"])




if __name__=="__main__":
    from models import UNet
    model_structure = UNet(
        in_channels=1,
        out_channels=2, 
        batch_norm=True, 
        upscale_mode="bilinear"
    )
    dummy_input = torch.rand((4, 1, 512, 512))
    
    pt_model_path="D:\\Code_store\\CT-Viewer-tk\\unet-2v.pt"
    onnx_model_path="unet-2v.onnx"
    #onnx_export(model_structure, pt_model_path, dummy_input,onnx_model_path)

    # testing and compare
    import torchvision
    from PIL import Image

    input_image = Image.open("D:\\chest-x-ray.jpeg")
    input_image = torchvision.transforms.functional.resize(input_image, (512, 512))
    input_image = torchvision.transforms.functional.to_tensor(input_image) - 0.5 
    print(input_image.shape)
    input_image = torch.stack([input_image])
    input_image = torch.mean(input_image, dim=1, keepdim=True)
    input_image = input_image.to("cpu")
    print(input_image.shape)
    
    load_models_and_compare(
    pt_model_path, onnx_model_path, input_data=input_image, model=model_structure, img_ch=1, output_ch=2
)