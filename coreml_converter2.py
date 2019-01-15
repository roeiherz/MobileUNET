import re

import onnx
import torch
from PIL import Image
from onnx import onnx_pb
from onnx_coreml import convert
from onnx_tf.backend import prepare

from nets.ImgWrapNet import ImgWrapNet
# %%
from nets.MobileNetV2_unet import MobileNetV2_unet
import numpy as np


def onnx_caffe2():
    """
    Read ONNX model and run it using Caffe2
    """

    model = onnx.load(TMP_ONNX)

    img = np.load('img.p')
    # img = img.transpose(1, 2, 0)
    # img = Image.fromarray(img, 'RGB')
    import onnx_caffe2.backend
    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if torch.cuda.is_available() else 'CPU')
    inp = {model.graph.input[0].name: img}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)


IMG_SIZE = 224
TMP_ONNX = 'tmp/tmp.onnx'
WEIGHT_PATH = 'outputs/UNET_224_weights_100000_days/0-best.pth'
ML_MODEL = re.sub('\.pth$', '.mlmodel', WEIGHT_PATH)
TF_MODEL = re.sub('\.pth$', '.pb', WEIGHT_PATH)


# onnx_caffe2()

# %%
# Convert to ONNX once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = MobileNetV2_unet()
model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu'))
# model = ImgWrapNet(torch.load(WEIGHT_PATH, map_location='cpu'))
model.to(device)
# model.eval()

torch.onnx.export(model,
                  torch.randn(1, 3, IMG_SIZE, IMG_SIZE),
                  TMP_ONNX)

# %%
# Print out ONNX model to confirm the number of output layer
onnx_model = onnx.load(TMP_ONNX)
print(onnx_model)

# %%
# Convert ONNX to CoreML model
model_file = open(TMP_ONNX, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
# 595 is the identifier of output.
coreml_model = convert(model_proto,
                       image_input_names=['0'],
                       image_output_names=['590'])
coreml_model.save(ML_MODEL)

# %%
# tf_rep = prepare(onnx_model)  # prepare tf representation
# tf_rep.export_graph(TF_MODEL)  # export the model
