"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
"""

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnx2pytorch
import torch.nn as nn
from torchsummary import summary

torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!


# write your code here ...
# Load the ONNX model
model = onnx.load("model/model.onnx")

# Extract the input and output names
input_name = model.graph.input[0].name
output_name = model.graph.output[0].name

# Define the PyTorch model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Initialize the layers
        self.conv_block1 = self.conv_block(3, 32)
        self.sigmoid=nn.Sigmoid()
        

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            
        )
        return block
    
    def block1(self,input):
        x = self.conv_block1(input)
        x1=self.sigmoid(x)
        x_out = torch.multiply(x, x1)
        return x_out

    def forward(self, x):
        # Pass the input through the layers
        x = self.block1(x)

        return x

# Create an instance of the model and print a summary of the layers
model = Model()
summary(model, (3, 160, 320))
torch.save(model, "trial.onnx")