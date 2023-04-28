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
import torchinfo

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
        self.conv_block2= self.conv_block(32, 64, stride=2)
        self.conv_block3 = self.conv_block(64, 64)
        self.conv_block4 = self.conv_block(64, 128, stride=2)
        self.conv_block5 = self.conv_block(128, 64,kernel_size=1,padding=0)
        self.conv_block6 = self.conv_block(256, 256,kernel_size=1,padding=0)
        self.conv_block7 = self.conv_block(256, 128,kernel_size=1,padding=0)
        self.conv_block8 = self.conv_block(128, 128,kernel_size=3,padding=1, stride=2)

        self.pooling = nn.MaxPool2d(2, stride=2)
        self.sigmoid=nn.Sigmoid()
        

    def conv_block(self, in_channels, out_channels,kernel_size=3,stride=1, padding=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            
        )
        return block
    
    def block(self,input):
        
        x1=self.sigmoid(input)
        x_out = torch.multiply(input, x1)
        return x_out

    def forward(self, x):
        # Pass the input through the layers
        print(x.shape)
        x = self.conv_block1(x)
        print(x.shape)
        x = self.block(x)
        x = self.conv_block2(x)
        x = self.block(x)
        print(x.shape)
        x = self.conv_block3(x)
        x = self.block(x)        
        x = self.conv_block4(x)
        x_branch1 = self.block(x)   
        print(x_branch1.shape)

        x = self.conv_block5(x_branch1)
        x_branch_1_1 = self.block(x)       
        x = self.conv_block3(x_branch_1_1)
        x = self.block(x) 
        x = self.conv_block3(x)
        x_branch_1_1_1 = self.block(x)
        x = self.conv_block3(x_branch_1_1_1)
        x = self.block(x) 
        x_branch_1_1_1_1 = self.conv_block3(x)

        x = self.conv_block5(x_branch1)
        x_branch2 = self.block(x)    
        x = torch.cat((x_branch2,x_branch_1_1,x_branch_1_1_1,x_branch_1_1_1_1),dim=1)
        print(x_branch2.shape,x_branch_1_1.shape,x_branch_1_1_1.shape,x_branch_1_1_1_1.shape)
        print(x.shape)
        x = self.conv_block6(x)
        x_branch3 = self.block(x)
        print(x_branch3.shape)


        x = self.conv_block7(x_branch3)
        x = self.block(x)
        x = self.conv_block8(x)
        x = self.block(x)
        print(x.shape)

        m = self.pooling(x_branch3)
        print(m.shape)
        m1 = self.conv_block7(m )
        m1 = self.block(m1)
        x1 = torch.cat((x,m1),dim=1)
        print(x1.shape,x.shape,m1.shape)

        return x1

# Create an instance of the model and print a summary of the layers
model = Model()
# summary(model, (3, 160, 320),batch_size = -1)
torchinfo.summary(model, (3, 160, 320), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 1)
torch.save(model, "trial.onnx")