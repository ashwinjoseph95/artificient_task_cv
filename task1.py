"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)
print("Original shape",y.shape)

# now write your custom layer
class CustomGroupedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True): #setting same parameter names as original convolution layer from pytorch
        super(CustomGroupedConv2D, self).__init__()

        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        
        # split the input and output channels into equal groups
        in_groups = in_channels // groups
        out_groups = out_channels // groups

        # create a stack of 1-grouped convolution layers with shared weights and bias
        self.conv_stack = nn.ModuleList([nn.Conv2d(in_groups, out_groups, kernel_size, stride=stride, 
                                                    padding=padding, dilation=dilation, groups=1, bias=bias) 
                                          for _ in range(groups)])
        self.bias = None
        
    def forward(self, x):
        # split the input tensor into groups
        x_groups = torch.split(x, int(x.size(1) / len(self.conv_stack)), dim=1)
        
        # apply each 1-grouped convolution layer to its corresponding input group
        y_groups = [conv(x_group) for conv, x_group in zip(self.conv_stack, x_groups)]
        
        # concatenate the output groups and add bias
        y = torch.cat(y_groups, dim=1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1).expand_as(y)
        return y


# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
groups=16
out_filter_shape=128
custom_layer = CustomGroupedConv2D(64, out_filter_shape, 3, stride=1, padding=1, groups=groups, bias=True)

# copy weights and bias from the original layer to the custom layer
for i, conv in enumerate(custom_layer.conv_stack):
    conv.weight.data = w_torch[i*(out_filter_shape//groups):(i+1)*(out_filter_shape//groups), :, :, :]
    conv.bias.data = b_torch[i*(out_filter_shape//groups):(i+1)*(out_filter_shape//groups)]

y_custom = custom_layer(x)
print("Custom shape",y_custom.shape)

# check if the outputs are equal
if (y_custom.shape)==(y.shape):
    print("SHapes are matching!")

print(torch.eq(y, y_custom))









        
