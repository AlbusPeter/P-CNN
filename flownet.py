import scipy.io as sio
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class FlowNet(nn.Module):

    def __init__(self):
        super(FlowNet, self).__init__()
        flownet_model = sio.loadmat('/data6/SRIP18-7/codes/PCNN/pretrained_models/flow_net.mat')
        flownet_modulelist = []
        pre_outchannel = 0

        for layer_number in range(len(flownet_model['layers'][0]) - 1):
            if len(flownet_model['layers'][0][layer_number][0][0]) == 2:
                layer = nn.ReLU()
                flownet_modulelist.append(layer)
            elif len(flownet_model['layers'][0][layer_number][0][0]) == 3:
                layer = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2)
                flownet_modulelist.append(layer)
            elif len(flownet_model['layers'][0][layer_number][0][0]) == 6:
                if flownet_model['layers'][0][layer_number][0][0][1][0] == 'conv':
                    layer_stride = int(flownet_model['layers'][0][layer_number][0][0][3][0][0])
                    layer_padding = int(flownet_model['layers'][0][layer_number][0][0][0][0][0])
                    (kernel_size,kernel_size,in_channels,out_channels) = np.array(flownet_model['layers'][0][layer_number][0][0][4]).shape

                    if pre_outchannel != 0:
                        group_num = pre_outchannel / in_channels
                    else:
                        group_num = 1

                    layer = nn.Conv2d(in_channels,out_channels,kernel_size,stride=layer_stride,padding=layer_padding,groups=int(group_num),bias=True)
                    layer.weight.data = torch.tensor(flownet_model['layers'][0][layer_number][0][0][4]).permute(3,2,0,1)
                    layer.bias.data = torch.tensor(flownet_model['layers'][0][layer_number][0][0][-1]).squeeze()
                    flownet_modulelist.append(layer)
                    pre_outchannel = out_channels
                else:
                    (top, bottom, left, right) = [int(x) for x in flownet_model['layers'][0][layer_number][0][0][2][0]]
                    layer = nn.ZeroPad2d((left,right,top,bottom))
                    flownet_modulelist.append(layer)
                    layer_stride = int(flownet_model['layers'][0][layer_number][0][0][1][0][0])
                    kernel_size = int(flownet_model['layers'][0][layer_number][0][0][-1][0][0])
                    layer = nn.MaxPool2d(kernel_size, stride=layer_stride)
                    flownet_modulelist.append(layer)

        self.mymodules = nn.Sequential(*flownet_modulelist[:-1])
        
    def forward(self, x):
        x = self.mymodules(x)
        # print(x.shape)
        return x.view(x.size(0), -1)

# net = FlowNet()
# print(net)

# input_data = torch.rand(2,3,227,227)

# y = net(input_data)

# print(y.shape)