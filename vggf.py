import scipy.io as sio
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# vgg_model = sio.loadmat('/data6/SRIP18-7/codes/PCNN/pretrained_models/imagenet-vgg-f.mat')

# print(torch.tensor(vgg_model['layers'][0][0][0][0][3]).shape) #first two numbers locate the layer, fifth number locate the property
# print(vgg_model['layers'][0][0][0][0][-1][0] == 'adhjada')
# (top, bottom, left, right) = np.int_(vgg_model['layers'][0][3][0][0][2][0])
# print(top)

# print(len(vgg_model['layers'][0][2][0][0]))

class VGGF(nn.Module):

    def __init__(self):
        super(VGGF, self).__init__()
        vgg_model = sio.loadmat('/data6/SRIP18-7/codes/PCNN/pretrained_models/imagenet-vgg-f.mat')
        vgg_modulelist = []

        for layer_number in range(len(vgg_model['layers'][0]) - 1):
            if len(vgg_model['layers'][0][layer_number][0][0]) == 2:
                layer = nn.ReLU()
                vgg_modulelist.append(layer)
            elif len(vgg_model['layers'][0][layer_number][0][0]) == 3:
                layer = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2)
                vgg_modulelist.append(layer)
            elif len(vgg_model['layers'][0][layer_number][0][0]) == 6:
                if vgg_model['layers'][0][layer_number][0][0][-1][0] == 'conv':
                    layer_stride = int(vgg_model['layers'][0][layer_number][0][0][1][0][0])
                    layer_padding = int(vgg_model['layers'][0][layer_number][0][0][2][0][0])
                    (kernel_size,kernel_size,in_channels,out_channels) = np.array(vgg_model['layers'][0][layer_number][0][0][4]).shape
                    layer = nn.Conv2d(in_channels,out_channels,kernel_size,stride=layer_stride,padding=layer_padding,bias=True)
                    layer.weight.data = torch.tensor(vgg_model['layers'][0][layer_number][0][0][4]).permute(3,2,0,1)
                    layer.bias.data = torch.tensor(vgg_model['layers'][0][layer_number][0][0][3]).squeeze()
                    vgg_modulelist.append(layer)
                else:
                    (top, bottom, left, right) = [int(x) for x in vgg_model['layers'][0][layer_number][0][0][2][0]]
                    layer = nn.ZeroPad2d((left,right,top,bottom))
                    vgg_modulelist.append(layer)
                    layer_stride = int(vgg_model['layers'][0][layer_number][0][0][1][0][0])
                    kernel_size = int(vgg_model['layers'][0][layer_number][0][0][-1][0][0])
                    layer = nn.MaxPool2d(kernel_size, stride=layer_stride)
                    vgg_modulelist.append(layer)

        self.mymodules = nn.Sequential(*vgg_modulelist[:-1])
        
    def forward(self, x):
        x = self.mymodules(x)
        # print(x.shape)
        return x.view(x.size(0), -1)

# net = VGGF()
# print(net)

# input_data = torch.rand(2,3,224,224)

# y = net(input_data)

# print(y.shape)

class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average
    def forward(self, output, y):#output: batchsize*n_class
        #print(output.requires_grad)
        #print(y.requires_grad)
        output_y=output[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()].view(-1,1)#view for transpose
        #margin - output[y] + output[i]
        loss=output-output_y+self.margin#contains i=y
        #remove i=y items
        loss[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()]=0
        #max(0,_)
        loss[loss<0]=0
        #^p
        if(self.p!=1):
            loss=torch.pow(loss,self.p)
        #add weight
        if(self.weight is not None):
            loss=loss*self.weight
        #sum up
        loss=torch.sum(loss)
        if(self.size_average):
            loss/=output.size()[0]#output.size()[0]
        return loss