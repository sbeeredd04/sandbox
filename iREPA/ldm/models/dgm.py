'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import legendre
import numpy as np
from torch.autograd import Variable
from scipy.stats import ortho_group
import math

# from util.midas_utils import dpt_transform, gm_transform_in, gm_transform_out

def create_bases():
    k=210
    a = np.array(range(32))/31.
    ab = np.meshgrid(a, a)
    xy = np.zeros((k, 32, 32))
    c = 0
    coeff = np.zeros((1, k))
    for l in range(20):
        for m in range(l+1):
            coeffx = legendre(l-m)
            coeffy = legendre(m)
            for i in range(32):
                for j in range(32):
                    #xy[c, i, j] = pow(ab[0][i, j], l)*pow(ab[1][i, j], m)
                    x = (2*i - 32 + 1)/(32-1)
                    px = coeffx(x)
                    y = (2*j - 32 + 1)/(32-1)
                    py = coeffy(y)
                    xy[c, i, j] = px*py
            coeff[0, c] = (2*(l-m)+1)*(2*m+1)/4
            c += 1
    # print(c)
    # for l in range(1, 20):
    #     for m in range(l+1):
    #         for i in range(32):
    #             for j in range(32):
    #                 xy[c, i, j] = pow(ab[0][i, j], m)*pow(ab[1][i, j], l)
    #         c += 1 
            #print(c)
    return xy, coeff  

class Blur(nn.Module):


    def __init__(self, planes):
        super(Blur, self).__init__()
        kernel_size = 3
        sigma = 1
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        # # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(planes, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=planes, out_channels=planes,
                                    kernel_size=kernel_size, groups=planes, bias=False, padding=1)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False
    def forward(self, x):
        return self.gaussian_filter(x)

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.InstanceNorm2d(num_features, affine=False)
    self.embed = nn.Linear(num_classes, num_features * 2, bias=False)
    #self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    #self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

  def forward(self, x, y):
    out = x#self.bn(x)
    gamma, beta = self.embed(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, k=3, p=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=k,padding=p,  stride=stride, bias=False)
        self.b = bn
        self.df = planes
        #if bn:
        self.bn1 = nn.BatchNorm2d(planes)
        #self.bn1 = nn.LayerNorm((planes, 32, 32))
        # else:
        #     self.bn1 = ConditionalBatchNorm2d(planes, 512)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=k,padding=p,
                              stride=1, bias=False)        
        #if bn:
        self.bn2 = nn.BatchNorm2d(planes)
        #self.bn2 = nn.LayerNorm((planes, 32, 32))
        # else:    
        #     self.bn2 = ConditionalBatchNorm2d(planes, 512)
        #print(k, p)
        self.shortcut = nn.Sequential()
        # kernel_size = 3
        # sigma = 1

        # # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        # x_cord = torch.arange(kernel_size)
        # x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        # y_grid = x_grid.t()
        # xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        # mean = (kernel_size - 1)/2.
        # variance = sigma**2.

        # # Calculate the 2-dimensional gaussian kernel which is
        # # the product of two gaussian distributions for two different
        # # variables (in this case called x and y)
        # gaussian_kernel = (1./(2.*math.pi*variance)) *\
        #                 torch.exp(
        #                     -torch.sum((xy_grid - mean)**2., dim=-1) /\
        #                     (2*variance)
        #                 )
        # # Make sure sum of values in gaussian kernel equals 1.
        # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # # Reshape to 2d depthwise convolutional weight
        # gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        # gaussian_kernel = gaussian_kernel.repeat(planes, 1, 1, 1)

        # self.gaussian_filter = nn.Conv2d(in_channels=planes, out_channels=planes,
        #                             kernel_size=kernel_size, groups=planes, bias=False, padding=1)

        # self.gaussian_filter.weight.data = gaussian_kernel
        # self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        #if self.b:
        #x = inp
        #print(self.conv1.weight.shape)
        #out = self.conv1(x)
        #out = self.conv2(out)
        #out = F.conv2d(x, F.normalize(self.conv1.weight, dim=0), self.conv1.bias,padding=2, groups=self.df)
        out = F.relu(self.bn1(self.conv1(x)))
        
        #out = self.gaussian_filter(out)
        #out = self.w1*out/((out.mean(dim=(2,3), keepdim=True)+1e-12))
        out = self.bn2(self.conv2(out))            
        #out = self.gaussian_filter(out)
        #print(out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        
        #out = self.w2*out/((out.mean(dim=(2,3), keepdim=True)+1e-12))
        return out


class BasicBlockG(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, k=3, p=1):
        super(BasicBlockG, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=k, padding=p, stride=stride, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=k,padding=p, 
                               stride=1, bias=False)        
        #if bn:
        self.bn2 = nn.BatchNorm2d(planes)
        #self.w = Variable(torch.ones(1, planes, 1, 1), requires_grad=True).cuda()


        self.shortcut = nn.Sequential()
       
    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        #out = self.gaussian_filter(out)
        out = self.bn2(self.conv2(out))         
        
        #out += self.shortcut(x)
        out = F.relu(out)
        #out = self.gaussian_filter(out)
        #s  = out.sum(dim=(2,3), keepdim=True)
        #out = self.w*out/(s+1e-6)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def moving_mean(x, training):
    if training:
        #n = torch.zeros(x.shape, dtype=torch.float32)
        w = torch.cuda.FloatTensor(1, x.shape[1], 1, 1).normal_(1.0, 0.2)
        b = torch.cuda.FloatTensor(1, x.shape[1], 1, 1).normal_(0.0, 0.2)
        return x*w+b
        #return (0.9*x+0.1*y)
    else:
        #print('test')
        return x
def dropout_without_scale(x, p, training):
    if training:
        m = torch.ones_like(x)*p
        m = torch.bernoulli(m)
        return x*m
    else:
        return x
def brelu(x):
   
    x = F.leaky_relu(x)
    return 1. - F.leaky_relu((1.0 -x))

class BasicBlockGM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, n=1,  k=3, p=1, hw=32, box=None):
        super(BasicBlockGM, self).__init__()
        layers = []
        pl = in_planes
        for i in range(n):
            layers.append(BasicBlock(pl, planes, stride, bn, k, p)) 
            pl= planes      
        self.layer = nn.Sequential(*layers)

        # self.conv5 = nn.Conv2d(
        #     planes, planes, kernel_size=1, padding=0, stride=1, bias=False)
                      
        # self.bn5 = nn.BatchNorm2d(planes)

        self.conv1g = nn.Conv2d(
            in_planes, planes, kernel_size=1, padding=0, stride=1, bias=False)
                      
        self.bn1g = nn.BatchNorm2d(planes)
 
        self.conv2g = nn.Conv2d(planes, planes, kernel_size=1,padding=0, 
                                stride=1, bias=False)        
        self.bn2g = nn.BatchNorm2d(planes)

        self.fc1_1 = nn.Linear(in_planes, 2*planes)
        #self.fc2_1 = nn.Linear(2*planes, 2*planes)
        self.df = planes
        #self.fc1_4 = nn.Linear(2*planes, 2*2, bias=True)
        self.fc1_2 = nn.Linear(2*planes, 2*self.df, bias=True)
        self.fc1_3 = nn.Linear(2*planes, self.df, bias=True)
        self.fc1_4 = nn.Linear(2*planes, 2*2, bias=True)
        self.fc1_5 = nn.Linear(2*planes, 2, bias=True)
        self.ln11 = nn.LayerNorm(2*self.df)
        self.ln12 = nn.BatchNorm2d(self.df)
        self.ln13 = nn.BatchNorm2d(self.df)
        
        self.hw = hw
        self.box = box
        #self.sf = nn.Softmax2d()
        self.do1 = nn.Dropout(p=0.1)
        self.do2 = nn.Dropout(p=0.1)
        self.pw = nn.Parameter(Variable(torch.zeros(1, self.df, 1, 1), requires_grad=True))
        self.pw1 = nn.Parameter(Variable(torch.ones(1, self.df, 1, 1), requires_grad=True))
        #self.conv12 = nn.Conv2d(self.df, self.df, kernel_size=1, stride=1)
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:

       
    def forward(self, x, xy1, grid, gridt, mask, ):
        #print(x.shape)
        
        out = self.layer(x +self.pw*grid)#+ x*F.relu(self.ln12(grid)))))

        # x = out
        # out = F.relu(self.bn5(self.conv5(out)))
        # out = F.relu(x+self.bn6(self.conv6(out)))
        #print(xy1.shape)
        xy1_ = F.relu(self.ln11(self.fc1_1(xy1)))
        #print(xy1_.shape)
        #xy1_ = F.relu(self.ln12(self.fc2_1(xy1_)))
        xy_b =  self.fc1_3(xy1_).view(-1, self.df, 1,1)
        xy_ = self.fc1_2(xy1_).view(-1,self.df , 2)
        #xy_1 = self.fc1_4(xy1_).view(-1,2 , 3)
        xy_b1 =  self.fc1_5(xy1_).view(-1, 2, 1,1)
        xy_1 = self.fc1_4(xy1_).view(-1,2 , 2)

        gridt = torch.matmul(xy_1, gridt.view(-1, 2, self.hw*self.hw)).view(-1, 2, self.hw, self.hw) + xy_b1
      
        #gridt = gridt.repeat(xy_1.shape[0], 1, 1, 1)
        # g1 = F.affine_grid(xy_1, gridt.size())
        # gridt = F.grid_sample(gridt, g1).view(-1,2,self.hw, self.hw)

        g1 = torch.matmul(xy_, gridt.view(-1, 2, self.hw*self.hw)).view(-1, self.df, self.hw, self.hw) + xy_b
      
        #g2 = torch.matmul(xy_, 1.0 - gridt.view(1, 2, -1)).view(-1, 2, self.hw, self.hw) - xy_b
        # g3 = torch.matmul(xy_, torch.rot90(gridt, k=2, dims=(2,3)).reshape(1, 2, -1)).view(-1, 2, self.hw, self.hw) + xy_b
        # g4 = torch.matmul(xy_, torch.rot90(gridt, k=3, dims=(2,3)).reshape(1, 2, -1)).view(-1, 2, self.hw, self.hw) + xy_b
        #g1 = g1.clamp(min=0.0, max=1.0)
        # g1[g1<0.0] = 1e-4
        # g1[g1>1.0] = 1e-4

        # xy_ = torch.split(xy_.view(-1, 4), 1, dim=1)
        # xy_b = torch.split(xy_b.view(-1, 2), 1, dim=1)
        # xy_ = torch.cat((xy_[0], xy_[1], xy_b[0], xy_[2], xy_[3], xy_b[1])).view(-1, 2 , 3)
        # # xyb = torch.cat((xy_[2], xy_[5]))

        # # box = torch.matmul(xy_, self.box.view(1, 2, 2)) + xy_b.view(-1,2, 1)
        # # box = (box.view(-1, 4)+1.0)*16.0
        # # box = box.clamp(min=0.0, max=32)
        # #box = torch.sum(g1, dim=1, keepdim=True)
        # mask = mask.repeat(xy_.shape[0], 1, 1, 1)
        # m = F.affine_grid(xy_, mask.size())
        # m = F.grid_sample(mask, m).view(-1,1,self.hw, self.hw)
        #print(m.shape)
        outg1 = F.relu(self.bn1g(self.conv1g(g1 +self.pw1*grid)))#*(self.hw*self.hw)/(torch.sum(m.view(out.shape[0], 1, -1), dim=-1).view(-1,1,1, 1)+ 1e-6))
        outg1 = F.relu(self.bn2g(self.conv2g(outg1)))
        # outg3 = self.conv3g(g3)
        # outg4 = self.conv4g(g4)

        grid =    outg1*out  #torch.sigmoid(self.ln13(out)) #+ outg2*out 
        xy1 =  torch.flatten(F.avg_pool2d(grid, x.shape[2]), 1)#/torch.pow(torch.flatten(F.avg_pool2d(out, x.shape[2]), 1) + 1e-4, self.pw)#*(self.hw*self.hw)/(torch.sum(m.view(out.shape[0], 1, -1), dim=-1)+ 1e-6)
       
        #xy1 = torch.sum(grid.view(out.shape[0], out.shape[1], -1), dim=-1)/

        return out, grid, xy1, outg1, gridt


class MyResNet1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, device=None):
        super(MyResNet1, self).__init__()
        

        self.device = device
        self.hw8 = 8
        self.hw = 32
        self.df=256
        h = (self.hw-1)
        a = (torch.Tensor(range(self.hw)))/(h)
        g = torch.meshgrid(a, a)
        self.gridt = nn.Parameter(torch.cat((g[0].view(1, 1, self.hw,self.hw), g[1].view(1, 1, self.hw,self.hw),
                        #1. - g[0].view(1, 1, self.hw,self.hw), 1. - g[1].view(1, 1, self.hw,self.hw),
                        ),
                        #torch.rot90(g[0].view(1, 1, self.hw,self.hw), k=1, dims=(2,3)),torch.rot90(g[1].view(1, 1, self.hw,self.hw), k=1, dims=(2,3)),
                        #torch.rot90(g[0].view(1, 1, self.hw,self.hw), k=2, dims=(2,3)),torch.rot90(g[1].view(1, 1, self.hw,self.hw), k=2, dims=(2,3)),
                        #torch.rot90(g[0].view(1, 1, self.hw,self.hw), k=3, dims=(2,3)),torch.rot90(g[1].view(1, 1, self.hw,self.hw), k=3, dims=(2,3))),
                        dim=1), requires_grad=False)
        self.grid = nn.Parameter(self.gridt.view(-1, 2, self.hw*self.hw), requires_grad=False)
        self.mask = nn.Parameter(torch.ones(1,1,self.hw,self.hw), requires_grad=False)
        self.xy_ = nn.Parameter(Variable(torch.rand(1, self.df, 2), requires_grad=True))
        self.xy_b = nn.Parameter(Variable(torch.rand(1,self.df, 1,1), requires_grad=True))
        self.box = nn.Parameter(torch.Tensor([[-1.0, -1.0], [1.0, 1.0]]).float(), requires_grad=False)

        # a = torch.Tensor(range(self.hw8))/(self.hw8-1.)
        # g = torch.meshgrid(a, a)
        # self.gridt8 = nn.Parameter(torch.cat((g[0].view(1, 1, self.hw8,self.hw8), g[1].view(1, 1, self.hw8,self.hw8)), dim=1))
        # self.grid8 = nn.Parameter(self.gridt8.view(-1, 2, self.hw8*self.hw8))       
        # self.xy_8 = nn.Parameter(Variable(torch.rand(1, self.df, 2), requires_grad=True))
        # self.xy_b8 = nn.Parameter(Variable(torch.rand(1,self.df, 1,1), requires_grad=True))


        self.in_planes = self.df
        #self.layerx2 = self._make_layer(block, self.df, 4, stride=1)
        self.in_planes = self.df
        self.layer01 = self._make_layer(BasicBlockG, self.df, 1, stride=1, k=1, p=0)
        #self.layer018 = self._make_layer(BasicBlockG, self.df, 2, stride=1, k=1, p=0)
        #self.in_planes = 16
        self.layer02 = self._make_layer(block, self.df, 4, stride=1, k=3, p=1)
        self.in_planes = self.df
        #self.conv01 = nn.Conv2d(2, self.df, kernel_size=1, stride=1)
        #self.convx2 = nn.Conv2d(self.df, self.df, kernel_size=4, stride=4)
        self.conv02 = nn.Conv2d(3, self.df, kernel_size=8, stride=8, padding=0)
        self.in02 = nn.BatchNorm2d(self.df)
        #self.in02 = nn.InstanceNorm2d(self.df)

        #self.convx2 = nn.Conv2d(self.df, self.df, kernel_size=3, stride=1, padding=1)
        self.resgm1 = BasicBlockGM(self.df, self.df, n=4, k=3, p=1, hw=self.hw, box = self.box)
        self.resgm2 = BasicBlockGM(self.df, self.df, n=4, k=3, p=1, hw=self.hw, box = self.box)
        self.resgm3 = BasicBlockGM(self.df, self.df, n=4, k=3, p=1, hw=self.hw, box = self.box)
        self.resgm4 = BasicBlockGM(self.df, self.df, n=2, k=3, p=1, hw=self.hw, box = self.box)
        #self.resgm5 = BasicBlockGM(self.df, self.df, n=1,k=3, p=1, hw=self.hw, box = self.box)
        # self.resgm6 = BasicBlockGM(self.df, self.df, n=2,k=1, p=0, hw=self.hw, box = self.box)
        # self.resgm7 = BasicBlockGM(self.df, self.df, n=2,k=1, p=0, hw=self.hw, box = self.box)
        # self.resgm8 = BasicBlockGM(self.df, self.df, n=4,k=1, p=0, hw=self.hw, box = self.box)
        # self.resgm9 = BasicBlockGM(self.df, self.df, k=1, p=0, hw=self.hw, box = self.box)
        # self.resgm10 = BasicBlockGM(self.df, self.df, k=1, p=0, hw=self.hw, box = self.box)
        # self.fc1_1 = nn.Linear(self.df,self.df)
        #self.fc1_2 = nn.Linear(self.df, 3*2)
        # self.fc1_3 = nn.Linear(self.df, self.df)
        # #self.fc1_4 = nn.Linear(256, 256)
        # self.ln11 = nn.LayerNorm(self.df)
        # self.ln12 = nn.LayerNorm(self.df)
        # self.layer11 = self._make_layer(BasicBlockG, self.df, 1, stride=1, k=1, p=0)
        # self.layer12 = self._make_layer(block, self.df,1, stride=1, k=1, p=0)
        # #self.in12 = nn.InstanceNorm2d(self.df)

        self.sf = nn.Softmax2d()
        self.linear = nn.Linear(self.df, num_classes)
        #self.linear1 = nn.Linear(self.df, num_classes)

        dropout = 0.1
        self.do1 = nn.Dropout(p=dropout)

        self.conv11 = nn.Conv2d(2, self.df, kernel_size=1, stride=1, bias=False)
        #self.conv12 = nn.Conv2d(self.df, 3, kernel_size=1, stride=1, bias=True)
        # self.conv13 = nn.Conv2d(2, self.df, kernel_size=1, stride=1, bias=False)
        # self.conv14 = nn.Conv2d(2, self.df, kernel_size=1, stride=1, bias=False)
    
    def _make_layer(self, block, planes, num_blocks, stride, k=3, p=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, k=k, p=p))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_moments=True):
        #x =  gm_transform_in(x)
        #x = self.do(x)
        #x = x.clamp(min=0.0, max=1.0)
        im = x#torch.split(x, 1, dim=1)
         # cl = []
        size = (x.shape[2], x.shape[3])
        gridt = self.gridt#.repeat(x.size(0), 1, 1, 1)
        # xy1_ = self.gm
        # xy_b = self.fc0_3(xy1_).view(-1,self.df, 1,1)
        
        # xy_ = self.fc0_2(xy1_).view(-1, self.df, 2)

        #grid = torch.matmul(self.xy_, self.grid).view(-1, self.df, self.hw, self.hw) + self.xy_b

        #grid = F.relu(self.conv01(grid))
        grid11 = self.layer01(self.conv11(gridt))
        grid1 = grid11
        #r = F.conv2d(x, grid11.view(self.df, 3, self.hw, self.hw), stride=self.hw)
        #grid12 = self.conv11(1.0 - self.gridt)
        # grid13 = self.conv13(torch.rot90(self.gridt, k=2, dims=(2,3)))
        # grid14 = self.conv14(torch.rot90(self.gridt, k=3, dims=(2,3)))
        #f_g =  torch.flatten(F.avg_pool2d(grid1, grid1.shape[2]), 1)
        #imgr = grid.view(256,1, 32, 32)
        #x = self.layerx2(self.convx2(x))
        # grid8 = torch.matmul(self.xy_8, self.grid8).view(1, self.df, self.hw8, self.hw8) + self.xy_b8
        # grid8 = grid8.view(self.df, 1, self.hw8, self.hw8)
        #print(x.shape)
        x = self.conv02(im)#F.conv2d(F.relu(self.conv02(x)), grid8, stride=8, groups=self.df)
        #print(x.shape)
        x = self.layer02(x)#
        #x = self.convx2(x)
        
        
        
        #y = torch.sigmoid(self.conv12(x))
        grid = grid11*x# + grid12*x
        xy1 = torch.flatten(F.avg_pool2d(grid, x.shape[2]), 1)
        #print(x.shape, grid11.shape, xy1.shape)
        #xy_ = self.fc1_2(xy1).view(-1, 2, 3)
        
        # g1 = F.affine_grid(xy_, im.size())
        # im = F.grid_sample(im, g1)
        #print(x.shape)
        x, grid, xy1, box, gridt = self.resgm1(x,xy1,grid, gridt, grid11)
        x, grid, xy1, box, gridt = self.resgm2(x,xy1,grid, gridt, box)
        x, grid, xy1, box, gridt = self.resgm3(x,xy1,grid, gridt, box)
        feat = xy1
        grid1 = grid
        x, grid, xy1, box, gridt = self.resgm4(x,xy1,grid, gridt, box)
        #x, grid, xy1, box, gridt = self.resgm5(x,xy1,grid, gridt, box)
        # x, grid, xy1, box, gridt = self.resgm6(x,xy1,grid, gridt, box)
        # x, grid, xy1, box, gridt = self.resgm7(x,xy1,grid, gridt, box)
        # x, grid, xy1, box, gridt = self.resgm8(x,xy1,grid, gridt, box)
        # x, grid, xy1, box = self.resgm10(x,xy1,grid, self.gridt, box)
        #imgr1 = torch.sum(grid*(xy1).view(-1, xy1.shape[1], 1, 1), dim=1, keepdim=True)

        # uncomment from 537 to 543
        imgr = torch.sum(grid1*(feat).view(-1, xy1.shape[1], 1, 1), dim=1, keepdim=True)

        imgr = imgr.view(imgr.size(0), -1)
        imgr = imgr - imgr.min(1, keepdim=True)[0]
        imgr = imgr/imgr.max(1, keepdim=True)[0]
        imgr = (imgr.view(-1, 1, self.hw, self.hw))
        imgr = nn.Upsample(size, mode='bilinear', align_corners=True)(imgr)
        # imgr1 = imgr
        # imgr = box
        # imgr = imgr.view(imgr.size(0), -1)
        # imgr = imgr - imgr.min(1, keepdim=True)[0]
        # imgr = imgr/imgr.max(1, keepdim=True)[0]
        # imgr = (imgr.view(-1, 1, self.hw, self.hw))       
        #out = F.relu(self.ln50(out+self.fc5_0(xy1_))) #+ F.relu(self.ln51(self.fc5_1(xy1_)))
        #xy1 = F.relu(self.ln10(self.linear1(xy1)))
        #xy1 = y4#torch.cat((y4), dim=-1)
        #print(xy1.shape)
        
        cl=self.linear(xy1)
        #box = self.linear1(box)
        #cl = torch.square(xy1.view(-1, 256, 1)- self.wg).sum(dim=1)
        # return gm_transform_out(imgr)
        # return cl, imgr, grid
        if return_moments:
            return cl, (grid, xy1), imgr
        
        return cl, imgr


def ResNet18(device=None, num_classes=1000):
    return MyResNet1(BasicBlock, [1, 1, 1, 1], device=device, num_classes=num_classes)


def ResNet34(device=None, num_classes=1000):
    return MyResNet1(BasicBlock, [3, 4, 6, 3], device=device, num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
