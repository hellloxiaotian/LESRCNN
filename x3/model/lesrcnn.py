import torch
import torch.nn as nn
import model.ops as ops

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.EResidualBlock(64, 64, group=group)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale") #value of scale is scale. 
        multi_scale = kwargs.get("multi_scale") # value of multi_scale is multi_scale in args.
        group = kwargs.get("group", 1) #if valule of group isn't given, group is 1.
        kernel_size = 3 #tcw 201904091123
        kernel_size1 = 1 #tcw 201904091123
        padding1 = 0 #tcw 201904091124
        padding = 1     #tcw201904091123
        features = 64   #tcw201904091124
        groups = 1       #tcw201904091124
        channels = 3
        features1 = 64
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        '''
           in_channels, out_channels, kernel_size, stride, padding,dialation, groups,
        '''
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv8 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv12 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv13 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv14 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv15 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv16 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv17 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size1,padding=0,groups=groups,bias=False))
        self.conv17_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv17_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv17_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv17_4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv18 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=3,kernel_size=kernel_size,padding=padding,groups=groups,bias=False))
        '''
        self.conv18 =  nn.Conv2d(in_channels=features,out_channels=features1,kernel_size=kernel_size1,padding=padding1,groups=groups,bias=False)
        self.ReLU = nn.ReLU(inplace=True)
        '''
        self.ReLU=nn.ReLU(inplace=True)
        self.upsample = ops.UpsampleBlock(64, scale=scale, multi_scale=multi_scale,group=1)
    def forward(self, x, scale):
        #print '-------dfd'
        x = self.sub_mean(x)
        c0 = x
        x1 = self.conv1(x)
        x1_1 = self.ReLU(x1)
        x2 = self.conv2(x1_1)
        x3 = self.conv3(x2)
        x2_3 = x1+x3
        x2_4 = self.ReLU(x2_3)
        x4 = self.conv4(x2_4)
        x5 = self.conv5(x4)
        x3_5 = x2_3 + x5 
        x3_6 = self.ReLU(x3_5)
        x6 = self.conv6(x3_6)
        x7 = self.conv7(x6)
        x7_1 = x3_5 + x7 
        x7_2 = self.ReLU(x7_1)
        x8 = self.conv8(x7_2)
        x9 = self.conv9(x8)
        x9_2 = x7_1 + x9
        x9_1 = self.ReLU(x9_2)
        x10 = self.conv10(x9_1)
        x11 = self.conv11(x10)
        x11_1 = x9_2 + x11
        x11_2 = self.ReLU(x11_1)
        x12 = self.conv12(x11_2)
        x13 = self.conv13(x12)
        x13_1 = x11_1 + x13
        x13_2 = self.ReLU(x13_1)
        x14 = self.conv14(x13_2)
        x15 = self.conv15(x14)
        x15_1 = x15+x13_1
        x15_2 = self.ReLU(x15_1)
        x16 = self.conv16(x15_2)
        x17 = self.conv17(x16)
        x17_2 = x17 + x15_1 
        x17_3 = self.ReLU(x17_2)
        temp = self.upsample(x17_3, scale=scale)
        x1111 = self.upsample(x1_1, scale=scale) #tcw
        temp1 = x1111+temp #tcw
        temp2 = self.ReLU(temp1)
        temp3 = self.conv17_1(temp2)
        temp4 = self.conv17_2(temp3)
        temp5 = self.conv17_3(temp4)
        temp6 = self.conv17_4(temp5)
        x18 = self.conv18(temp6)
        out = self.add_mean(x18)
        #out = x18
        return out
