import jittor as jt
from jittor import nn, Module

def ConvBaseR(in_c, out_c, kernel_size, stride, padding):
    base = nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )
    jt.init.relu_invariant_gauss_(base[0].weight, mode="fan_out")

    return base

def ConvBase(in_c, out_c, kernel_size, stride, padding):
    base = nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_c),
    )
    jt.init.relu_invariant_gauss_(base[0].weight, mode="fan_out")

    return base

class ResnetBase(Module):
    def __init__(self, in_c, out_c, num, stride_in=1):
        super().__init__()
        self.C = 4
        self.num = num
        self.relu = nn.ReLU(inplace=True)

        self.AuxConv = ConvBase(in_c=in_c, out_c=out_c, kernel_size=1,
                                stride=stride_in, padding=0)

        self.mainconvs = [nn.Sequential(
            ConvBaseR(in_c=in_c, out_c=out_c // self.C, kernel_size=1,
                     stride=1, padding=0),
            ConvBaseR(in_c=out_c // self.C, out_c=out_c // 4, kernel_size=3,
                     stride=stride_in, padding=1),
            ConvBase(in_c=out_c // self.C, out_c=out_c, kernel_size=1,
                     stride=1, padding=0),
        )]
        for i in range(num - 1):
            self.minconvs.append(
                nn.Sequential(
                    ConvBaseR(in_c=out_c, out_c=out_c // self.C, kernel_size=1,
                             stride=1, padding=0),
                    ConvBaseR(in_c=out_c // self.C, out_c=out_c // 4, kernel_size=3,
                             stride=1, padding=1),
                    ConvBase(in_c=out_c // self.C, out_c=out_c, kernel_size=1,
                             stride=1, padding=0),
                )
            )

    def forward(self, x):
        x1 = self.AuxConv(x)
        for i in range(self.num):
            x = self.mainconvs[i](x)
            x += x1
            if i < self.num - 1:
                x1 = x

        return x

def BackboneResnet(setlist):
    info_cap = nn.Sequential(
        ConvBaseR(in_c=3, out_c=64, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=2, stride=2),
        ResnetBase(in_c=64, out_c=256, num=setlist[0]),
        ResnetBase(in_c=256, out_c=512, num=setlist[1], stride_in=2),
        ResnetBase(in_c=512, out_c=1024, num=setlist[2], stride_in=2),
    )
    classfier = nn.Sequential(
        ResnetBase(in_c=1024, out_c=2048, num=setlist[3], stride_in=2),
        nn.AdaptiveAvgPool2d(7),
    )

    return info_cap, classfier





