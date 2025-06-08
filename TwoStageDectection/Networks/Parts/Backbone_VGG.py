import jittor as jt
from jittor import nn, Module


def ConvBase(in_c, out_c, kernel_size, stride, padding):
    base = nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.Relu(),
    )
    jt.init.relu_invariant_gauss_(base[0].weight, mode="fan_out")

    return base


def VggBase(in_c, out_c, num):
    mainconvs = nn.Sequential(
        ConvBase(in_c=in_c, out_c=out_c, kernel_size=3, stride=1, padding=1),
    )
    for i in range(num - 1):
        mainconvs.append(ConvBase(in_c=out_c, out_c=out_c, kernel_size=3, stride=1, padding=1),
                         )
    return mainconvs


def Vgg(setlist, dropout=0.5):
    info_cap = nn.Sequential(
        VggBase(in_c=3, out_c=64, num=setlist[0]),
        nn.MaxPool2d(kernel_size=2, stride=2),
        VggBase(in_c=64, out_c=128, num=setlist[1]),
        nn.MaxPool2d(kernel_size=2, stride=2),
        VggBase(in_c=128, out_c=256, num=setlist[2]),
        nn.MaxPool2d(kernel_size=2, stride=2),
        VggBase(in_c=256, out_c=512, num=setlist[3]),
        nn.MaxPool2d(kernel_size=2, stride=2),
        VggBase(in_c=512, out_c=512, num=setlist[4]),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    classfier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

    jt.init.relu_invariant_gauss_(classfier[0].weight, mode="fan_out")
    jt.init.relu_invariant_gauss_(classfier[3].weight, mode="fan_out")

    return info_cap, classfier
