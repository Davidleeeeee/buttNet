""" Full assembly of the parts to form the complete network """

from .butterflynet_parts import *


class ButterflyNet(nn.Module):
    def __init__(self, n_classes, layer_numbers=3):
        super(ButterflyNet, self).__init__()
        self.n_channels = 1
        self.n_classes = n_classes
        self.layer_numbers = layer_numbers
        self.patch_encoder = encoder(layer_numbers=layer_numbers)
        self.img_encoder = encoder(layer_numbers=layer_numbers-1)
        self.ups1 = nn.ModuleList()
        self.ups2 = nn.ModuleList()
        for i in range(layer_numbers-1, -1, -1):
            self.ups1.append(Up(32 * times * (2 ** i), 16 * times * (2 ** i)))
        for i in range(layer_numbers - 2, -1, -1):
            self.ups2.append(Up(32 * times * (2 ** i), 16 * times * (2 ** i)))
        self.outc1 = OutConv(16 * times, n_classes)
        self.outc2 = OutConv(16 * times, n_classes)

    def new_layer_infer(self, x):
        for i in range(self.layer_numbers):
            x = self.ups[i](x)
        x = self.outc(x)
        return x

    def forward(self, x):
        x1 = self.patch_encoder(x)
        x2 = self.img_encoder(x)
        for i in range(self.layer_numbers):
            x1 = self.ups1[i](x1)
        for i in range(self.layer_numbers-1):
            x2 = self.ups2[i](x2)
        logit1 = self.outc1(x1)
        logit2 = self.outc2(x2)
        return logit1, logit2
