'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) #, inplace=True)
        out = F.relu(self.bn2(self.conv2(out))) #, inplace=True)
        return out


class mobilenet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    #cfg = [64, 128, 128, 256, 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    cfg = [64, 128, 128, 256, 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, aux_classifiers_blocks=[]):
        super(mobilenet, self).__init__()
        self.is_aux = (aux_classifiers_blocks is not None)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1, in_planes_hook_layer1 = self._make_layers(in_planes = 32, cfg = [64, 128, 128, 256, 256])
        self.layer2, in_planes_hook_layer2 = self._make_layers(in_planes = in_planes_hook_layer1, cfg = [(512,2), 512, 512, 512, 512, 512])
        self.layer3, _ = self._make_layers(in_planes = in_planes_hook_layer2, cfg = [(1024,2), 1024])
        self.linear = nn.Linear(1024, num_classes)
        if self.is_aux:
            self.ep1_layer2, out_planes = self._make_layers(in_planes = in_planes_hook_layer1, cfg = [(512,2), 512, 512, 512])
            self.ep1_layer3, _ = self._make_layers(in_planes = out_planes, cfg = [(1024,2), 1024])
            self.ep1_linear = nn.Linear(1024, num_classes)

            self.ep2_layer3, _ = self._make_layers(in_planes = in_planes_hook_layer2, cfg = [(2048,2), 2048])
            self.ep2_linear = nn.Linear(2048, num_classes)


    def _make_layers(self, in_planes, cfg):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers), out_planes

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out_layer1 = self.layer1(out)
        out_layer2 = self.layer2(out_layer1)
        out = self.layer3(out_layer2)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        ep3_out = self.linear(out)

        if self.is_aux:
            out = self.ep1_layer2(out_layer1)
            out = self.ep1_layer3(out)
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            ep1_out = self.ep1_linear(out)

            out = self.ep2_layer3(out_layer2)
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            ep2_out = self.ep2_linear(out)
            return (ep1_out, ep2_out, ep3_out)

        return ep3_out
