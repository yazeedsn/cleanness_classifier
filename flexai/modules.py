import torch
from torch.nn.init import kaiming_uniform_
from torch.nn import Module, ReLU, Linear, Conv2d, BatchNorm2d, Sequential

class GeneralModule(Module):
    def __inti__(self):
        super().__init__()
    

class DenseLayer(GeneralModule):
    def __init__(self, in_f, out_f, act=ReLU(), norm=None, bias=True, init_func=lambda w,b: kaiming_uniform_(w)):
        super(DenseLayer, self).__init__()
        self.linear = Linear(in_f, out_f, bias=bias)
        if bias: b = self.linear.bias
        else: b = torch.empty([1])
        if init_func: init_func(self.linear.weight, b)
        if norm: self.norm = norm
        if act: self.act = act
    
    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, 'norm'): x = self.norm(x)
        if hasattr(self, 'act'): x = self.act(x)
        return x
    
    
class ConvLayer(GeneralModule):
    def __init__(self, in_c, out_c, ks=(3,3), stride=2, padding=1, act=ReLU(), bias=True, norm=None, init_func=lambda w,b: kaiming_uniform_(w)):
        super(ConvLayer, self).__init__()
        self.conv = Conv2d(in_c, out_c, ks, stride, padding, bias=bias)
        if bias: b = self.conv.bias
        else: b = torch.empty([1])
        if init_func: init_func(self.conv.weight, b)
        if norm: self.norm = norm
        if act: self.act = act

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'norm'): x = self.norm(x)
        if hasattr(self, 'act'): x = self.act(x)
        return x
    
    

class ResLayer(GeneralModule):
    def __init__(self, in_f, out_f, ks=(3,3), stride=2, padding=1, act=None, batch_norm=False, bias=True, init_func=lambda w,b: kaiming_uniform_(w)):
        super(ResLayer, self).__init__()
        self.conv1 = Conv2d(in_f, out_f, ks, stride, padding, bias=bias)
        if bias: b = self.conv1.bias 
        else: b = torch.empty([1])
        init_func(self.conv1.weight, b)
        if batch_norm: self.bn1 = BatchNorm2d(out_f)
        self.relu = ReLU() if not act else act
        self.conv2 = Conv2d(out_f, out_f, ks, 1, padding, bias=bias)
        if bias: b = self.conv2.bias
        else: b = torch.empty([1])
        init_func(self.conv2.weight, b)
        if batch_norm: self.bn2 = BatchNorm2d(out_f)
        self.downsample = None
        if (in_f != out_f) or stride != 1:
            layers = []
            layers.append(Conv2d(in_f, out_f, 1, stride, bias=False))
            init_func(layers[0].weight, torch.empty((1)))
            if batch_norm:
                layers.append(BatchNorm2d(out_f))
            self.downsample = Sequential(*layers)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        x = self.conv1(x)
        if hasattr(self, 'bn1'): x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if hasattr(self, 'bn2'): x = self.bn2(x)
        return self.relu(x + identity)
    