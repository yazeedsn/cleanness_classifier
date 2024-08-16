import torch

class RandomNoise:
    def __init__(self, scales):
        self.scales = torch.tensor(scales)
    
    def __call__(self, input):
        transformed = input.clone()
        mean, std = input.mean(), input.std()
        scales = self.scales * torch.rand(size=(2,))
        sizes = (scales * input.shape[-2]).to(torch.uint8)
        dims = torch.tensor(input.shape[-2:])
        r = (dims - sizes)
        x, y = torch.randint(high=r[0], size=(1,)).item(), torch.randint(high=r[1], size=(1,)).item()
        x2, y2 = x+sizes[0].item(), y+sizes[1].item()
        x1, y1 = torch.randint(high=r[0], size=(1,)).item(), torch.randint(high=r[1], size=(1,)).item()
        x3, y3 = x1+sizes[0].item(), y1+sizes[1].item()
        transformed[..., x:x2, y:y2] += transformed[..., x1:x3, y1:y3]
        transformed[..., x:x2, y:y2] -= mean
        transformed[..., x:x2, y:y2] /= std
        return transformed
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"