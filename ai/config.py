from torchvision.models import resnet18
from torch.nn import Sequential, Linear, ReLU, Dropout

PRETRAINED = True
DEFAULT_MODEL = resnet18(pretrained=PRETRAINED)
DEFAULT_MODEL_FC = Sequential(
    ReLU(),
    Dropout(p=0.7),
    Linear(DEFAULT_MODEL.fc.in_features, 128),
    ReLU(),
    Dropout(0.7),
    Linear(128, 2)
)