import torch 
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
from torchvision.transforms import *
from PIL import Image

class image_encoder(nn.Module):

  def __init__(self):
    super(image_encoder, self).__init__()

    self.layer_1 = nn.Linear(in_features = 2048, out_features = 1024)
    self.act = nn.ReLU()
    self.layer_2 = nn.Linear(in_features = 1024, out_features = 512)

  def forward(self, image_data):

    weight = torchvision.models.ResNet50_Weights.DEFAULT
    model = resnet50(weights = weight)

    model.fc = torch.nn.Sequential(*list(model.fc.children())[:-1])

    out_1 = model(image_data)
    out_2 = self.act(self.layer_1(out_1))
    image_vector = self.layer_2(out_2)

    return image_vector