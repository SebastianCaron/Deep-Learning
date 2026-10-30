import torch.nn as nn
import torchvision.models as models

class ArtClassifier(nn.Module):
    def __init__(self, nb_classes: int):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        nb_in = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(nb_in, nb_classes)

    def forward(self, x):
        return self.base_model(x)