import torch.nn as nn
import torchvision.models as models

class ArtClassifier(nn.Module):
    def __init__(self, nb_classes: int, latent_space_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        nb_in = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(nb_in, latent_space_dim),
            nn.BatchNorm1d(latent_space_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(latent_space_dim, nb_classes)
        )

    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)