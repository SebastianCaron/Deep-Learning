import torch.nn as nn
import torchvision.models as models

class ArtClassifier(nn.Module):

    def __init__(self, nb_classes : int, latent_space_dim : int = 512, dropout : float = 0.4):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        nb_in = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.base_model = self.base_model
        
        self.latent_space = nn.Sequential(
            nn.Linear(nb_in, latent_space_dim),
            nn.BatchNorm1d(latent_space_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

        self.classifier = nn.Linear(latent_space_dim, nb_classes)
        
        nn.init.kaiming_normal_(self.latent_space[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.latent_space[0].bias, 0.0)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x):
        features = self.base_model(x)
        latent = self.latent_space(features)
        result = self.classifier(latent)
        return result
