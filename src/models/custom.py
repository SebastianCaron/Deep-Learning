
import torch.nn as nn


class CustomArtCNN(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        
        self.features = nn.Sequential(
            self._make_block(3, 64),
            self._make_block(64, 128),
            self._make_block(128, 256),
            self._make_block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        self._initialize_weights()

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        embedding = self.embedder(features)
        return self.classifier(embedding)