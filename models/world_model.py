import torch.nn as nn
import torch.nn.functional as F

#Fully Connected or Dense layer based autoencoder

class WorldModel(nn.Module):

    def __init__(self ):
        super(WorldModel, self).__init__()
        
        self.classifier = nn.Sequential(
                            nn.Conv2d(1, 32, 3, 1),
                            nn.ReLU(True),
                            nn.Conv2d(32, 64, 3, 1),
                            nn.ReLU(True),
                            nn.MaxPool2d(2),
                            nn.Dropout(0.25),
                            nn.Flatten(1),
                            nn.Linear(9216, 128),
                            nn.ReLU(True),
                            nn.Dropout(0.5),
                            nn.Linear(128, 64),
                            nn.ReLU(True),
                            nn.Linear(64, 10)
                            )

    
    def forward(self, data):
        X = self.classifier(data)
        return F.log_softmax(X, dim=1)
