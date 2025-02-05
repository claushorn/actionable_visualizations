import yaml
from sympy import plot
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class SemanticGroupingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, n_classes, version=2):
        super(SemanticGroupingNet, self).__init__()
        if version == 1:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim//2),  
                nn.ReLU(),
                nn.Linear(input_dim//2, embedding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, input_dim//2),
                nn.ReLU(),
                nn.Linear(input_dim//2, input_dim),
                nn.Sigmoid()
            )
        elif version == 2:
            dropout_rate=0.3
            # Encoder with BatchNorm and Dropout
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.BatchNorm1d(input_dim // 2),       # Batch normalization after the first Linear layer
                nn.ReLU(),
                nn.Dropout(dropout_rate),             # Dropout after activation
                nn.Linear(input_dim // 2, embedding_dim),
                nn.BatchNorm1d(embedding_dim)         # Batch normalization after embedding layer
            )
            # Decoder with BatchNorm and Dropout
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, input_dim // 2),
                nn.BatchNorm1d(input_dim // 2),       # Batch normalization
                nn.ReLU(),
                nn.Dropout(dropout_rate),             # Dropout after activation
                nn.Linear(input_dim // 2, input_dim),
                nn.Sigmoid()
            )

        # add softmax to map to classes
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, n_classes),
            nn.Softmax()
        )
    def forward(self, x): # for autoencoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def forward_supervised(self, x): # for encoder
        # add softmax to map to classes
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    def get_embedding(self, x):
        return self.encoder(x)
    
