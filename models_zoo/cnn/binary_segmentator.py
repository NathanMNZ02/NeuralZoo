import torch
import torch.nn as nn

class BinarySegmentator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = None # Estrazione delle feature 
        self.pixel_decoder = None # Decodifica delle feature a livello di pixel
        
        self.classifier = None # Predizione del valore di ogni pixel (foreground/background)
    
    def forward(self, x):
        features = self.backbone(x)
        pixel_features = self.pixel_decoder(features)
        
        logits = self.classifier(pixel_features)
        
        return logits
    
# Loss:
# Focal Loss --> concentra la loss sui pixel più complicati dove la rete sbaglia
# Dice Loss --> misura la sovrapposizione tra predizione e target