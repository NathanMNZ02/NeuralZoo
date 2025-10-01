import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class PixelDecoder(nn.Module):
    def __init__(self, in_channels_list, hidden_dim: int = 128):
        super().__init__()
        # conv separate per ogni feature map
        self.convs = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, kernel_size=3, padding=1) for c in in_channels_list
        ])
        
    def forward(self, f1, f2, f3, f4):
        target_size = f1.shape[-2:]
        
        # conv1x1 per uniformare canali
        f4 = self.convs[3](f4)
        f3 = self.convs[2](f3)
        f2 = self.convs[1](f2)
        f1 = self.convs[0](f1)

        # upsample tutte le feature map alla risoluzione di f1
        f4 = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)

        # somma top-down
        f = f1 + f2 + f3 + f4
        
        return f
    
class BinarySegmentator(nn.Module): 
    def __init__(self):
        super().__init__()
        
        self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2) # Estrazione delle feature 
        
        out_channels_list = [24, 40, 112, 960] # Numero di canali delle feature maps estratte   
        self.pixel_decoder = PixelDecoder(out_channels_list, hidden_dim=128) # Decodifica delle feature a livello di pixel
                
        # Upsample graduali
        self.skip_f4 = nn.Conv2d(960, 128, kernel_size=3, padding=1) # f4 -> 128 canali
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=1.416, mode='bilinear', align_corners=False),
            nn.Conv2d(128 + 128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) # Upsample x1.4142
        
        self.skip_f3 = nn.Conv2d(112, 64, kernel_size=3, padding=1) # f3 -> 64 canali
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=1.416, mode='bilinear', align_corners=False),
            nn.Conv2d(64 + 64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ) # Upsample x1.4142 
        
        self.skip_f2 = nn.Conv2d(40, 32, kernel_size=3, padding=1) # f2 -> 32 canali
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=1.418, mode='bilinear', align_corners=False),  # porta a ~600x600
            nn.Conv2d(32 + 32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        ) # Upsample x1.4142
        
        self.skip_f1 = nn.Conv2d(24, 16, kernel_size=3, padding=1) # f1 -> 16 canali
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=1.418, mode='bilinear', align_corners=False),  # porta a ~600x600
            nn.Conv2d(16 + 16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        ) # Upsample x1.4142
        
        # Refine head: attenzione ai canali in ingresso (32 + 32 = 64)
        self.refine_head = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=2, dilation=2),  # receptive field più ampio
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4), # ancora più ampio
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
                
    def get_size_in_mb(self):
        num_params = sum(p.numel() for p in self.parameters())
        size_in_MB = num_params * 4 / (1024**2)  # float32
        return size_in_MB
    
    def forward(self, x):
        # Backbone
        f1 = self.backbone.features[:3](x)    # b x 24 x 150 x 150
        f2 = self.backbone.features[3:6](f1)  # b x 40 x 75 x 75
        f3 = self.backbone.features[6:12](f2) # b x 112 x 38 x 38
        f4 = self.backbone.features[12:](f3)  # b x 960 x 19 x 19

        # Pixel decoder
        f = self.pixel_decoder(f1, f2, f3, f4) # b x 128 x 150 x 150

        # skip con f4 ->  Upsample 1
        f4_skip = F.interpolate(self.skip_f4(f4), size=f.shape[-2:], mode='bilinear', align_corners=False)
        f = torch.cat([f, f4_skip], dim=1)
        f = self.upsample1(f)               

        # skip con f3 ->  Upsample 2
        f3_skip = F.interpolate(self.skip_f3(f3), size=f.shape[-2:], mode='bilinear', align_corners=False)
        f = torch.cat([f, f3_skip], dim=1)
        f = self.upsample2(f) 
        
                # skip con f3 ->  Upsample 2
        f2_skip = F.interpolate(self.skip_f2(f2), size=f.shape[-2:], mode='bilinear', align_corners=False)
        f = torch.cat([f, f2_skip], dim=1)
        f = self.upsample3(f) 
        
        # skip con f3 ->  Upsample 2
        f1_skip = F.interpolate(self.skip_f1(f1), size=f.shape[-2:], mode='bilinear', align_corners=False)
        f = torch.cat([f, f1_skip], dim=1)
        f = self.upsample4(f) 

        # Refine head
        out = self.refine_head(f) # b x 1 x 600 x 600
        return out
    
b = BinarySegmentator()
print(b.get_size_in_mb())
print(b.forward(torch.randn(1, 3, 600, 600)).shape)

# Loss:
# Focal Loss --> concentra la loss sui pixel più complicati dove la rete sbaglia
# Dice Loss --> misura la sovrapposizione tra predizione e target