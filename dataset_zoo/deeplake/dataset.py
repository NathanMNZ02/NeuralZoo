import torch
import deeplake
import numpy as np

# Percorso del dataset Extended Complex Scene Saliency Dataset (ECSSD)
ECSSD_DS_PATH = "hub://activeloop/ecssd"

class DeeplakeSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, load_path: str, transform=None):
        super().__init__()
        ds = deeplake.load(load_path)
        self.images = ds.images
        self.masks = ds.masks
        
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].numpy()
        if image.shape[2] == 1: image = np.concatenate([image]*3, axis=-1)
        mask = self.masks[idx].numpy().astype('uint8')
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Nota: image già in formato [C, H, W] grazie a ToTensorV2 (non applicato alla mask)
        mask = mask.permute(2, 0, 1).float() # [1, H, W] 
        return image, mask
