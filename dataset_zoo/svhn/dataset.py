import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from scipy.io import loadmat

def get_svhn_default_transform():
    """
    Ritorna le trasformazioni di default applicare sulle immagini e sulle annotazioni del dataset
    Street View House Number.
    
    Returns:
        A.Compose
    """
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2()
    ])
    
class StreetViewHouseNumbersDataset(torch.utils.data.Dataset):
    """
    Classe Dataset per rappresentare Street View House Numbers (SVHN).

    Args:
        file_path (str): Percorso al file .mat contenente il dataset.
        transform (callable, optional): Trasformazioni da applicare alle immagini.
        percentage (float): percentuale di elementi caricati dal dataset
    """
    def __init__(self, mat_file: str): 
        self.name = "Street View House Numbers"
        self.default_transform = get_svhn_default_transform()  

        data = loadmat(mat_file)
        self.images = data['X'] # Le immagini sono in un array 4D (32, 32, 3, N)
        self.labels = data['y'].squeeze() # Le etichette sono in un array 1D (N,)
                
    def __len__(self):
        return self.images.shape[3]
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        image_np = self.images[:, :, :, idx]  # Estrazione dell'immagine        
        transformed = self.default_transform(image = image_np)
        label = self.labels[idx] - 1  # Le etichette sono da 1 a 10, convertiamo in 0-9
        
        target = {
            "image": transformed["image"],
            "label": torch.tensor(label, dtype=torch.long),
        }
        return target