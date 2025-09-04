import torch
import torch.nn as nn
import torch.nn.functional as F
import models_zoo.gan.criterion as ct
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from torchvision.utils import make_grid
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from models_zoo.cnn.number_classifier import NumberClassifierPredictor

### -----------------------------------------
### UTILS
### -----------------------------------------

def create_grid(
    gen_images: torch.Tensor, 
    nrow: int,
    padding: int = 2) -> np.ndarray:
    """
    Crea una griglia di immagini generate e la salva su disco.

    Args:
        gen_images (torch.Tensor): Batch di immagini generate, shape (B, 3, H, W).
        nrow (int): Numero di immagini per riga nella griglia.
        padding (int, optional): Spaziatura tra le immagini nella griglia. Defaults to 2.
    """
    images = (gen_images + 1) / 2 # [-1, 1] -> [0, 1]
    grid = make_grid(images, nrow=nrow, padding=padding).numpy()
    return grid

### -----------------------------------------
### NET
### -----------------------------------------

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
        
class ConvCritic(nn.Module):
    """
    Critico convoluzionale per valutare se un'immagine è reale o no.
        
    Args:
        num_classes (int): Numero di classi nel dataset.
        ndf (int): Numero di feature maps del discriminatore.
        nc (int): Numero di canali in input dell'immagine.
    """
    def __init__(self, num_classes: int, ndf: int = 64, nc: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(nc + num_classes, ndf, kernel_size=4, stride=2, padding=1, bias=False), # nc * 2 x 32 x 32 -> ndf x 16 x 16
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), # ndf x 16 x 16 -> 2*ndf x 8 x 8
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), # 2*ndf*8x8 -> 4*ndf*4x4
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=0, bias=False), # 4*ndf*4x4 -> 1x1x1
            nn.Flatten()
        )
        self.apply(weights_init)
        
    def forward(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        l = F.one_hot(l, num_classes=self.num_classes).float()
        l = l.unsqueeze(2).unsqueeze(3)
        l = l.expand(-1, -1, x.size(2), x.size(3))
        x_cond = torch.cat([x, l], dim=1)
        
        x = self.model(x_cond)     
        return x

class ConvGenerator(nn.Module):
    """
    Generatore convoluzionale per generare immagini.

    Args:
        num_classes (int): Numero di classi nel dataset.
        nz (int): Dimensione del vettore di rumore.
        ngf (int): Numero di feature maps nel generatore.
        nc (int): Numero di canali in output dell'immagine.
    """
    def __init__(self, num_classes: int, nz: int = 100, ngf: int = 64, nc: int = 3):
        super(ConvGenerator, self).__init__()
        self.num_classes = num_classes
        self.nz = nz
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz + num_classes, ngf * 4, 4, 1, 0, bias=False), # ngf*4 x 4 x 4
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # ngf*4 x 4 x 4 -> ngf*2 x 8 x 8
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), # ngf*2 x 8 x 8 -> ngf x 16 x 16
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), # ngf x 16 x 16 -> nc x 32 x 32
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor, l: torch.Tensor):
        l = F.one_hot(l, num_classes=self.num_classes).float()
        z_cond = torch.cat([z, l], dim=1)
        z_cond = z_cond.unsqueeze(2).unsqueeze(3) # [B, nz+num_classes] -> [B, nz+num_classes, 1, 1]
        return self.model(z_cond)
    
### -----------------------------------------
### TRAINER
### -----------------------------------------

class NumberGeneratorTrainer:
    def __init__(
        self,
        num_classes: int, 
        classifier: NumberClassifierPredictor,
        device: str
        ):   
        self.device = device
        self.model_c = ConvCritic(num_classes, 64, 3).to(device)
        self.model_g = ConvGenerator(num_classes, 100, 64, 3).to(device)
        self.classifier = classifier
        
        self.optimizer_c = torch.optim.Adam(self.model_c.parameters(), lr=1e-4, betas=(0.0, 0.9))
        self.optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=1e-4, betas=(0.0, 0.9))
        
        self.criterion_c = ct.EmdCriticLoss()
        self.criterion_g = ct.EmdGeneratorLoss()
        
        self.fixed_z = torch.randn(64, self.model_g.nz, device=device)
        self.fixed_labels = torch.randint(0, num_classes, (64,), dtype=torch.long, device=device)
        
        self.history = {
            "critic_loss": [],
            "generator_loss": [],
            "confusion_matrices": [],
            "peak": [],
            "rmse": [],
            "grid": []
        }
        
    def __train_step__(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        critic_steps: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Step di addestramento del modello.

        Args:
            images (torch.Tensor): tensore contenente il batch di immagini del dataset
            labels (torch.Tensor): tensore contenente il batch di labels del dataset

        Returns:
            torch.Tensor: perdita del critico
            torch.Tensor: perdita del generatore
        """
        self.model_c.train()
        self.model_g.eval()
        
        B = images.shape[0]

        critic_loss = 0
        for _ in range(critic_steps):
            self.optimizer_c.zero_grad()
            
            # 1. Estrazione degli score per le immagini reali
            real_scores = self.model_c(images, labels)
            
            # 2. Estrazione degli score per le immagini finte
            z = torch.randn(B, self.model_g.nz, device=self.device)
            f_labels = torch.randint(0, self.model_g.num_classes, (B,), dtype=torch.long, device=self.device)
            f_images = self.model_g(z, f_labels)
            
            fake_scores = self.model_c(f_images, f_labels)
            
            # 3. Calcolo della loss del critico
            loss = self.criterion_c(real_scores, fake_scores) + ct.gradient_penality(self.model_c, images, labels, f_images, f_labels)  
            loss.backward()
            
            self.optimizer_c.step()
            critic_loss += loss
            
        critic_loss /= critic_steps
        
        self.model_c.eval()
        self.model_g.train()
        
        self.optimizer_g.zero_grad()
        
        z = torch.randn(B, self.model_g.nz, device=self.device)
        f_labels = torch.randint(0, self.model_g.num_classes, (B,), dtype=torch.long, device=self.device)
        f_images = self.model_g(z, f_labels)
        
        fake_scores = self.model_c(f_images, f_labels)
        
        # 4. Il generatore prova ad ingannare il critico
        generator_loss = self.criterion_g(fake_scores)
        generator_loss.backward()
        self.optimizer_g.step()
        
        return critic_loss, generator_loss
    
    def __valid_step__(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
        ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Step di validazione del modello.

        Args:
            images (torch.Tensor): tensore contenente il batch di immagini del dataset
            labels (torch.Tensor): tensore contenente il batch di labels del dataset

        Returns:
            np.ndarray: tensore contenente le labels predette
            np.ndarray: tensore contenente le labels target
            torch.Tensor: Peak Signal-to-Noise Ratio 
            torch.Tensor: errore quadratico medio
        """
        self.model_g.eval()
        with torch.no_grad():
            B = images.shape[0]
            
            z = torch.randn(B, self.model_g.nz, device=self.device)
            f_images = self.model_g(z, labels)
            
            # 1. Classifica le immagini
            predicted_labels = self.classifier.predict(x=f_images)
            true_labels = labels.cpu().numpy()
            
            f_images_norm = (f_images + 1) / 2 # [-1, 1] -> [0, 1]
            images_norm = (images + 1) / 2
            
            # 2. Calcola PSNR per capire quanto sono simili le immagini generate da quelle presenti nel dataset
            MAX = 1
            psnr = 10 * torch.log10(MAX**2 / (f_images_norm - images_norm).pow(2).mean()) # Se alto allora immagine generate simili all'originale
            
            # 3. Calcolo del RMSE per capire quanto simili sono le immagini generate da quelle presenti nel dataset
            rmse = torch.sqrt((f_images_norm - images_norm).pow(2).mean()) # Se basso allora immagine generate simili all'originale
        
        return predicted_labels, true_labels, psnr, rmse
    
    def __print_epoch__(
        self, 
        epoch: int, 
        epochs: int,
        critic_loss: float,
        generator_loss: float,
        peak: float, 
        rmse: float
        ):
        """
        Stampa il resoconto dell'epoca di addestramento.

        Args:
            epoch (_type_): epoca attuale.
            epochs (int, optional): epoche di addestramento.
            critic_loss (float): perdita del critico.
            generator_loss (float): perdita del generatore.
            peak (float): metrica peak.
            rmse (float): metrica rmse.
        """
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Critic Loss: {critic_loss:.4f}")
        print(f"  Generator Loss: {generator_loss:.4f}")
        print(f"  Peak: {peak:.4f}")
        print(f"  Rmse: {rmse:.4f}")

    def train_loop(
        self, 
        train_dataloader: torch.utils.data.DataLoader, 
        valid_dataloader: torch.utils.data.DataLoader,
        save_path: str,
        epochs: int = 50,
        ) -> dict[str, list]:
        """
        Loop di addestramento del modello.

        Args:
            train_dataloader (torch.utils.data.DataLoader): dataset di addestramento
            valid_dataloader (torch.utils.data.DataLoader): dataset di validazione
            save_path (str): percorso di salvataggio
            epochs (int, optional): epoche di addestramento, default 10.

        Returns:
            dict[str, list]: resoconto dell'addestramento.
        """
        
        for epoch in range(epochs):
            
            running_critic_loss = 0.0
            running_generator_loss = 0.0
            pb_train_dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in pb_train_dataloader:
                critic_loss, generator_loss = self.__train_step__(
                    batch["images"].to(self.device),
                    batch["labels"].to(self.device),
                    critic_steps=5
                )
                
                running_critic_loss += critic_loss.item()
                running_generator_loss += generator_loss.item()
                pb_train_dataloader.set_postfix(
                    critic_loss = running_critic_loss / (pb_train_dataloader.n + 1),
                    generator_loss = running_generator_loss / (pb_train_dataloader.n + 1)
                )
                
            avg_critic_loss = running_critic_loss / len(train_dataloader)
            avg_generator_loss = running_generator_loss / len(train_dataloader)
            self.history["critic_loss"].append(avg_critic_loss)
            self.history["generator_loss"].append(avg_generator_loss)
            
            running_peak = 0.0
            running_rmse = 0.0
            all_preds, all_labels = [], []
            pb_valid_dataloader = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{epoch}", leave=False)
            for batch in pb_valid_dataloader:
                preds, labels, peak, rmse  = self.__valid_step__(
                    batch["images"].to(self.device),
                    batch["labels"].to(self.device),
                )
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                running_peak += peak.item()
                running_rmse += rmse.item()
                pb_valid_dataloader.set_postfix(
                    peak = running_peak / (pb_valid_dataloader.n + 1),
                    rmse = running_rmse / (pb_valid_dataloader.n + 1)
                )
                
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.model_g.num_classes)))
            self.history["confusion_matrices"].append(cm)
            
            avg_peak = running_peak / len(valid_dataloader)
            avg_rmse = running_rmse / len(valid_dataloader)
            self.history["peak"].append(avg_peak)
            self.history["rmse"].append(avg_rmse)

            self.model_g.eval()
            with torch.no_grad():
                f_images = self.model_g(self.fixed_z, self.fixed_labels)
                self.history["grid"].append(create_grid(f_images, nrow=4))
                
            self.__print_epoch__(epoch, epochs, avg_critic_loss, avg_generator_loss, avg_peak, avg_rmse) 

        torch.save(self.model_g.state_dict(), save_path)

### -----------------------------------------
### VISUALIZER
### -----------------------------------------

class NumberGeneratorVisualizer:
    def __init__(self, train_history: dict[str, list], num_classes: int):
        self.history = train_history
        self.num_classes = num_classes

    def plot_losses(self):
        """
        Grafico delle perdite del Critico e del Generatore.
        """
        plt.figure(figsize=(8,5))
        plt.plot(self.history["critic_loss"], label="Critic Loss")
        plt.plot(self.history["generator_loss"], label="Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Critic and Generator Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_metrics(self):
        """
        Grafico di PSNR e RMSE durante l'addestramento.
        """
        plt.figure(figsize=(8,5))
        plt.plot(self.history["peak"], label="PSNR")
        plt.plot(self.history["rmse"], label="RMSE")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title("PSNR and RMSE")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_cm(self, epoch: int = -1):
        """
        Grafico della matrice di confusione.
        """
        if epoch >= len(self.history["confusion_matrices"]):
            epoch = -1

        cm = self.history["confusion_matrices"][epoch]
                
        plt.figure(figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=list(range(self.num_classes)),
                    yticklabels=list(range(self.num_classes)))
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Confusion Matrix (epoch {epoch if epoch>=0 else len(self.history['confusion_matrices'])})")
        plt.show()
        
    def plot_grid(self, epoch: int = -1):
        """
        Mostra la griglia di immagini generate.
        """
        if epoch >= len(self.history["grid"]):
            epoch = -1

        grid = self.history["grid"][epoch]
        plt.figure(figsize=(8,8))
        plt.imshow(np.transpose(grid, (1,2,0)))
        plt.axis('off')
        plt.title(f"Generated Images (epoch {epoch if epoch>=0 else len(self.history['grid'])})")
        plt.show()
    
### -----------------------------------------
### PREDICTOR
### -----------------------------------------

class NumberGeneratorPredictor:
    def __init__(
        self,
        model_path: str,
        num_classes: int,
        device: str):
        
        self.device = device
        self.model = ConvGenerator(num_classes, 100, 32, 3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
    def generate(
        self, 
        l: torch.Tensor
        ) -> torch.Tensor:
        """
        Genera un immagine con il modello di generazione immagini.

        Args:
            l (torch.Tensor): label in input.

        Returns:
            torch.Tensor: immagine generata
        """
        self.model.eval()
        with torch.no_grad():
            l = l.to(self.device)
            z = torch.randn(l.shape[0], self.model.nz, device=self.device)
            
            img = self.model(z, l)
        return img
        