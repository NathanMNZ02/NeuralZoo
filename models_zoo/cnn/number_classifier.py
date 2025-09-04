import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

### -----------------------------------------
### NET
### -----------------------------------------
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
        
class NumberClassifier(nn.Module):
    """
    Classificatore di numeri.

    Args:
        num_classes (int): numero di classi in cui classificare le immagini.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2, stride=2, padding=0), # 32x3x3 -> 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0), # 64x16x16 -> 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0), # 128x8x8 -> 256x4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0), # 256x4x4 -> 512x2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            
            nn.Flatten()
        )  
        self.fc = nn.Linear(512 * 2 * 2, out_features=num_classes)
        self.apply(weights_init)
        
    def forward(self, x: torch.Tensor, features: bool = False) -> torch.Tensor:
        x = self.model(x)
        l = self.fc(x)
        
        if features:
            return l, x
        return l
    
### -----------------------------------------
### TRAINER
### -----------------------------------------

class NumberClassifierTrainer:
    def __init__(
        self, 
        num_classes: int, 
        device: str
        ):
        self.device = device
        self.model = NumberClassifier(num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        self.history = {
            "train_loss": [],
            "valid_loss": [],
            "confusion_matrices": []
        }
        
    def __train_step__(
        self,
        images: torch.Tensor, 
        labels: torch.Tensor
        ) -> torch.Tensor:
        """
        Step di addestramento del modello.

        Args:
            images (torch.Tensor): tensore contenente il batch di immagini del dataset
            labels (torch.Tensor): tensore contenente il batch di labels del dataset

        Returns:
            torch.Tensor: perdita dello step di addestramento
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        preds = self.model(images) # Logits delle labels predette
        
        loss = self.criterion(preds, labels) 
        loss.backward()
        
        self.optimizer.step()
        return loss
        
    def __valid_step__(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
        ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Step di validazione del modello.

        Args:
            images (torch.Tensor): tensore contenente il batch di immagini del dataset
            labels (torch.Tensor): tensore contenente il batch di labels del dataset

        Returns:
            torch.Tensor: perdita dello step di validazione
            np.ndarray: tensore contenente le labels predette
            np.ndarray: tensore contenente le labels target
        """
        self.model.eval()
        with torch.no_grad():
            preds = self.model(images)
            loss = self.criterion(preds, labels)
            
            predicted_labels = preds.argmax(dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()
        
        return loss, predicted_labels, true_labels
    
    def __print_epoch__(
        epoch: int, 
        epochs: int, 
        train_loss: float, 
        valid_loss: float
        ):
        """
        Stampa il resoconto dell'epoca di addestramento

        Args:
            epoch (_type_): epoca attuale.
            epochs (int, optional): epoche di addestramento.
            train_loss (_type_): perdita di addestramento.
            valid_loss (_type_): perdita di validazione.
        """
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}")
        
    def train_loop(
        self, 
        train_dataloader: torch.utils.data.DataLoader, 
        valid_dataloader: torch.utils.data.DataLoader,
        save_path: str,
        epochs: int = 10, 
        patience: int = 5        
        ) -> dict[str, list]:
        """
        Loop di addestramento del modello.

        Args:
            train_dataloader (torch.utils.data.DataLoader): dataset di addestramento
            valid_dataloader (torch.utils.data.DataLoader): dataset di validazione
            save_path (str): percorso di salvataggio
            epochs (int, optional): epoche di addestramento, default 10.
            patience (int, optional): pazienza per cui non viene avviato l'early stopping, default 5.

        Returns:
            dict[str, list]: resoconto dell'addestramento.
        """ 
        
        best_valid_loss = 0.0
        for epoch in range(epochs):

            running_train_loss = 0.0
            pb_train_dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in pb_train_dataloader:
                loss = self.__train_step__(
                    batch["images"].to(self.device),
                    batch["labels"].to(self.device)
                )
                
                running_train_loss += loss.item()
                pb_train_dataloader.set_postfix(
                    train_loss = running_train_loss  / (pb_train_dataloader.n + 1)
                )
                
            avg_train_loss = running_train_loss / len(train_dataloader)
            self.history["train_loss"].append(avg_train_loss)
                
            running_valid_loss = 0.0
            all_preds, all_labels = [], []
            pb_valid_dataloader = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{epoch}", leave=False)
            for batch in pb_valid_dataloader:
                loss, preds, labels = self.__valid_step__(
                    batch["images"].to(self.device),
                    batch["labels"].to(self.device)
                )
                
                running_valid_loss += loss.item()
                all_preds.extend(preds)
                all_labels.extend(labels)
                pb_valid_dataloader.set_postfix(
                    valid_loss = running_valid_loss  / (pb_train_dataloader.n + 1)
                )
                
            avg_valid_loss = running_valid_loss / len(valid_dataloader)
            self.history["valid_loss"].append(avg_valid_loss)
            
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.model.num_classes)))
            self.history["confusion_matrices"].append(cm)
                        
            self.__print_epoch__(epoch, epochs, avg_train_loss, avg_valid_loss) 
            if avg_valid_loss < best_valid_loss:
                self.best_valid_loss = avg_valid_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), save_path)
                print("  --> Model saved!")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epochs.")
                
            if self.epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        return self.history

### -----------------------------------------
### VISUALIZER
### -----------------------------------------

class NumberClassifierVisualizer:
    def __init__(
        self,
        train_history: dict[str, list],
        num_classes: int):
        
        self.history = train_history
        self.num_classes = num_classes
        
    def plot_losses(self):
        """
        Crea un grafico con le perdite del modello.
        """
        plt.figure(figsize=(8,5))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["valid_loss"], label="Valid Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_cm(
        self,         
        epoch: int = -1):
        """
        Crea un grafico con la matrice di confusione del modello.

        Args:
            epoch (int, optional): epoca per cui creare il grafico di confusione.
        """
        if epoch >= len(self.history["conf_matrices"]):
            epoch = -1
            
        cm = self.history["conf_matrices"][epoch]
                
        plt.figure(figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=list(range(self.num_classes)),
                    yticklabels=list(range(self.num_classes)))
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Confusion Matrix (epoch {epoch if epoch>=0 else len(self.history['conf_matrices'])})")
        plt.show()

### -----------------------------------------
### PREDICTOR
### -----------------------------------------

class NumberClassifierPredictor:
    def __init__(
        self, 
        model_path: str, 
        num_classes: int, 
        device: str):
        
        self.device = device
        self.model = NumberClassifier(num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
    def predict(
        self, 
        x: torch.Tensor
        ) -> np.ndarray:
        """
        Effettua le predizioni sul modello di classificazione numeri.

        Args:
            x (torch.Tensor): immagini in input

        Returns:
            torch.Tensor: labels predette
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            preds = self.model(x)
            predicted_labels = preds.argmax(dim=1).cpu().numpy()
        return predicted_labels

        
        