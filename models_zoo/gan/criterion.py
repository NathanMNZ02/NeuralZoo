import torch
import torch.nn as nn

### -----------------------------------------
### WGAN-GP
### -----------------------------------------

def gradient_penality(model_c: nn.Module, real_data: torch.Tensor, real_labels: torch.Tensor, fake_data: torch.Tensor, fake_labels: torch.Tensor) -> torch.Tensor:
    """
    Calcola la gradient penality per forzare il critico a essere una funzione 1-Lipschitz (varazione in uscita non supera la varazione in input).
    
    Formula: 
        lambda*Mean((||∇_{hat_x}(D(hat_x))|| - 1)^2) 
    dove hat_x è il batch di immagini interpolate tra le immagini reali e quella false.
    
    Args:
        model_c (nn.Module): istanza del critico.
        real_data (torch.Tensor): Dati reali.
        real_labels (torch.Tensor): Etichette dei dati reali.
        fake_data (torch.Tensor): Dati falsi.

    Returns:
        torch.Tensor: La perdita di gradient penalty.
    """
    batch_size = real_data.size(0)
    
    # Epsilon per ottenere una distribuzione uniforme
    eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
    eps = eps.expand_as(real_data)
    
    # Interpolazione tra dati reali e dati finti
    interpolated_images = eps * real_data + (1 - eps) * fake_data
    interpolated_images = interpolated_images.requires_grad_(True)
    interpolated_labels = real_labels
    
    # Ritorna gli scores sulle immagini
    interp_logits = model_c(interpolated_images, interpolated_labels)        
    grad_outputs = torch.ones_like(interp_logits, device=real_data.device)
    
    # Calcola il gradiente
    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=interpolated_images,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calcola e ritorna la norma del gradiente
    gradients = gradients.view(batch_size, -1)
    grad_norm = torch.norm(gradients, 2, dim=1)
    return torch.mean((grad_norm - 1) ** 2)

class EmdCriticLoss(nn.Module):
    """
    Funzione di perdita per il critico basata sulla distanza di Earth Mover (EMD).
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Calcola la perdita per il critico basata sulla distanza di Earth Mover (EMD).

        Args:
            real_logits (torch.Tensor): tensore contenente i logits dei punteggi reali.
            fake_logits (torch.Tensor): tensore contenente i logits punteggi falsi.

        Returns:
            torch.Tensor: valore scalare della perdita.
        """
        emd_loss = -(real_logits - fake_logits).mean()
        return emd_loss 
        
class EmdGeneratorLoss(nn.Module):
    """
    Funzione di perdita per il generatore basata sulla distanza di Earth Mover (EMD).
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, fake_logits: torch.Tensor):
        """
        Calcola la perdita per il generatore basata sulla distanza di Earth Mover (EMD).

        Args:
            fake_logits (torch.Tensor): tensore contenente i logits dei punteggi falsi da massimizzare.

        Returns:
            torch.Tensor: valore scalare della perdita.
        """
        emd_loss = -fake_logits.mean()
        return emd_loss