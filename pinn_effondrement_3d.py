import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Configuration du device (CPU suffit amplement ici)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. ARCHITECTURE DU RÉSEAU (PINN)
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Entrées : (x, y, z, t) -> 4 neurones
        # Sortie  : Potentiel V   -> 1 neurone
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        return self.net(inputs)

# ==========================================
# 2. PERTE PHYSIQUE (Dérivation Automatique)
# ==========================================
def calculer_loss(model, x, y, z, t, G=1.0, M0=5.0, R0=4.0):
    # Activer le calcul des gradients par rapport aux coordonnées spatiales
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    
    # Prédiction du potentiel V(x,y,z,t)
    V = model(x, y, z, t)
    
    # Dérivées premières dV/dx, dV/dy, dV/dz
    dV_dx = torch.autograd.grad(V, x, torch.ones_like(V), create_graph=True)[0]
    dV_dy = torch.autograd.grad(V, y, torch.ones_like(V), create_graph=True)[0]
    dV_dz = torch.autograd.grad(V, z, torch.ones_like(V), create_graph=True)[0]
    
    # Dérivées secondes d^2V/dx^2, etc. (Laplacien)
    d2V_dx2 = torch.autograd.grad(dV_dx, x, torch.ones_like(dV_dx), create_graph=True)[0]
    d2V_dy2 = torch.autograd.grad(dV_dy, y, torch.ones_like(dV_dy), create_graph=True)[0]
    d2V_dz2 = torch.autograd.grad(dV_dz, z, torch.ones_like(dV_dz), create_graph=True)[0]
    
    laplacien_V = d2V_dx2 + d2V_dy2 + d2V_dz2
    
    # Densité source analytique rho(r, t) pour le guidage physique
    r2 = x**2 + y**2 + z**2
    R_t = torch.clamp(R0 * (1.0 - (t / 2.0)**2), min=0.1)
    rho_centre = M0 / ((np.pi**1.5) * (R_t**3))
    rho_source = rho_centre * torch.exp(-r2 / (R_t**2))
    
    # Residual de l'équation de Poisson : nabla^2 V - 4*pi*G*rho = 0
    f_pinn = laplacien_V - 4.0 * np.pi * G * rho_source
    loss_physics = torch.mean(f_pinn**2)
    
    return loss_physics

# ==========================================
# 3. ENTRAÎNEMENT DU RÉSEAU
# ==========================================
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Génération de points de colocalisation aléatoires dans l'espace 3D + Temps
N_samples = 2000
x_train = (torch.rand(N_samples, 1) - 0.5) * 10.0
y_train = (torch.rand(N_samples, 1) - 0.5) * 10.0
z_train = (torch.rand(N_samples, 1) - 0.5) * 10.0
t_train = torch.rand(N_samples, 1) * 1.5

print("Début de l'entraînement du PINN...")
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = calculer_loss(model, x_train, y_train, z_train, t_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Époque [{epoch+1}/{epochs}] - Loss Physique : {loss.item():.6f}")

print("Entraînement terminé !")