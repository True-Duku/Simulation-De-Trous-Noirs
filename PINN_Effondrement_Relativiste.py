import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configuration du device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. ARCHITECTURE DU RÉSEAU (PINN)
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
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
# 2. PERTE PHYSIQUE
# ==========================================
def calculer_loss(model, x, y, z, t, G=1.0, M0=5.0, R0=4.0):
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    
    V = model(x, y, z, t)
    
    dV_dx = torch.autograd.grad(V, x, torch.ones_like(V), create_graph=True)[0]
    dV_dy = torch.autograd.grad(V, y, torch.ones_like(V), create_graph=True)[0]
    dV_dz = torch.autograd.grad(V, z, torch.ones_like(V), create_graph=True)[0]
    
    d2V_dx2 = torch.autograd.grad(dV_dx, x, torch.ones_like(dV_dx), create_graph=True)[0]
    d2V_dy2 = torch.autograd.grad(dV_dy, y, torch.ones_like(dV_dy), create_graph=True)[0]
    d2V_dz2 = torch.autograd.grad(dV_dz, z, torch.ones_like(dV_dz), create_graph=True)[0]
    
    laplacien_V = d2V_dx2 + d2V_dy2 + d2V_dz2
    
    r2 = x**2 + y**2 + z**2
    R_t = torch.clamp(R0 * (1.0 - (t / 2.0)**2), min=0.1)
    rho_centre = M0 / ((np.pi**1.5) * (R_t**3))
    rho_source = rho_centre * torch.exp(-r2 / (R_t**2))
    
    f_pinn = laplacien_V - 4.0 * np.pi * G * rho_source
    return torch.mean(f_pinn**2)

def generer_donnees(N_samples=2000):
    x = (torch.rand(N_samples, 1) - 0.5) * 10.0
    y = (torch.rand(N_samples, 1) - 0.5) * 10.0
    z = (torch.rand(N_samples, 1) - 0.5) * 10.0
    t = torch.rand(N_samples, 1) * 1.5
    return x.to(device), y.to(device), z.to(device), t.to(device)

# ==========================================
# 3. ENTRAÎNEMENT DU RÉSEAU
# ==========================================
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Entraînement du PINN en cours...")
epochs = 1000
for epoch in range(epochs):
    x_train, y_train, z_train, t_train = generer_donnees(N_samples=2000)
    optimizer.zero_grad()
    loss = calculer_loss(model, x_train, y_train, z_train, t_train)
    loss.backward()
    optimizer.step()

print("Entraînement terminé. Création de l'animation...")

# ==========================================
# 4. ANIMATION 2D + 1D
# ==========================================
# Préparation des grilles de test
N_grid = 100
grid_lin = torch.linspace(-5.0, 5.0, N_grid)
X, Y = torch.meshgrid(grid_lin, grid_lin, indexing='ij')
Z = torch.zeros_like(X)

r_lin = torch.linspace(0.01, 5.0, 200).unsqueeze(1).to(device)
zeros = torch.zeros_like(r_lin)

# Configuration de la figure Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Données initiales t = 0
t_init = 0.0
x_flat = X.reshape(-1, 1).to(device)
y_flat = Y.reshape(-1, 1).to(device)
z_flat = Z.reshape(-1, 1).to(device)
t_flat = torch.full_like(x_flat, t_init)

with torch.no_grad():
    V_2d = model(x_flat, y_flat, z_flat, t_flat).cpu().numpy().reshape(N_grid, N_grid)
    V_1d = model(r_lin, zeros, zeros, torch.full_like(r_lin, t_init)).cpu().numpy()

# Graphique 1 : Tranche 2D Heatmap
im = ax1.imshow(V_2d, extent=[-5, 5, -5, 5], origin='lower', cmap='inferno', vmin=-15, vmax=0)
fig.colorbar(im, ax=ax1, label="Potentiel V")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Graphique 2 : Profil Radial 1D
line, = ax2.plot(r_lin.cpu().numpy(), V_1d, color='firebrick', lw=2)
ax2.set_xlim(0, 5)
ax2.set_ylim(-15, 0.5)
ax2.set_xlabel("Distance au centre r")
ax2.set_ylabel("Potentiel V(r)")
ax2.grid(True)

# Fonction de mise à jour pour chaque frame
frames = np.linspace(0.0, 1.45, 60)

def update(frame_t):
    t_tensor_2d = torch.full_like(x_flat, frame_t)
    t_tensor_1d = torch.full_like(r_lin, frame_t)
    
    with torch.no_grad():
        V_2d_t = model(x_flat, y_flat, z_flat, t_tensor_2d).cpu().numpy().reshape(N_grid, N_grid)
        V_1d_t = model(r_lin, zeros, zeros, t_tensor_1d).cpu().numpy()
        
    im.set_data(V_2d_t)
    line.set_ydata(V_1d_t)
    
    ax1.set_title(f"Tranche 2D (z=0) à t = {frame_t:.2f}")
    ax2.set_title(f"Puits de potentiel V(r) à t = {frame_t:.2f}")
    return im, line

ani = animation.FuncAnimation(fig, update, frames=frames, interval=80, blit=False)

# Sauvegarde sous forme de GIF animé
ani.save("effondrement_pinn.gif", writer="pillow", fps=15)
plt.tight_layout()
plt.show()
print("Animation sauvegardée sous 'effondrement_pinn.gif'.")