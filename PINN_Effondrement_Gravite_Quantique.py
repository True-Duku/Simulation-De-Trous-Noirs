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
# 2. PERTE PHYSIQUE AVEC CORRECTION LQG
# ==========================================
def calculer_loss_lqg(model, x, y, z, t, G=1.0, M0=5.0, R_min=0.8, t_bounce=0.8, rho_P=10.0):
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    
    V = model(x, y, z, t)
    
    # Dérivées premières
    dV_dx = torch.autograd.grad(V, x, torch.ones_like(V), create_graph=True)[0]
    dV_dy = torch.autograd.grad(V, y, torch.ones_like(V), create_graph=True)[0]
    dV_dz = torch.autograd.grad(V, z, torch.ones_like(V), create_graph=True)[0]
    
    # Dérivées secondes (Laplacien)
    d2V_dx2 = torch.autograd.grad(dV_dx, x, torch.ones_like(dV_dx), create_graph=True)[0]
    d2V_dy2 = torch.autograd.grad(dV_dy, y, torch.ones_like(dV_dy), create_graph=True)[0]
    d2V_dz2 = torch.autograd.grad(dV_dz, z, torch.ones_like(dV_dz), create_graph=True)[0]
    
    laplacien_V = d2V_dx2 + d2V_dy2 + d2V_dz2
    
    r2 = x**2 + y**2 + z**2
    
    # 1. Rayon de l'étoile avec rebond quantique (contourne le point 0)
    R_t = torch.sqrt(R_min**2 + 2.5 * (t - t_bounce)**2)
    
    # 2. Densité classique
    rho_centre = M0 / ((np.pi**1.5) * (R_t**3))
    rho_classique = rho_centre * torch.exp(-r2 / (R_t**2))
    
    # 3. Terme de correction de la Gravité Quantique à Boucles (LQG)
    # rho_eff = rho * (1 - rho / rho_P)
    rho_eff = rho_classique * (1.0 - rho_classique / rho_P)
    
    # Résidu de l'équation de Poisson-LQG
    f_pinn = laplacien_V - 4.0 * np.pi * G * rho_eff
    return torch.mean(f_pinn**2)

def generer_donnees(N_samples=2500):
    x = (torch.rand(N_samples, 1) - 0.5) * 10.0
    y = (torch.rand(N_samples, 1) - 0.5) * 10.0
    z = (torch.rand(N_samples, 1) - 0.5) * 10.0
    # Plage temporelle incluant l'effondrement et le rebond (t_bounce = 0.8)
    t = torch.rand(N_samples, 1) * 1.6
    return x.to(device), y.to(device), z.to(device), t.to(device)

# ==========================================
# 3. ENTRAÎNEMENT DU RÉSEAU
# ==========================================
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Entraînement du PINN LQG (Effondrement + Rebond)...")
epochs = 1200
for epoch in range(epochs):
    x_train, y_train, z_train, t_train = generer_donnees(N_samples=2500)
    optimizer.zero_grad()
    loss = calculer_loss_lqg(model, x_train, y_train, z_train, t_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 300 == 0:
        print(f"Époque [{epoch+1}/{epochs}] - Loss Physique LQG : {loss.item():.6f}")

print("Entraînement terminé. Génération de l'animation du rebond quantique...")

# ==========================================
# 4. ANIMATION 2D + 1D DU REBOND QUANTIQUE
# ==========================================
N_grid = 100
grid_lin = torch.linspace(-5.0, 5.0, N_grid)
X, Y = torch.meshgrid(grid_lin, grid_lin, indexing='ij')
Z = torch.zeros_like(X)

r_lin = torch.linspace(0.01, 5.0, 200).unsqueeze(1).to(device)
zeros = torch.zeros_like(r_lin)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Données initiales t = 0
t_init = 0.0
x_flat = X.reshape(-1, 1).to(device)
y_flat = Y.reshape(-1, 1).to(device)
z_flat = Z.reshape(-1, 1).to(device)

with torch.no_grad():
    V_2d = model(x_flat, y_flat, z_flat, torch.full_like(x_flat, t_init)).cpu().numpy().reshape(N_grid, N_grid)
    V_1d = model(r_lin, zeros, zeros, torch.full_like(r_lin, t_init)).cpu().numpy()

# Graphique 1 : Tranche 2D
im = ax1.imshow(V_2d, extent=[-5, 5, -5, 5], origin='lower', cmap='inferno', vmin=-12, vmax=1)
fig.colorbar(im, ax=ax1, label="Potentiel effectif V")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Graphique 2 : Profil Radial 1D
line, = ax2.plot(r_lin.cpu().numpy(), V_1d, color='royalblue', lw=2)
ax2.set_xlim(0, 5)
ax2.set_ylim(-12, 2)
ax2.set_xlabel("Distance au centre r")
ax2.set_ylabel("Potentiel V(r)")
ax2.grid(True)

# Frames couvrant t=0.0 à t=1.6 (Rebond à t=0.8)
frames = np.linspace(0.0, 1.6, 70)

def update(frame_t):
    t_tensor_2d = torch.full_like(x_flat, frame_t)
    t_tensor_1d = torch.full_like(r_lin, frame_t)
    
    with torch.no_grad():
        V_2d_t = model(x_flat, y_flat, z_flat, t_tensor_2d).cpu().numpy().reshape(N_grid, N_grid)
        V_1d_t = model(r_lin, zeros, zeros, t_tensor_1d).cpu().numpy()
        
    im.set_data(V_2d_t)
    line.set_ydata(V_1d_t)
    
    # Changement dynamique du titre selon la phase
    if frame_t < 0.7:
        phase = "Phase 1 : Effondrement"
    elif 0.7 <= frame_t <= 0.9:
        phase = "Phase 2 : Rebond Quantique (Densité Planck)"
    else:
        phase = "Phase 3 : Expulsion / Trou Blanc"
        
    ax1.set_title(f"Tranche 2D (z=0) à t = {frame_t:.2f}\n{phase}")
    ax2.set_title(f"Puits de potentiel V(r) à t = {frame_t:.2f}")
    return im, line

ani = animation.FuncAnimation(fig, update, frames=frames, interval=80, blit=False)

# Sauvegarde sous forme de GIF
ani.save("effondrement_lqg_bounce.gif", writer="pillow", fps=15)
plt.tight_layout()
plt.show()
print("Animation sauvegardée sous 'effondrement_lqg_bounce.gif'.")