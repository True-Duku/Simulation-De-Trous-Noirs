import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Scikit-Learn & UMAP pour le Manifold Learning
from sklearn.manifold import MDS
import umap

# Configuration du device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. ARCHITECTURE DU RÉSEAU (PINN)
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        self.activation = nn.Tanh()

    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        h1 = self.activation(self.fc1(inputs))
        h2 = self.activation(self.fc2(h1))
        h3 = self.activation(self.fc3(h2)) # Représentation latente (64D)
        return self.out(h3)

    def extract_latent(self, x, y, z, t):
        """ Extrait la représentation interne (avant la dernière couche) """
        inputs = torch.cat([x, y, z, t], dim=1)
        h1 = self.activation(self.fc1(inputs))
        h2 = self.activation(self.fc2(h1))
        h3 = self.activation(self.fc3(h2))
        return h3

# ==========================================
# 2. PERTE PHYSIQUE AVEC CORRECTION LQG
# ==========================================
def calculer_loss_lqg(model, x, y, z, t, G=1.0, M0=5.0, R_min=0.8, t_bounce=0.8, rho_P=10.0):
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
    R_t = torch.sqrt(R_min**2 + 2.5 * (t - t_bounce)**2)
    
    rho_centre = M0 / ((np.pi**1.5) * (R_t**3))
    rho_classique = rho_centre * torch.exp(-r2 / (R_t**2))
    
    # Correction LQG
    rho_eff = rho_classique * (1.0 - rho_classique / rho_P)
    
    f_pinn = laplacien_V - 4.0 * np.pi * G * rho_eff
    return torch.mean(f_pinn**2)

def generer_donnees(N_samples=2500):
    x = (torch.rand(N_samples, 1) - 0.5) * 10.0
    y = (torch.rand(N_samples, 1) - 0.5) * 10.0
    z = (torch.rand(N_samples, 1) - 0.5) * 10.0
    t = torch.rand(N_samples, 1) * 1.6
    return x.to(device), y.to(device), z.to(device), t.to(device)

# ==========================================
# 3. ENTRAÎNEMENT DU PINN
# ==========================================
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Entraînement du PINN LQG...")
epochs = 1200
for epoch in range(epochs):
    x_train, y_train, z_train, t_train = generer_donnees(N_samples=2500)
    optimizer.zero_grad()
    loss = calculer_loss_lqg(model, x_train, y_train, z_train, t_train)
    loss.backward()
    optimizer.step()

print("Entraînement terminé !")

# ==========================================
# 4. MANIFOLD LEARNING (MDS & UMAP)
# ==========================================
print("Extraction des représentations latentes pour l'analyse de variété...")

# Échantillonnage de points fixes à différents instants t
N_pts = 300
times = np.linspace(0.0, 1.6, 9) # 9 pas de temps de t=0 à t=1.6

x_eval = (torch.rand(N_pts, 1) - 0.5) * 6.0
y_eval = (torch.rand(N_pts, 1) - 0.5) * 6.0
z_eval = (torch.rand(N_pts, 1) - 0.5) * 6.0

latent_vectors = []
time_labels = []

model.eval()
with torch.no_grad():
    for t_val in times:
        t_eval = torch.full_like(x_eval, t_val)
        h3 = model.extract_latent(x_eval.to(device), y_eval.to(device), z_eval.to(device), t_eval.to(device))
        
        latent_vectors.append(h3.cpu().numpy())
        time_labels.append(np.full(N_pts, t_val))

X_latent = np.vstack(latent_vectors) # Matrice de taille (9 * N_pts, 64)
labels = np.concatenate(time_labels)

# 1. Reduction MDS
print("Calcul de la projection MDS...")
mds = MDS(n_components=2, random_state=42, normalized_stress='auto')
X_mds = mds.fit_transform(X_latent)

# 2. Reduction UMAP
print("Calcul de la projection UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_latent)

# ==========================================
# 5. AFFICHAGE DES PROJECTIONS
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Visualisation MDS
sc1 = ax1.scatter(X_mds[:, 0], X_mds[:, 1], c=labels, cmap='coolwarm', s=15, alpha=0.8)
ax1.set_title("Espace Latent du PINN - Projection MDS 2D")
ax1.set_xlabel("MDS Dim 1")
ax1.set_ylabel("MDS Dim 2")
fig.colorbar(sc1, ax=ax1, label="Temps $t$ (Bleu: Effondrement, Rouge: Rebond)")

# Visualisation UMAP
sc2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='coolwarm', s=15, alpha=0.8)
ax2.set_title("Espace Latent du PINN - Projection UMAP 2D")
ax2.set_xlabel("UMAP Dim 1")
ax2.set_ylabel("UMAP Dim 2")
fig.colorbar(sc2, ax=ax2, label="Temps $t$")

plt.suptitle("Trajectoire de la dynamique quantique dans l'Espace Latent (64D -> 2D)", fontsize=14)
plt.tight_layout()
plt.show()