import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, Isomap
import umap

# ==========================================
# 1. GÉNÉRATION DU DATASET (3D -> Haute Dimension)
# ==========================================
N = 32                     # Grille 32x32x32
L = 10.0
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
z = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-5

G, M0, R_initial = 1.0, 5.0, 4.0
n_frames = 80
t_vals = np.linspace(0, 1.8, n_frames)

dataset_haute_dim = []
rayons_t = []

for t in t_vals:
    R_t = max(R_initial * (1 - (t / 2.0)**2), 0.1)
    rho_centre = M0 / ((np.pi**1.5) * (R_t**3))
    rho = rho_centre * np.exp(-(R**2) / (R_t**2))
    
    # Aplatissement de la grille 3D en un vecteur 1D de 32 768 variables
    dataset_haute_dim.append(rho.flatten())
    rayons_t.append(R_t)

X_data = np.array(dataset_haute_dim)  # Forme : (80, 32768)

# ==========================================
# 2. RÉDUCTION DE DIMENSION (MANIFOLD LEARNING)
# ==========================================
print("Calcul de MDS...")
mds = MDS(n_components=2, random_state=42, normalized_stress='auto')
X_mds = mds.fit_transform(X_data)

print("Calcul d'Isomap...")
isomap = Isomap(n_components=2, n_neighbors=10)
X_isomap = isomap.fit_transform(X_data)

print("Calcul d'UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_data)

# ==========================================
# 3. VISUALISATION DES ESPACES LATENTS
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

algos = [('MDS (Distance)', X_mds), 
         ('Isomap (Géodésique)', X_isomap), 
         ('UMAP (Topologie)', X_umap)]

for ax, (nom, X_2d) in zip(axes, algos):
    # La couleur des points indique le temps t de l'effondrement
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=t_vals, cmap='plasma', s=40, edgecolor='k', linewidth=0.5)
    # Ligne reliant la trajectoire temporelle
    ax.plot(X_2d[:, 0], X_2d[:, 1], 'k--', alpha=0.4)
    
    ax.set_title(f"Projection {nom}")
    ax.set_xlabel("Dimension Latente 1")
    ax.set_ylabel("Dimension Latente 2")
    ax.grid(True, alpha=0.3)

cbar = fig.colorbar(sc, ax=axes.ravel().tolist())
cbar.set_label('Temps de l\'effondrement $t$')

plt.suptitle("Réduction de Dimension 3D -> 2D de l'Équation d'Effondrement", fontsize=14)
plt.tight_layout()
plt.show()