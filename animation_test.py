import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import MDS
import umap

# --------------------------------------------------
# 1. SIMULATION OU CHARGEMENT DE DONNÉES LATENTES
# --------------------------------------------------
np.random.seed(42)
N_pts = 100        # Nombre de points par pas de temps
N_frames = 50      # Nombre de pas de temps (frames)
D_latent = 64      # Dimension de la couche cachée du PINN

# Simulation d'un déplacement continu dans l'espace latent 64D
X_latent_list = []
time_steps = np.linspace(0, 1.6, N_frames)

for t in time_steps:
    # Trajectoire circulaire dans l'espace 64D (modélisant effondrement + rebond)
    center = np.zeros(D_latent)
    center[0] = np.cos(np.pi * t / 0.8)
    center[1] = np.sin(np.pi * t / 0.8)
    
    # Nuage de points autour du centre
    pts = center + np.random.normal(scale=0.15, size=(N_pts, D_latent))
    X_latent_list.append(pts)

# Concaténation de tout l'historique : (N_frames * N_pts, 64)
X_latent_concat = np.vstack(X_latent_list)

# --------------------------------------------------
# 2. PROJECTION GLOBALE (MDS ou UMAP)
# --------------------------------------------------
print("Calcul du repère 2D global avec UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
X_2d_global = reducer.fit_transform(X_latent_concat)

# Découpage du résultat par frame
X_2d_by_frame = np.split(X_2d_global, N_frames)

# --------------------------------------------------
# 3. CRÉATION DE L'ANIMATION AVEC MATPLOTLIB
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Limites fixes basées sur la projection globale
x_min, x_max = X_2d_global[:, 0].min() - 1, X_2d_global[:, 0].max() + 1
y_min, y_max = X_2d_global[:, 1].min() - 1, X_2d_global[:, 1].max() + 1

# Scatter plot initial (première frame)
scat = ax.scatter(X_2d_by_frame[0][:, 0], X_2d_by_frame[0][:, 1], 
                  c='royalblue', alpha=0.7, edgecolors='k', s=30)

# Trajectoire du centre de masse (effet de traînée)
trail, = ax.plot([], [], 'r--', lw=2, label="Trajectoire moyenne")
centers_2d = [frame.mean(axis=0) for frame in X_2d_by_frame]

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title(f"Évolution de l'Espace Latent UMAP à t = {time_steps[0]:.2f}")
ax.set_xlabel("UMAP Dim 1")
ax.set_ylabel("UMAP Dim 2")
ax.grid(True)
ax.legend()

# Fonction de mise à jour à chaque frame
def update(frame_idx):
    current_data = X_2d_by_frame[frame_idx]
    current_t = time_steps[frame_idx]
    
    # Mise à jour des positions des points
    scat.set_offsets(current_data)
    
    # Couleur dynamique selon la phase (bleu: effondrement, rouge: rebond/expansion)
    color = 'royalblue' if current_t < 0.8 else 'crimson'
    scat.set_color(color)
    
    # Mise à jour de la traînée du centre de masse
    curr_centers = np.array(centers_2d[:frame_idx + 1])
    trail.set_data(curr_centers[:, 0], curr_centers[:, 1])
    
    # Mise à jour du titre
    phase = "Effondrement" if current_t < 0.8 else "Rebond / Expansion"
    ax.set_title(f"Espace Latent UMAP à t = {current_t:.2f} ({phase})")
    
    return scat, trail

ani = animation.FuncAnimation(fig, update, frames=N_frames, interval=80, blit=False)

# Sauvegarde en GIF
ani.save("animation_latent_umap.gif", writer="pillow", fps=12)
plt.show()