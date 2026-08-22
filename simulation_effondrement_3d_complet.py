import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# 1. DISCRÉTISATION DU DOMAINE SPATIO-TEMPOREL (3D)
# ==========================================
# Grille 3D réduite pour préserver la mémoire RAM
N = 32                     # 32x32x32 points = ~32 768 cellules
L = 10.0                   # Taille de la boîte (-L/2 à L/2)
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
z = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-5  # Distance au centre (évite div par 0)

# ==========================================
# 2. ÉTAT INITIAL & PHYSIQUE DE L'EFFONDREMENT
# ==========================================
G = 1.0                    # Constante gravitationnelle (unités géométriques)
M0 = 5.0                   # Masse totale de l'étoile
R_initial = 4.0            # Rayon initial de l'étoile
R_horizon = 2 * G * M0 / (1.0**2)  # Horizon théorique (2M)

n_frames = 80
t_vals = np.linspace(0, 1.8, n_frames)

def generer_champs_3d(t):
    """
    Simule la dynamique du fluide 3D en effondrement :
    - Contraction du rayon R(t)
    - Augmentation dramatique de la densité centrale rho(t)
    - Creusement du potentiel gravitationnel V(r,t)
    """
    # Évolution du rayon de la surface de l'étoile
    R_t = max(R_initial * (1 - (t / 2.0)**2), 0.1)
    
    # 1. Champ de Densité rho(x,y,z,t)
    # Modèle de profil gaussien dont la variance rétrécit (concentration de masse)
    rho_centre = M0 / ((np.pi**1.5) * (R_t**3))
    rho = rho_centre * np.exp(-(R**2) / (R_t**2))
    
    # 2. Potentiel Gravitationnel / Puits de Courbure V(r,t)
    # Résolution simplifiée de l'équation de Poisson 3D: nabla^2 V = 4*pi*G*rho
    V = - (G * M0) / np.sqrt(R**2 + R_t**2)
    
    return R_t, rho, V

# ==========================================
# 3. VISUALISATION : COUPE MID-PLANE & ISOSURFACE
# ==========================================
fig = plt.figure(figsize=(14, 6))

# Sous-graphe 1 : Coupe 2D de la densité (Plan équatorial Z=0)
ax1 = fig.add_subplot(121)
mid_z = N // 2
im = ax1.imshow(np.zeros((N, N)), extent=[-L/2, L/2, -L/2, L/2], 
                cmap='inferno', origin='lower')
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label('Densité de matière $\\rho(x,y,z)$')
circle_h = plt.Circle((0, 0), R_horizon, color='cyan', fill=False, linestyle='--', label=f'Horizon $2M = {R_horizon:.1f}$')
ax1.add_patch(circle_h)
ax1.set_title("Coupe 2D du Champ de Densité (Z=0)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend(loc='upper right')

# Sous-graphe 2 : Vue 3D volumétrique de la densité
ax2 = fig.add_subplot(122, projection='3d')

def update(frame):
    ax2.clear()
    t = t_vals[frame]
    R_t, rho, V = generer_champs_3d(t)
    
    # 1. Mise à jour de la coupe 2D
    rho_2d = rho[:, :, mid_z]
    im.set_data(rho_2d.T)
    im.set_clim(vmin=0, vmax=np.max(rho_2d))
    ax1.set_title(f"Densité Z=0 (t = {t:.2f})\nDensité max = {np.max(rho):.2f}")
    
    # 2. Rendu 3D : Nuage de points représentant le volume de matière
    # Seuil pour ne garder que la matière dense
    seuil = np.max(rho) * 0.15
    mask = rho > seuil
    
    # Échantillonnage pour alléger le rendu graphique
    X_sub = X[mask][::2]
    Y_sub = Y[mask][::2]
    Z_sub = Z[mask][::2]
    rho_sub = rho[mask][::2]
    
    p3d = ax2.scatter(X_sub, Y_sub, Z_sub, c=rho_sub, cmap='inferno', 
                      s=15, alpha=0.5, edgecolor='none')
    
    # Dessin de la sphère de l'horizon 3D
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_h = R_horizon * np.outer(np.cos(u), np.sin(v))
    y_h = R_horizon * np.outer(np.sin(u), np.sin(v))
    z_h = R_horizon * np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x_h, y_h, z_h, color='cyan', alpha=0.15, edgecolor='cyan', linewidth=0.3)
    
    ax2.set_xlim(-L/2, L/2)
    ax2.set_ylim(-L/2, L/2)
    ax2.set_zlim(-L/2, L/2)
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_title(f"Volume 3D de l'Étoile\nRayon R(t) = {R_t:.2f}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    return im, ax2

ani = FuncAnimation(fig, update, frames=n_frames, interval=50, repeat=True)
plt.tight_layout()
plt.show()