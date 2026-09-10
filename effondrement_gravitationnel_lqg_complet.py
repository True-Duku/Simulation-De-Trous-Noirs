import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# 1. DISCRÉTISATION DU DOMAINE SPATIO-TEMPOREL (3D)
# ==========================================
N = 32                     # 32x32x32 points
L = 10.0                   # Taille du domaine (-L/2 à L/2)
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
z = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-5  # Distance au centre

# ==========================================
# 2. PARAMÈTRES ET ÉQUATIONS EFFECTIVES LQG
# ==========================================
G = 1.0                    # Constante gravitationnelle
M0 = 5.0                   # Masse totale
R_initial = 4.0            # Rayon initial
R_horizon = 2 * G * M0     # Horizon théorique (2M)

# Paramètres de gravité quantique
rho_P = 1.5                # Densité maximale de Planck (seuil de saturation)
R_min = 0.8                # Rayon minimal au moment du rebond quantique

# Simulation étendue pour observer l'effondrement ET le rebond
n_frames = 120
t_vals = np.linspace(0, 3.0, n_frames)
t_bounce = 1.2             # Instant du rebond quantique (t_bounce)

def generer_champs_lqg_3d(t):
    """
    Simule la dynamique quantique LQG :
    - Trajectoire hyperbolique du rayon R(t) avec rebond
    - Saturation de la densité centrale rho(t) à la densité de Planck rho_P
    - Potentiel gravitationnel régularisé au cœur (sans singularité)
    """
    # 1. Évolution du rayon avec Rebond Quantique (Quantum Bounce)
    # R(t) décroît jusqu'à R_min à t_bounce, puis réaugmente (trou blanc)
    R_t = R_min + (R_initial - R_min) * ((t - t_bounce) / t_bounce)**2
    
    # 2. Champ de Densité rho(x,y,z,t) avec correction effectives LQG
    # Calcul de la densité classique idéale (qui tendrait vers l'infini)
    rho_classique_centre = M0 / ((np.pi**1.5) * (R_t**3))
    
    # Correction LQG : rho_eff = rho / (1 + rho/rho_P) -> saturation stricte à rho_P
    rho_centre = rho_classique_centre / (1.0 + rho_classique_centre / rho_P)
    rho = rho_centre * np.exp(-(R**2) / (R_t**2))
    
    # 3. Potentiel Gravitationnel Régularisé V(r,t)
    # Suppression du terme -G*M/0 grâce au cœur étalé de rayon R_min
    V = - (G * M0) / np.sqrt(R**2 + R_t**2 + (1.0 / rho_P))
    
    return R_t, rho, V

# ==========================================
# 3. VISUALISATION : COUPE MID-PLANE & VOLUME 3D
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
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

# Sous-graphe 2 : Vue 3D volumétrique de la matière
ax2 = fig.add_subplot(122, projection='3d')

def update(frame):
    ax2.clear()
    t = t_vals[frame]
    R_t, rho, V = generer_champs_lqg_3d(t)
    
    # Indicateur de phase (Effondrement vs Rebond)
    phase = "Effondrement" if t < t_bounce else "Rebond Quantique (Trou Blanc)"
    
    # 1. Mise à jour de la coupe 2D
    rho_2d = rho[:, :, mid_z]
    im.set_data(rho_2d.T)
    im.set_clim(vmin=0, vmax=rho_P)  # Échelle fixée à la densité de Planck
    ax1.set_title(f"Densité Z=0 (t = {t:.2f}) - {phase}\nDensité max = {np.max(rho):.2f} / $\\rho_P = {rho_P}$")
    ax1.legend(loc='upper right')
    
    # 2. Rendu 3D : Nuage de points
    seuil = np.max(rho) * 0.15
    mask = rho > seuil
    
    X_sub = X[mask][::2]
    Y_sub = Y[mask][::2]
    Z_sub = Z[mask][::2]
    rho_sub = rho[mask][::2]
    
    ax2.scatter(X_sub, Y_sub, Z_sub, c=rho_sub, cmap='inferno', 
                s=15, alpha=0.5, edgecolor='none', vmin=0, vmax=rho_P)
    
    # Sphère représentant l'horizon des événements
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
    ax2.set_title(f"Volume 3D (LQG)\nRayon R(t) = {R_t:.2f}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    return im, ax2

ani = FuncAnimation(fig, update, frames=n_frames, interval=50, repeat=True)
plt.tight_layout()
plt.show()