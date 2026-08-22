import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ==========================================
# 1. PARAMÈTRES PHYSIQUES
# ==========================================
M = 1.0           # Masse du trou noir
R_H = 2.0 * M     # Horizon des événements (2M)
R0 = 6.0          # Rayon initial de l'étoile

# ==========================================
# 2. GÉNÉRATION DES PARTICULES DE L'ÉTOILE (Volumétrique 3D)
# ==========================================
np.random.seed(42)  # Pour la reproductibilité
N_particles = 1500  # Nombre de points pour former le volume

# Distribution uniforme à l'intérieur d'une sphère de rayon R0
r_pos = R0 * (np.random.rand(N_particles) ** (1/3))
theta_pos = np.arccos(1 - 2 * np.random.rand(N_particles))
phi_pos = 2 * np.pi * np.random.rand(N_particles)

# Coordonnées cartésiennes initiales (x0, y0, z0)
X0 = r_pos * np.sin(theta_pos) * np.cos(phi_pos)
Y0 = r_pos * np.sin(theta_pos) * np.sin(phi_pos)
Z0 = r_pos * np.cos(theta_pos)

# ==========================================
# 3. RÉSOLUTION DE L'EFFONDREMENT (Oppenheimer-Snyder)
# ==========================================
def effondrement(tau, R):
    r = R[0]
    if r <= 0.001:
        return [0.0]
    return [-np.sqrt((2.0 * M / r) - (2.0 * M / R0))]

def touche_singularite(tau, R):
    return R[0] - 0.01
touche_singularite.terminal = True

sol = solve_ivp(effondrement, (0.0, 50.0), [R0], 
                t_eval=np.linspace(0.0, 20.0, 120), 
                events=touche_singularite, method='RK45')

tau_vals = sol.t
R_vals = sol.y[0]

# ==========================================
# 4. MAILLAGE DE L'HORIZON (Sphère 3D Fixe)
# ==========================================
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
X_horizon = R_H * np.outer(np.cos(u), np.sin(v))
Y_horizon = R_H * np.outer(np.sin(u), np.sin(v))
Z_horizon = R_H * np.outer(np.ones(np.size(u)), np.cos(v))

# ==========================================
# 5. ANIMATION MATPLOTLIB 3D
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    R_actuel = R_vals[frame]
    tau_actuel = tau_vals[frame]
    
    # 1. Facteur d'échelle pour rétrécir le volume
    scale = R_actuel / R0
    X_actuel = X0 * scale
    Y_actuel = Y0 * scale
    Z_actuel = Z0 * scale
    
    # 2. Couleur des particules selon le passage sous l'horizon
    if R_actuel > R_H:
        couleur_matiere = 'crimson'
        alpha_val = 0.6
    else:
        couleur_matiere = 'darkred'
        alpha_val = 0.3
        
    # 3. Affichage du volume de matière de l'étoile
    ax.scatter(X_actuel, Y_actuel, Z_actuel, color=couleur_matiere, 
               s=12, alpha=alpha_val, label=f'Matière de l\'étoile (R = {R_actuel:.2f})')
    
    # 4. Affichage de la sphère de l'horizon (2M)
    ax.plot_surface(X_horizon, Y_horizon, Z_horizon, color='black', 
                    alpha=0.3, edgecolor='dimgray', linewidth=0.3)
    
    # 5. Habillage du repère 3D
    lim = R0 * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_title(f"Effondrement 3D de l'Étoile (Volume)\nTemps propre τ = {tau_actuel:.2f}", fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper right')
    
    return ax,

ani = FuncAnimation(fig, update, frames=len(R_vals), interval=40, repeat=True)
plt.tight_layout()
plt.show()