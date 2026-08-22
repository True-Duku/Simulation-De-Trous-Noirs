import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ==========================================
# 1. PARAMÈTRES PHYSIQUES & MAILLAGE
# ==========================================
M = 1.0           # Masse du trou noir
R_H = 2.0 * M     # Horizon (2M = 2.0)
R0 = 8.0          # Rayon initial de l'étoile

# Grille de coordonnées polaires (r, theta) pour la membrane
r_grid = np.linspace(0.01, 12.0, 150)
theta_grid = np.linspace(0, 2 * np.pi, 60)
R_mesh, THETA_mesh = np.meshgrid(r_grid, theta_grid)
X = R_mesh * np.cos(THETA_mesh)
Y = R_mesh * np.sin(THETA_mesh)

# ==========================================
# 2. CALCUL DE L'EFFONDREMENT (Temps propre tau)
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
                t_eval=np.linspace(0.0, 30.0, 150), 
                events=touche_singularite, method='RK45')

tau_vals = sol.t
R_vals = sol.y[0]

# ==========================================
# 3. PROJECTION DE LA COURBURE z(r)
# ==========================================
# Calcule la profondeur z de la membrane selon le rayon actuel de l'étoile R_star
def calculer_surface_z(R_star):
    Z = np.zeros_like(R_mesh)
    for i in range(R_mesh.shape[0]):
        for j in range(R_mesh.shape[1]):
            r = R_mesh[i, j]
            if r >= R_star:
                # Extérieur de l'étoile : Paraboloïde de Flamm z(r) = 2 * sqrt(2M * (r - 2M))
                # On remplace temporairement la masse effective par la masse sous le rayon R_star
                if r >= R_H:
                    Z[i, j] = -2.0 * np.sqrt(2.0 * M * (r - R_H))
                else:
                    Z[i, j] = -2.0 * np.sqrt(abs(2.0 * M * (r - R_H)))
            else:
                # Intérieur de l'étoile : Raccordement parabolique doux de la matière
                z_edge = -2.0 * np.sqrt(2.0 * M * max(0.0, R_star - R_H))
                # Profil quadratique intérieur pour faire un fond de cuvette continu
                Z[i, j] = z_edge - (1.0 - (r / R_star)**2) * 1.5
    return Z

# ==========================================
# 4. ANIMATION MATPLOTLIB 3D
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Fonction de mise à jour de chaque image
def update(frame):
    ax.clear()
    R_actuel = R_vals[frame]
    tau_actuel = tau_vals[frame]
    
    # Calcul de la nouvelle forme de la membrane
    Z = calculer_surface_z(R_actuel)
    
    # 1. Rendu de la membrane d'espace-temps (Bleu / Cyan)
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.75, linewidth=0.2, edgecolor='k')
    
    # 2. Tracer la frontière de l'étoile sur la membrane (Cercle rouge)
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_star = R_actuel * np.cos(theta_circle)
    y_star = R_actuel * np.sin(theta_circle)
    z_star_val = -2.0 * np.sqrt(2.0 * M * max(0.0, R_actuel - R_H)) if R_actuel >= R_H else -1.5
    z_star = np.full_like(x_star, z_star_val)
    
    couleur_etoile = 'red' if R_actuel > R_H else 'black'
    ax.plot(x_star, y_star, z_star, color=couleur_etoile, linewidth=4, 
            label=f'Surface Étoile (R = {R_actuel:.2f})')
    
    # 3. Tracer l'horizon 2M (Cercle pointillé noir)
    x_h = R_H * np.cos(theta_circle)
    y_h = R_H * np.sin(theta_circle)
    z_h = np.full_like(x_h, 0.0)
    ax.plot(x_h, y_h, z_h, color='black', linestyle='--', linewidth=2, label=f'Horizon (2M = {R_H:.1f})')
    
    # Configuration du repère 3D
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 2)
    ax.set_title(f"Déformation de l'espace 2D (Paraboloïde de Flamm)\nTemps propre τ = {tau_actuel:.2f}", fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Profondeur du puits z')
    ax.legend(loc='upper right')
    
    return surf,

ani = FuncAnimation(fig, update, frames=len(R_vals), interval=40, repeat=True)
plt.tight_layout()
plt.show()