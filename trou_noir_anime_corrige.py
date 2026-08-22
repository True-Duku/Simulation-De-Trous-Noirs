import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ==========================================
# 1. PARAMÈTRES PHYSIQUES & CALCUL
# ==========================================
M = 1.0
r_H = 2.0 * M
b = 5.0  # Essaye 5.196 pour le voir tourner longtemps !

r0 = 30.0
phi0 = 0.0
phip0 = b / (r0**2)
rp0 = -np.sqrt(1.0 - (b**2 * (1.0 - (2.0 * M) / r0)) / (r0**2))

def geodesique_schwarzschild(lambda_param, Y):
    r, rp, phi, phip = Y
    if r <= r_H:
        return [0.0, 0.0, 0.0, 0.0]
    d2r = - (M * rp**2) / (r * (r - 2.0 * M)) + (r - 2.0 * M) * (r - 3.0 * M) * phip**2
    d2phi = - (2.0 * rp * phip) / r
    return [rp, d2r, phip, d2phi]

Y0 = [r0, rp0, phi0, phip0]
lambda_span = (0.0, 200.0)
lambda_eval = np.linspace(0.0, 200.0, 2000)

def touche_horizon(lambda_param, Y):
    return Y[0] - (r_H + 0.01)
touche_horizon.terminal = True

sol = solve_ivp(geodesique_schwarzschild, lambda_span, Y0, t_eval=lambda_eval, events=touche_horizon, method='RK45')

# Transformation des données
r_vals = sol.y[0]
phi_vals = sol.y[2]
x_vals = r_vals * np.cos(phi_vals)
y_vals = r_vals * np.sin(phi_vals)

# ==========================================
# 2. ANIMATION MATPLOTLIB
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))

trou_noir = plt.Circle((0, 0), r_H, color='black', label='Horizon des événements')
ax.add_patch(trou_noir)

ligne_trajectoire, = ax.plot([], [], color='blue', linewidth=2, label=f'Photon (b={b})')
point_photon, = ax.plot([], [], color='red', marker='o', markersize=8)

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title("Animation d'une géodésique (Schwarzschild)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
ax.set_aspect('equal')

def init():
    ligne_trajectoire.set_data([], [])
    point_photon.set_data([], [])
    return ligne_trajectoire, point_photon

def update(frame):
    # Sécurité pour ne pas dépasser la taille réelle des tableaux
    idx = min(frame, len(x_vals) - 1)
    
    ligne_trajectoire.set_data(x_vals[:idx], y_vals[:idx])
    point_photon.set_data([x_vals[idx]], [y_vals[idx]])
    return ligne_trajectoire, point_photon

# CORRECTION CLÉ :
# On utilise la taille réelle de x_vals (len(x_vals)) + 20 frames de pause à la fin
total_frames = len(x_vals) + 20 

ani = FuncAnimation(
    fig, 
    update, 
    frames=total_frames, 
    init_func=init, 
    blit=True, 
    interval=30,     # Vitesse (30 ms entre chaque image)
    repeat=True      # Recommence en boucle !
)


# Forcer Spyder à afficher la fenêtre vidéo : %matplotlib qt dans sa console
plt.show()

