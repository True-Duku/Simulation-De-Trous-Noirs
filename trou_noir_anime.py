import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ==========================================
# 1. PARAMÈTRES PHYSIQUES & CALCUL (Idem)
# ==========================================
M = 1.0
r_H = 2.0 * M
b = 5.0  # Tu pourras tester 5.196 ou 6.5 ici aussi !

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
lambda_eval = np.linspace(0.0, 200.0, 2000) # 2000 points pour une animation fluide

def touche_horizon(lambda_param, Y):
    return Y[0] - (r_H + 0.01)
touche_horizon.terminal = True

sol = solve_ivp(geodesique_schwarzschild, lambda_span, Y0, t_eval=lambda_eval, events=touche_horizon, method='RK45')

# Transformation des données calculées
r_vals = sol.y[0]
phi_vals = sol.y[2]
x_vals = r_vals * np.cos(phi_vals)
y_vals = r_vals * np.sin(phi_vals)

# ==========================================
# 2. CONFIGURATION DE L'ANIMATION MATPLOTLIB
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))

# Éléments statiques (le trou noir)
trou_noir = plt.Circle((0, 0), r_H, color='black', label='Horizon des événements')
ax.add_patch(trou_noir)

# Éléments dynamiques (vides au départ, mis à jour par l'animation)
ligne_trajectoire, = ax.plot([], [], color='blue', linewidth=2, label=f'Photon (b={b})')
point_photon, = ax.plot([], [], color='red', marker='o', markersize=8) # Le photon brillant

# Habillage du graphique
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title("Animation d'une géodésique (Schwarzschild)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
ax.set_aspect('equal')

# Fonction d'initialisation de l'animation
def init():
    ligne_trajectoire.set_data([], [])
    point_photon.set_data([], [])
    return ligne_trajectoire, point_photon

# Fonction appelée à chaque frame (mise à jour des données)
def update(frame):
    # On affiche la trajectoire du début jusqu'à la frame actuelle
    ligne_trajectoire.set_data(x_vals[:frame], y_vals[:frame])
    # On place le point rouge à la position exacte de la frame actuelle
    point_photon.set_data([x_vals[frame]], [y_vals[frame]])
    return ligne_trajectoire, point_photon

# Lancement de l'animation
# frames=len(x_vals) détermine le nombre total d'images
# interval=15 détermine le temps en millisecondes entre chaque image
ani = FuncAnimation(fig, update, frames=len(x_vals), init_func=init, blit=True, interval=15, repeat=False)

plt.show()