import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. PARAMÈTRES PHYSIQUES & INITIALISATION
# ==========================================
M = 1.0
r_H = 2.0 * M  # Rayon de Schwarzschild
b = 5.0        # Paramètre d'impact du photon

r0 = 30.0      # Position initiale (loin du trou noir)
phi0 = 0.0     # Angle initial

# Vitesse initiale pour phi' (d_phi/d_lambda)
phip0 = b / (r0**2)

# Vitesse initiale pour r' (d_r/d_lambda) dérivée de ds^2 = 0
# Le signe négatif indique que le photon se dirige vers le trou noir
rp0 = -np.sqrt(1.0 - (b**2 * (1.0 - (2.0 * M) / r0)) / (r0**2))

# ==========================================
# 2. DÉFINITION DU SYSTÈME DIFFÉRENTIEL
# ==========================================
def geodesique_schwarzschild(lambda_param, Y):
    """
    Définit le système d'équations différentielles pour la géodésique.
    Y[0] = r, Y[1] = r', Y[2] = phi, Y[3] = phi'
    """
    r, rp, phi, phip = Y
    
    # Sécurité pour éviter la division par zéro ou des rayons négatifs
    if r <= r_H:
        return [0.0, 0.0, 0.0, 0.0]
        
    # Équations du mouvement (dérivées secondes)
    # r'' = ...
    d2r = - (M * rp**2) / (r * (r - 2.0 * M)) + (r - 2.0 * M) * (r - 3.0 * M) * phip**2
    # phi'' = ...
    d2phi = - (2.0 * rp * phip) / r
    
    return [rp, d2r, phip, d2phi]

# ==========================================
# 3. CONSTRUIRE LE PIPELINE DE RÉSOLUTION (SOLVER)
# ==========================================
# Conditions initiales regroupées dans un vecteur
Y0 = [r0, rp0, phi0, phip0]
lambda_span = (0.0, 500.0)  # Équivalent de {λ, 0, 500}
lambda_eval = np.linspace(0.0, 500.0, 10000) # Pour un tracé fluide

# Événement de sécurité : arrêter l'intégration si le photon touche l'horizon
def touche_horizon(lambda_param, Y):
    return Y[0] - (r_H + 0.01)
touche_horizon.terminal = True  # Arrête la simulation si cet événement passe à 0

# Résolution numérique (solve_ivp remplace NDSolve)
sol = solve_ivp(
    geodesique_schwarzschild, 
    lambda_span, 
    Y0, 
    t_eval=lambda_eval, 
    events=touche_horizon,
    method='RK45', # Runge-Kutta d'ordre 5(4)
    rtol=1e-8, 
    atol=1e-10
)

# ==========================================
# 4. TRANSFORMATION DES DONNÉES & VISUALISATION
# ==========================================
# Extraction des résultats (r et phi)
r_vals = sol.y[0]
phi_vals = sol.y[2]

# Conversion des coordonnées polaires en cartésiennes
x_vals = r_vals * np.cos(phi_vals)
y_vals = r_vals * np.sin(phi_vals)

# Création du graphique avec Matplotlib
fig, ax = plt.subplots(figsize=(8, 8))

# Dessin de l'horizon des événements (le trou noir)
trou_noir = plt.Circle((0, 0), r_H, color='black', label='Horizon des événements')
ax.add_patch(trou_noir)

# Dessin de la trajectoire du photon
ax.plot(x_vals, y_vals, color='blue', linewidth=2, label=f'Trajectoire du photon (b={b})')

# Configuration de la scène
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title("Simulation d'une géodésique autour d'un trou noir de Schwarzschild")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
ax.set_aspect('equal')

plt.show()