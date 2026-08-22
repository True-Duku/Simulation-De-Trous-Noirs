import matplotlib


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ==========================================
# 1. PARAMÈTRES PHYSIQUES
# ==========================================
M = 1.0           # Masse du trou noir / de l'étoile
R_H = 2.0 * M     # Horizon des événements (Schwarzschild)
R0 = 10.0         # Rayon initial de l'étoile (doit être > R_H)

# ==========================================
# 2. RÉSOLUTION DES ÉQUATIONS (Temps Propre tau)
# ==========================================
# Équation de la chute libre de la surface : dR/dtau
def effondrement(tau, R):
    r = R[0]
    if r <= 0.001:  # arrêt juste avant la singularité
        return [0.0]
    # Équation d'Oppenheimer-Snyder (chute libre depuis R0)
    dR_dtau = -np.sqrt((2.0 * M / r) - (2.0 * M / R0))
    return [dR_dtau]

tau_span = (0.0, 100.0)
tau_eval = np.linspace(0.0, 100.0, 1000)

# Arrêt si la surface touche la singularité R = 0
def touche_singularite(tau, R):
    return R[0] - 0.01
touche_singularite.terminal = True

sol = solve_ivp(effondrement, tau_span, [R0], t_eval=tau_eval, events=touche_singularite, method='RK45')

tau_vals = sol.t
R_vals = sol.y[0]

# ==========================================
# 3. CONVERSION EN TEMPS COORDONNÉE (t - Observateur)
# ==========================================
# On calcule dt/dtau pour chaque point afin d'obtenir le temps t perçu de loin
# dt/dtau = sqrt(1 - 2M/R0) / (1 - 2M/R)
t_vals = np.zeros_like(tau_vals)
for i in range(1, len(tau_vals)):
    r = R_vals[i-1]
    if r > R_H + 0.001:
        dt_dtau = np.sqrt(1.0 - (2.0 * M / R0)) / (1.0 - (2.0 * M / r))
        dtau = tau_vals[i] - tau_vals[i-1]
        t_vals[i] = t_vals[i-1] + dt_dtau * dtau
    else:
        # Quand R approche R_H, le temps t tend vers l'infini
        t_vals[i:] = np.inf
        break

# ==========================================
# 4. ANIMATION MATPLOTLIB
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- GRAPHE DE GAUCHE : L'Animation Visual-Spatial ---
# Horizon futur (ligne pointillée noire)
horizon_cercle = plt.Circle((0, 0), R_H, color='black', fill=False, linestyle='--', linewidth=2, label='Futur Horizon (2M)')
ax1.add_patch(horizon_cercle)

# L'étoile en effondrement (disque rouge)
etoile_patch = plt.Circle((0, 0), R0, color='crimson', alpha=0.7, label='Matière de l\'étoile')
ax1.add_patch(etoile_patch)

ax1.set_xlim(-R0*1.1, R0*1.1)
ax1.set_ylim(-R0*1.1, R0*1.1)
ax1.set_aspect('equal')
ax1.set_title("Effondrement de l'Étoile (Espace)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Textes d'information sur le graphique
texte_info = ax1.text(0.02, 0.02, '', transform=ax1.transAxes, fontsize=10, 
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# --- GRAPHE DE DROITE : Courbe Rayon vs Temps ---
line_tau, = ax2.plot([], [], 'r-', label='Temps propre tau (Sur l\'étoile)')
line_t, = ax2.plot([], [], 'b--', label='Temps t (Observateur lointain)')
ax2.axhline(R_H, color='black', linestyle=':', label='Horizon (2M)')

ax2.set_xlim(0, max(tau_vals))
ax2.set_ylim(0, R0 * 1.1)
ax2.set_xlabel('Temps')
ax2.set_ylabel('Rayon R')
ax2.set_title('Trajectoire R(temps)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# --- FONCTION DE MISE À JOUR DE L'ANIMATION ---
def update(frame):
    r_actuel = R_vals[frame]
    tau_actuel = tau_vals[frame]
    t_actuel = t_vals[frame]
    
    # 1. Mise à jour du disque de l'étoile
    etoile_patch.set_radius(r_actuel)
    
    # Si la surface franchit l'horizon, on change la couleur pour marquer le coup !
    if r_actuel <= R_H:
        etoile_patch.set_color('black')
        etoile_patch.set_alpha(0.9)
    else:
        etoile_patch.set_color('crimson')
        etoile_patch.set_alpha(0.7)
        
    # 2. Mise à jour des textes
    t_str = f"{t_actuel:.2f}" if not np.isinf(t_actuel) else "Infini (Grounded)"
    texte_info.set_text(f"Rayon R = {r_actuel:.2f}\nTemps propre τ = {tau_actuel:.2f}\nTemps observateur t = {t_str}")
    
    # 3. Mise à jour du graphique des courbes
    line_tau.set_data(tau_vals[:frame], R_vals[:frame])
    # Pour t, on filtre les valeurs infinies pour éviter les erreurs d'affichage
    valid_t = ~np.isinf(t_vals[:frame])
    line_t.set_data(t_vals[:frame][valid_t], R_vals[:frame][valid_t])
    
    return etoile_patch, texte_info, line_tau, line_t

ani = FuncAnimation(fig, update, frames=len(R_vals), interval=30, repeat=True)
plt.tight_layout()
plt.show()