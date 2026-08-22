import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ==========================================
# 1. CONSTANTES PHYSIQUES (SI)
# ==========================================
G = 6.6743e-11
c = 299792458.0
M = 2.0e30               # 1 Masse Solaire (~2e30 kg)
Rs = (2 * G * M) / (c**2) # Rayon de Schwarzschild (~2966 m)
R0 = 50000.0             # Rayon initial de l'étoile (50 km)

# Propriétés de la voiture (Acier)
L0 = 4.0                 # Longueur initiale (m)
W0 = 1.8                 # Largeur initiale (m)
densite_acier = 7800.0   # kg/m³
limite_rupture = 400e6   # 400 MPa (Limite élastique)

# ==========================================
# 2. ÉQUATIONS DE CHUTE (Temps propre tau)
# ==========================================
def eq_effondrement(tau, y):
    r = y[0]
    if r <= 1.0:
        return [0.0]
    # dr/dtau pour la surface de l'étoile en chute libre
    dr_dtau = -np.sqrt(2 * G * M * (1.0/r - 1.0/R0))
    return [dr_dtau]

def touche_singularite(tau, y):
    return y[0] - 10.0
touche_singularite.terminal = True

# Résolution temporelle
tau_span = (0.0, 1.0)
tau_eval = np.linspace(0.0, 0.005, 300) # Échelle de temps très courte (fraction de seconde)

sol = solve_ivp(eq_effondrement, tau_span, [R0], t_eval=tau_eval, 
                events=touche_singularite, method='RK45')

tau_vals = sol.t
R_vals = sol.y[0]

# ==========================================
# 3. CALCULS DES DEFORMATIONS SUR LA VOITURE
# ==========================================
contraintes = []
longueurs = []
largeurs = []
rupture_index = None

for i, r in enumerate(R_vals):
    # Gradient de gravité (Marée) = 2GM / r³
    grad_maree = (2 * G * M) / (r**3)
    
    # Contrainte de traction subie sur la longueur de la voiture (Pa)
    sigma = densite_acier * grad_maree * (L0**2) / 8.0
    contraintes.append(sigma / 1e6) # Conversion en MPa
    
    # Déformation relative (Loi de Hooke approximée : epsilon = sigma / E)
    # Module d'Young de l'acier E ~ 200 GPa
    E_acier = 200e9
    elongation = 1.0 + (sigma / E_acier)
    
    L_actuel = L0 * elongation
    W_actuel = W0 / np.sqrt(elongation) # Conservation approximative du volume
    
    longueurs.append(L_actuel)
    largeurs.append(W_actuel)
    
    if sigma >= limite_rupture and rupture_index is None:
        rupture_index = i

# ==========================================
# 4. ANIMATION MATPLOTLIB
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- AXE 1 : Étoile & Voiture ---
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title("Effondrement de l'étoile & Position de la Voiture")

# Horizon
horizon_circle = plt.Circle((0, 0), Rs, color='black', fill=False, linestyle='--', label=f'Horizon Rs ({Rs/1000:.1f} km)')
ax1.add_patch(horizon_circle)

# Étoile
etoile_patch = plt.Circle((0, 0), R0, color='crimson', alpha=0.4, label="Surface de l'Étoile")
ax1.add_patch(etoile_patch)

# Représentation de la voiture (un rectangle schématique à la surface)
voiture_patch = plt.Rectangle((0, 0), W0*200, L0*200, color='blue', angle=0.0)
ax1.add_patch(voiture_patch)

# --- AXE 2 : Telemétrie mécanique ---
line_sigma, = ax2.plot([], [], 'b-', linewidth=2, label="Contrainte subie (MPa)")
ax2.axhline(limite_rupture/1e6, color='red', linestyle='--', label="Limite de rupture Acier (400 MPa)")
ax2.set_xlim(0, max(tau_vals)*1000)
ax2.set_ylim(0, (limite_rupture/1e6) * 1.5)
ax2.set_xlabel("Temps propre τ (ms)")
ax2.set_ylabel("Tension mécanique (MPa)")
ax2.set_title("Forces de marée subies par la structure")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

txt_status = ax1.text(0.02, 0.92, '', transform=ax1.transAxes, fontsize=10, 
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

def update(frame):
    r_actuel = R_vals[frame]
    tau_ms = tau_vals[frame] * 1000
    sigma_actuel = contraintes[frame]
    L_actuel = longueurs[frame]
    W_actuel = largeurs[frame]
    
    # Mise à jour rayon étoile
    etoile_patch.set_radius(r_actuel)
    
    # Positionnement de la voiture au sommet de l'étoile
    # On exagère la taille du rectangle pour qu'il reste visible à l'échelle de l'étoile
    zoom_factor = 300 
    w_visu = W_actuel * zoom_factor
    l_visu = L_actuel * zoom_factor
    voiture_patch.set_xy((-w_visu/2, r_actuel))
    voiture_patch.set_width(w_visu)
    voiture_patch.set_height(l_visu)
    
    # Couleur de la voiture selon la rupture
    if rupture_index is not None and frame >= rupture_index:
        voiture_patch.set_color('darkorange') # Disloquée
        status_txt = f"RAYON : {r_actuel/1000:.1f} km\nVOITURE : DISLOQUÉE / DISRUPTÉE !\nContrainte : {sigma_actuel:.0f} MPa"
    else:
        voiture_patch.set_color('blue')
        status_txt = f"RAYON : {r_actuel/1000:.1f} km\nVOITURE : Élastique (Intacte)\nContrainte : {sigma_actuel:.2f} MPa"
        
    txt_status.set_text(status_txt)
    
    # Courbe de contrainte
    line_sigma.set_data(tau_vals[:frame]*1000, contraintes[:frame])
    
    # Redimensionnement auto du graphique
    lim = max(r_actuel * 1.2, Rs * 2)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-Rs, lim)
    
    return etoile_patch, voiture_patch, line_sigma, txt_status

ani = FuncAnimation(fig, update, frames=len(R_vals), interval=30, repeat=True)
plt.tight_layout()
plt.show()