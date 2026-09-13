import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# 1. INITIALISATION DE LA FIGURE ET DES PANNEAUX
# ==========================================
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
ax_thermo = fig.add_subplot(gs[0, 1])
ax_nuc = fig.add_subplot(gs[1, 0])
ax_em = fig.add_subplot(gs[1, 1])

# Domaine temporel de la simulation (unités arbitraires de phase évolutive)
n_frames = 120
t_vals = np.linspace(0, 10.0, n_frames)

# Historiques pour le suivi temporel des panneaux 3 et 4
t_history = []
H_history, He_history, CO_history = [], [], []
B_history, Pdeg_ratio_history = [], []

# Grille spatiale radiale pour les profils thermodynamiques
r_grid = np.linspace(0.01, 10.0, 200)

# ==========================================
# 2. MODÈLE ANALYTIQUE DES PHASES ÉVOLUTIVES
# ==========================================
def calculer_etat_etoile(t):
    """
    Renseigne la physique globale selon 3 phases majeures :
    - Phase 1 (0 <= t < 4) : Séquence Principale -> Géante Rouge (Expansion)
    - Phase 2 (4 <= t < 7) : Branche Asymptotique (AGB) & Nébuleuse Planétaire
    - Phase 3 (7 <= t <= 10): Naine Blanche & Refroidissement
    """
    if t < 4.0:
        # Phase 1 : Expansion du rayon, contraction du cœur
        tau = t / 4.0
        R_star = 1.0 + 6.0 * (tau**2)
        R_core = 0.5 - 0.45 * tau
        R_nebula = 0.0
        T_c = 1.5e7 * (1.0 + 4.0 * tau)
        rho_c = 1e2 * (1.0 + 9.0 * tau)
        H_frac = max(0.0, 0.7 * (1.0 - tau))
        He_frac = 1.0 - H_frac
        CO_frac = 0.0
        B_core = 100.0 * (1.0 + 4.0 * tau)
        Pdeg_ratio = 0.1 + 0.3 * tau  # Pression thermique dominante
        phase_str = "Séquence Principale / Géante Rouge"
        
    elif t < 7.0:
        # Phase 2 : Éjection de l'enveloppe (Nébuleuse) & Flash He
        tau = (t - 4.0) / 3.0
        R_star = 7.0 * (1.0 - 0.95 * tau)
        R_core = 0.05 * (1.0 - 0.8 * tau)
        R_nebula = 1.0 + 8.0 * tau  # Coquille en expansion rapide
        T_c = 7.5e7 * (1.0 + 0.3 * np.sin(np.pi * tau))
        rho_c = 1e3 * 10**(2.5 * tau)
        H_frac = 0.0
        He_frac = max(0.0, 0.3 * (1.0 - tau))
        CO_frac = 1.0 - He_frac
        B_core = 500.0 * 10**(1.5 * tau)
        Pdeg_ratio = 0.4 + 10.0 * tau
        phase_str = "Phase AGB / Éjection Nébuleuse Planétaire"
        
    else:
        # Phase 3 : Naine Blanche finale
        tau = (t - 7.0) / 3.0
        R_star = 0.01  # Rayon ~ taille de la Terre
        R_core = 0.01
        R_nebula = 9.0 + 5.0 * tau  # Nébuleuse se dilue au loin
        T_c = 9.0e7 * np.exp(-1.5 * tau)  # Refroidissement
        rho_c = 1e6
        H_frac, He_frac, CO_frac = 0.0, 0.0, 1.0
        B_core = 25000.0
        Pdeg_ratio = 100.0  # Dégénérescence électronique totale
        phase_str = "Naine Blanche (Cœur C/O Dégénéré)"

    return R_star, R_core, R_nebula, T_c, rho_c, H_frac, He_frac, CO_frac, B_core, Pdeg_ratio, phase_str

# ==========================================
# 3. FONCTION DE MISE À JOUR DE L'ANIMATION
# ==========================================
def update(frame):
    t = t_vals[frame]
    R_s, R_c, R_neb, T_c, rho_c, H_f, He_f, CO_f, B_c, Pdeg_r, phase = calculer_etat_etoile(t)
    
    # Mise à jour des historiques
    t_history.append(t)
    H_history.append(H_f)
    He_history.append(He_f)
    CO_history.append(CO_f)
    B_history.append(B_c)
    Pdeg_ratio_history.append(Pdeg_r)
    
    # Nettoyage des panneaux
    ax_3d.clear()
    ax_thermo.clear()
    ax_nuc.clear()
    ax_em.clear()
    
    # ----------------------------------------------------
    # PANNEAU 1 : STRUCTURE 3D (Surface, Coquille, Cœur)
    # ----------------------------------------------------
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    # Cœur dense (Rouge/Blanc)
    x_c = R_c * np.outer(np.cos(u), np.sin(v))
    y_c = R_c * np.outer(np.sin(u), np.sin(v))
    z_c = R_c * np.outer(np.ones(np.size(u)), np.cos(v))
    ax_3d.plot_surface(x_c, y_c, z_c, color='red' if t < 7 else 'white', alpha=0.9)
    
    # Enveloppe stellaire / Naine blanche
    if R_s > 0.05:
        x_s = R_s * np.outer(np.cos(u), np.sin(v))
        y_s = R_s * np.outer(np.sin(u), np.sin(v))
        z_s = R_s * np.outer(np.ones(np.size(u)), np.cos(v))
        ax_3d.plot_surface(x_s, y_s, z_s, color='orange', alpha=0.15)
    
    # Coquille de la Nébuleuse Planétaire (Phase AGB et après)
    if R_neb > 0:
        x_n = R_neb * np.outer(np.cos(u), np.sin(v))
        y_n = R_neb * np.outer(np.sin(u), np.sin(v))
        z_n = R_neb * np.outer(np.ones(np.size(u)), np.cos(v))
        ax_3d.plot_wireframe(x_n, y_n, z_n, color='lime', alpha=0.5, linewidth=0.7)
        
    ax_3d.set_xlim(-10, 10)
    ax_3d.set_ylim(-10, 10)
    ax_3d.set_zlim(-10, 10)
    ax_3d.set_title(f"1. Geometrie 3D — {phase}\n(t = {t:.1f})")
    ax_3d.set_box_aspect([1, 1, 1])

    # ----------------------------------------------------
    # PANNEAU 2 : THERMODYNAMIQUE (Profils T et Rho)
    # ----------------------------------------------------
    # Profils gaussiens/décroissants dépendant du rayon effectif
    T_profile = T_c * np.exp(-r_grid / (R_s + 0.1))
    rho_profile = rho_c * np.exp(-r_grid / (R_c + 0.05))
    
    ax_thermo.plot(r_grid, T_profile, 'r-', label="Température $T(r)$ [K]")
    ax_thermo.set_ylabel("Température (K)", color='r')
    ax_thermo.tick_params(axis='y', labelcolor='r')
    ax_thermo.set_yscale('log')
    
    ax2_twin = ax_thermo.twinx()
    ax2_twin.plot(r_grid, rho_profile, 'b--', label="Densité $\\rho(r)$ [g/cm³]")
    ax2_twin.set_ylabel("Densité (g/cm³)", color='b')
    ax2_twin.tick_params(axis='y', labelcolor='b')
    ax2_twin.set_yscale('log')
    
    ax_thermo.set_xlabel("Rayon relatif $r$")
    ax_thermo.set_title("2. Thermodynamique : Profils Radiaux")
    ax_thermo.grid(True, linestyle=':', alpha=0.6)

    # ----------------------------------------------------
    # PANNEAU 3 : PHYSIQUE NUCLÉAIRE (Composition du cœur)
    # ----------------------------------------------------
    ax_nuc.plot(t_history, H_history, 'g-', label="Hydrogène ($H$)")
    ax_nuc.plot(t_history, He_history, 'y-', label="Hélium ($He$)")
    ax_nuc.plot(t_history, CO_history, 'm-', label="Carbone/Oxygène ($C/O$)")
    ax_nuc.set_xlim(0, 10)
    ax_nuc.set_ylim(-0.05, 1.05)
    ax_nuc.set_xlabel("Temps évolutif $t$")
    ax_nuc.set_ylabel("Fraction massique au cœur")
    ax_nuc.set_title("3. Nucléaire : Abondances au Cœur")
    ax_nuc.legend(loc='center left')
    ax_nuc.grid(True, linestyle=':', alpha=0.6)

    # ----------------------------------------------------
    # PANNEAU 4 : ÉLECTROMAGNÉTISME & DÉGÉNÉRESCENCE
    # ----------------------------------------------------
    ax_em.plot(t_history, B_history, 'c-', label="Champ $B_{coeur}$ (Gauss)")
    ax_em.set_ylabel("Champ Magnétique $B$ (G)", color='c')
    ax_em.tick_params(axis='y', labelcolor='c')
    ax_em.set_yscale('log')
    
    ax4_twin = ax_em.twinx()
    ax4_twin.plot(t_history, Pdeg_ratio_history, 'm--', label="$P_{deg} / P_{th}$")
    ax4_twin.set_ylabel("Rapport $P_{dégénérescence} / P_{thermique}$", color='m')
    ax4_twin.tick_params(axis='y', labelcolor='m')
    ax4_twin.set_yscale('log')
    
    ax_em.set_xlim(0, 10)
    ax_em.set_xlabel("Temps évolutif $t$")
    ax_em.set_title("4. EM & Pression de Dégénérescence")
    ax_em.grid(True, linestyle=':', alpha=0.6)

# ==========================================
# 4. LANCEMENT DE L'ANIMATION
# ==========================================
ani = FuncAnimation(fig, update, frames=n_frames, interval=80, repeat=True)
plt.show()