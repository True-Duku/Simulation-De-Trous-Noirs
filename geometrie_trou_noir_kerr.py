import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==========================================
# 1. ÉQUATIONS DE LA GÉOMÉTRIE DE KERR
# ==========================================
# Coordonnées de Boyer-Lindquist -> Cartésiennes
# x = sqrt(r^2 + a^2) * sin(theta) * cos(phi)
# y = sqrt(r^2 + a^2) * sin(theta) * sin(phi)
# z = r * cos(theta)

def calculer_geometrie_kerr(M, a):
    # Sécurité pour ne pas dépasser la masse extrémale a <= M
    a = min(a, M * 0.9999)
    
    # Horizons (r+ et r-)
    r_plus = M + np.sqrt(M**2 - a**2)
    r_moins = M - np.sqrt(M**2 - a**2)
    
    # Maillage theta et phi
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Limite statique / Ergosurface (r_E)
    r_E = M + np.sqrt(np.maximum(0, M**2 - a**2 * np.cos(THETA)**2))
    
    # Calcul des surfaces 3D
    # 1. Ergosurface
    X_E = np.sqrt(r_E**2 + a**2) * np.sin(THETA) * np.cos(PHI)
    Y_E = np.sqrt(r_E**2 + a**2) * np.sin(THETA) * np.sin(PHI)
    Z_E = r_E * np.cos(THETA)
    
    # 2. Horizon Externe r+
    X_plus = np.sqrt(r_plus**2 + a**2) * np.sin(THETA) * np.cos(PHI)
    Y_plus = np.sqrt(r_plus**2 + a**2) * np.sin(THETA) * np.sin(PHI)
    Z_plus = r_plus * np.cos(THETA)
    
    # 3. Horizon Interne r-
    X_moins = np.sqrt(r_moins**2 + a**2) * np.sin(THETA) * np.cos(PHI)
    Y_moins = np.sqrt(r_moins**2 + a**2) * np.sin(THETA) * np.sin(PHI)
    Z_moins = r_moins * np.cos(THETA)
    
    return (r_plus, r_moins, theta, 
            (X_E, Y_E, Z_E), (X_plus, Y_plus, Z_plus), (X_moins, Y_moins, Z_moins))

# ==========================================
# 2. CONFIGURATION DE LA FIGURE MATPLOTLIB
# ==========================================
M = 1.0
a_init = 0.85  # Spin initial (85% du maximum)

fig = plt.figure(figsize=(15, 7))

# Souse-graphe 1 : Coupe 2D (Plan x-z)
ax2d = fig.add_subplot(121)

# Sous-graphe 2 : Vue 3D
ax3d = fig.add_subplot(122, projection='3d')

def tracer_geometrie(a_val):
    ax2d.clear()
    ax3d.clear()
    
    r_plus, r_moins, theta, surf_E, surf_plus, surf_moins = calculer_geometrie_kerr(M, a_val)
    
    # --- RENDU 2D (Plan Méridien X-Z) ---
    r_E_2d = M + np.sqrt(np.maximum(0, M**2 - a_val**2 * np.cos(theta)**2))
    
    # Coordonnées pour la coupe 2D
    x_E = np.sqrt(r_E_2d**2 + a_val**2) * np.sin(theta)
    z_E = r_E_2d * np.cos(theta)
    
    x_plus = np.sqrt(r_plus**2 + a_val**2) * np.sin(theta)
    z_plus = r_plus * np.cos(theta)
    
    x_moins = np.sqrt(r_moins**2 + a_val**2) * np.sin(theta)
    z_moins = r_moins * np.cos(theta)
    
    # Remplissage de l'ergosphère
    ax2d.fill_betweenx(z_E, -x_E, x_E, color='gold', alpha=0.3, label='Ergosphère')
    ax2d.fill_betweenx(z_plus, -x_plus, x_plus, color='black', alpha=0.8, label=f'Horizon Externe r+ ({r_plus:.2f})')
    ax2d.fill_betweenx(z_moins, -x_moins, x_moins, color='indigo', alpha=0.9, label=f'Horizon Interne r- ({r_moins:.2f})')
    
    # Contour de la limite statique
    ax2d.plot(x_E, z_E, 'orange', linewidth=2, linestyle='--', label='Limite Statique')
    ax2d.plot(-x_E, z_E, 'orange', linewidth=2, linestyle='--')
    
    # Singularité en anneau à z=0, x = ±a
    ax2d.scatter([a_val, -a_val], [0, 0], color='crimson', s=40, zorder=5, label=f'Singularité anneau (r=0, a={a_val:.2f})')
    
    ax2d.set_xlim(-2.5*M, 2.5*M)
    ax2d.set_ylim(-2.5*M, 2.5*M)
    ax2d.set_aspect('equal')
    ax2d.set_xlabel('Axe équatorial X')
    ax2d.set_ylabel('Axe de rotation Z')
    ax2d.set_title(f"Coupe 2D (Plan X-Z)\nSpin a = {a_val:.2f} M")
    ax2d.grid(True, alpha=0.3)
    ax2d.legend(loc='upper right', fontsize=8)
    
    # --- RENDU 3D ---
    X_E, Y_E, Z_E = surf_E
    X_p, Y_p, Z_p = surf_plus
    
    # Affichage 3D : Ergosphère translucide + Horizon externe sombre
    ax3d.plot_surface(X_E, Y_E, Z_E, color='gold', alpha=0.25, edgecolor='orange', linewidth=0.2)
    ax3d.plot_surface(X_p, Y_p, Z_p, color='black', alpha=0.85, edgecolor='dimgray', linewidth=0.2)
    
    lim = 2.2 * M
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_title(f"Géométrie 3D de Kerr\n(Ergosphère dorée & Horizon noir)")
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

tracer_geometrie(a_init)

# ==========================================
# 3. CURSEUR INTERACTIF POUR MODIFIER LE SPIN (a)
# ==========================================
plt.subplots_adjust(bottom=0.15)
ax_slider = plt.axes([0.25, 0.03, 0.5, 0.03])
slider_a = Slider(ax_slider, 'Spin (a)', 0.0, 0.999, valinit=a_init, valfmt='%.3f M')

def update(val):
    a_val = slider_a.val
    tracer_geometrie(a_val)
    fig.canvas.draw_idle()

slider_a.on_changed(update)

plt.show()