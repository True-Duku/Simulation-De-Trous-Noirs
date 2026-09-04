# Simulation-De-Trous-Noirs

Gravity & Black Hole Simulations :

Simulations numériques et modélisation de la relativité générale (Schwarzschild, Kerr, 
Oppenheimer-Snyder). 

● Objectifs : 

○ Modéliser l'effondrement gravitationnel et la dynamique de la matière (forces de marée, 
spaghettification). 

○ Visualiser les déformations d'espace-temps en 1D, 2D (membrane de Flamm) et 3D 
volumétrique. 

○ Projet Machine Learning / Geometrical Learning : Appliquer des techniques de 
manifold learning (UMAP, t-SNE, Autoencodeurs géométriques) pour réduire la 
dimensionnalité des équations d'Einstein et extraire les structures fondamentales des 
métriques complexes. 

EXPLIQUATION DES FORMULES

### 🌌 Équations & Grandeurs Physiques de l'Effondrement Gravitationnel : formules utilisées (issues de variables formelles)

Pour simuler l'effondrement dynamique d'une étoile jusqu'à la formation d'un trou noir, le modèle doit calculer simultanément la courbure de l'espace-temps et la dynamique du fluide stellaire.

#### 1. Champ Gravitationnel & Courbure de l'Espace-Temps

* **Équations du champ d'Einstein (Formulation 4D)**
    G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R = \frac{8\pi G}{c^4} T_{\mu\nu}
  * **Calcule :** La courbure de l'espace-temps (G_{\mu\nu}) générée par la distribution de matière et d'énergie (T_{\mu\nu}).

* **Découpage Spatio-Temporel (Formulation ADM / 3+1)**
    ds^2 = -\alpha^2 c^2 dt^2 + \gamma_{ij} (dx^i + \beta^i dt)(dx^j + \beta^j dt)
  * **Calcule :** L'évolution temporelle de la métrique 3D (\gamma_{ij}), le décalage temporel propre/observateur (\alpha, *lapse*) et l'entraînement spatial      (\beta^i, *shift*).

#### 2. Hydrodynamique Relativiste & Matière

* **Tenseur Énergie-Impulsion du Fluide Parfait**
    T_{\mu\nu} = (\rho c^2 + P + e) u_\mu u_\nu + P g_{\mu\nu}
  * **Calcule :** L'énergie totale du fluide combinant la densité de masse ($\rho$), la pression ($P$), l'énergie interne ($e$) et la quadri-vitesse ($u_\mu$).

* **Conservation de la Matière & du Mouvement (Équations d'Euler Relativistes)**
    \nabla_\mu (\rho u^\mu) = 0 \quad \text{et} \quad \nabla_\mu T^{\mu\nu} = 0
  * **Calcule :** La conservation du nombre de baryons (masse) et le transfert de quantité de mouvement / énergie lors de la contraction.

* **Équation d'État de la Matière (EOS)**
    P = P(\rho, e)
  * **Calcule :** La pression de résistance de la matière (ex. équation polytropique P = K \rho^\gamma pour une étoile à neutrons) s'opposant au poids gravitationnel.

#### 3. Trajectoires & Forces de Marée

* **Équation des Géodésiques (Chute Libre)**
    \frac{d^2 x^\mu}{d\tau^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\tau} \frac{dx^\beta}{d\tau} = 0
  * **Calcule :** La trajectoire des éléments de matière ou des particules de test en temps propre (\tau) à travers les symboles de Christoffel (\Gamma^\mu_{\alpha\beta}).

* **Déviation Géodésique (Forces de Marée / Spaghettification)**
    \frac{D^2 n^\mu}{d\tau^2} = {R^\mu}_{\alpha\beta\gamma} u^\alpha u^\beta n^\gamma
  * **Calcule :** L'étirement radial et la compression transversale subis par un objet de taille finie via le tenseur de Riemann ({R^\mu}_{\alpha\beta\gamma}).

#### 4. Diagnostic & Horizontologie

* **Rayon de Schwarzschild / Horizon des Événements**
   R_s = \frac{2GM}{c^2}
  * **Calcule :** La limite critique au-deçà de laquelle la vitesse de libération dépasse la vitesse de la lumière c.

* **Scalaire de Kretschmann (Détection de la Singularité)**
    K = R^{\alpha\beta\gamma\delta} R_{\alpha\beta\gamma\delta}
  * **Calcule :** La mesure absolue de la courbure pour détecter la singularité physique (K \to \infty quand r \to 0) indépendamment du système de coordonnées.
