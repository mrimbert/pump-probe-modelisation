import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
import concurrent.futures
import time

# --------------------
# PARAMÈTRES UTILISATEURS A MODIFIER
# --------------------

#Renseigner ci-dessous les valeurs de la simulation à effectuer
#L'unité nécessaire pour le bon fonctionnement du code est indiquée entre crochet

#Dossier des résultats [doit être créé au préalable !]
results_folder = "results/" #Dossier de sauvegarde des résultats, doit se terminer par "/"

#Nombre de cœurs à utiliser pour le parallélisme (mettre 1 pour exécution séquentielle)
NBR_COEUR = 6

#Paramètres du matériau
n0 = 3.7 #[adim]
wavelength = 780e-9 # [m]
gamma = 1/(50e-15) # [s^-1]
tau_c = 1e-12 # [s]
mr = 5.36e-32 #[kg] masse réduite effective
Egap = 2.272e-19 #[J]
d_cv = (1.12e-28)**2 #[C².m²] moment dipolaire, indiquer |d_cv|²

#Paramètre de simulation
n_gold = 0.15 - 1j*4.74 #[adim] indice de réfraction du miroir d'or
couches_lineaires = np.array([], dtype=np.float64) # Format : [[indice, epaisseur], ...] Laisser vide si juste air
#Liste des listes [indice, epaisseur] des couches linéaires précédent le semiconducteur, laisser vide si seulement air
theta_inc_deg = 20 #Angle d'incidence du faisceau sur la structure (en degrés, 0° = incidence normale)
delays_ps = np.linspace(-0.5, 2, 15) #[ps] Délais à simuler pour le scan temporel, autour du pic du pulse d'entrée
epaisseurs_nm = np.linspace(253,1000,150) #[nm] Plage d'épaisseurs à simuler

#Paramètres de discrétisation
NEt = 100 # nombre d’échantillons énergie 
Nt = 10000 # nombre de pas en temps
Nz = 10 # nombre de pas en espace
Etmax = 8.01e-21 #[J] borne supérieure de l'intégrale de polarisation

#Paramètres du pulse d'entrée
FWHM = 100e-15 #[s] durée à mi-hauteur du pulse d'entrée
tau0 = 1e-12 #[s] temps de pic du pulse d'entrée
amplitude = 1.5e7 #[V/m] amplitude de la pompe utilisée dans le code

# --------------------
# CONSTANTES PHYSIQUES FONDAMENTALES
# --------------------

c = 3e8 #[m.s^-1]
h_bar = 1.05e-34 #[J.s]
m0 = 9.1e-31 #[kg] masse d'un électron
q = 1.6e-19 #[C] charge élémentaire
epsilon0 = 8.85e-12 
mu0 = 4*np.pi*1e-7

# -------------------
# TABLEAUX DE SORTIE 
# -------------------

#On liste ci-dessous les différents tableaux des variables à calculer.
#Ils peuvent être utilisés pour tracer les différentes courbes à la fin de la simulation.

#Tableaux des enveloppes du champ électrique
E_plus = np.zeros((Nz, Nt), dtype=complex) #Enveloppe du champ croissante
E_moins = np.zeros((Nz, Nt), dtype=complex) #Enveloppe du champ décroissante
E = np.zeros((Nz,Nt), dtype=complex) #Enveloppe croissante + décroissante du champ

#Tableaux de l'indice optique effectif et du coefficient d'absorption
n_eff_array = np.zeros(Nt, dtype=complex)
alpha_array = np.zeros(Nt) 

#Tableaux pour stocker l'évolution des populations
rho_e = np.zeros((NEt, Nz, Nt))
rho_h = np.zeros((NEt, Nz, Nt))

#Tableau pour stocker la valeur du produit de convolution
F_array = np.zeros((NEt, Nz, Nt), dtype=complex)
F_array_plus = np.zeros((NEt, Nz, Nt), dtype=complex)
F_array_moins = np.zeros((NEt, Nz, Nt), dtype=complex)

#Tableau pour stocker la valeur de la polarisation
polarisation= np.zeros((Nz,Nt),dtype=complex)
polarisation_plus= np.zeros((Nz,Nt),dtype=complex)
polarisation_moins= np.zeros((Nz,Nt),dtype=complex)

N_entree = np.zeros(Nt)

# --------------------
# VALEURS DERIVÉES
# --------------------
v_g = c/n0 #[m.s^-1] vitesse de groupe
omega0 = (2*np.pi*c/wavelength) #[rad.s^-1] pulsation de référence
k0 = n0*omega0/c

theta_inc_rad = np.radians(theta_inc_deg) # Conversion de l'angle d'incidence en radians

g = (d_cv*omega0)/(4*n0*c*epsilon0*h_bar) #[adim]
kappa = d_cv/(2*h_bar**2)  # force du couplage champ-populations utilisé dans l'évolution des populations
D0 = 1/(2*np.pi**2)*(2*mr/(h_bar)**2)**(3/2)

Et0 = h_bar*omega0-Egap

Et = np.linspace(-Et0,Etmax*2,NEt)

a = 0.64/q #Paramètre de non-parabolicité
Dr = D0*np.sqrt((Et+Et0)*(1+a*(Et+Et0)))*(1+2*a*(Et+Et0)) #Densité d'états en 3D (il s'agit d'un tableau en Et)

sigma = FWHM / (2*np.sqrt(2*np.log(2)))

if len(couches_lineaires) == 0:
    couches_lineaires = np.zeros((0, 2), dtype=np.float64)


# --------------------
# FONCTIONS OPTIMISÉES AVEC NJIT (PURE FUNCTIONS)
# --------------------
# Ces fonctions restent globales car elles sont compilées et sans état

@njit
def get_multistack_coeffs_njit(layers_array, theta_rad):
    """
    Calcule Fresnel et TMM pour incidence oblique (Polarisation s / TE).
    Retourne r_in, t_in, r_out, t_out ET le cos(theta) dans le substrat (pour la suite).
    """
    # Matrice Identité complexe
    M = np.eye(2, dtype=np.complex128)
    n_curr = 1.0 + 0j # Air
    theta_curr = theta_rad
    
    # Itération sur les couches (si layers_array n'est pas vide)
    n_layers = layers_array.shape[0]
    
    for i in range(n_layers):
        n_next = layers_array[i, 0]
        d_next = layers_array[i, 1]

        # Snell
        sin_theta_next = (n_curr * np.sin(theta_curr)) / n_next
        cos_theta_curr = np.sqrt(1 - np.sin(theta_curr)**2 + 0j)
        cos_theta_next = np.sqrt(1 - sin_theta_next**2 + 0j)

        # Fresnel s (TE) Interface
        num_r = n_curr * cos_theta_curr - n_next * cos_theta_next
        den_r = n_curr * cos_theta_curr + n_next * cos_theta_next
        r = num_r / den_r
        t = (2 * n_curr * cos_theta_curr) / den_r
        
        M_int = (1/t) * np.array([[1, r], [r, 1]], dtype=np.complex128)
        M = np.dot(M, M_int)
        
        # Propagation
        phi = (2 * np.pi * n_next * d_next * cos_theta_next) / wavelength
        M_prop = np.array([[np.exp(-1j*phi), 0j], [0j, np.exp(1j*phi)]], dtype=np.complex128)
        M = np.dot(M, M_prop)
        
        n_curr = n_next
        theta_curr = np.arcsin(sin_theta_next)
        
    # Dernière interface : vers Substrat (n0)
    sin_theta_sub = (n_curr * np.sin(theta_curr)) / n0
    cos_theta_curr = np.sqrt(1 - np.sin(theta_curr)**2 + 0j)
    cos_theta_sub = np.sqrt(1 - sin_theta_sub**2 + 0j)
    
    num_r = n_curr * cos_theta_curr - n0 * cos_theta_sub
    den_r = n_curr * cos_theta_curr + n0 * cos_theta_sub
    r = num_r / den_r
    t = (2 * n_curr * cos_theta_curr) / den_r
    
    M_int = (1/t) * np.array([[1, r], [r, 1]], dtype=np.complex128)
    M = np.dot(M, M_int)
    
    det_M = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    r_in = M[1,0] / M[0,0]
    t_in = 1.0 / M[0,0]
    r_out = -M[0,1] / M[0,0]
    t_out = det_M / M[0,0] 
    
    return r_in, t_in, r_out, t_out, cos_theta_sub

@njit
def F_moins_njit(F_array_moins, E_moins, alpha, gamma, Et, h_bar, m, n):
    F_array_moins[:, m, n] = alpha*F_array_moins[:,m,n-1] + 1/(gamma+(Et)/h_bar*1j)*(1-alpha)*E_moins[m,n-1]
    return F_array_moins

@njit
def F_plus_njit(F_array_plus, E_plus, alpha, gamma, Et, h_bar, m, n):
    F_array_plus[:, m, n] = alpha*F_array_plus[:,m,n-1] + 1/(gamma+(Et)/h_bar*1j)*(1-alpha)*E_plus[m,n-1]
    return F_array_plus

@njit
def F_njit(F_array, F_array_plus, F_array_moins, k0, dz, m, n):
    F_array[:, m, n] = F_array_plus[:,m,n]*np.exp(-1j*k0*dz*m) + F_array_moins[:,m,n]*np.exp(1j*k0*dz*m)
    return F_array

@njit
def f_plus_njit(rho_e, rho_h, F_array_plus, Dr, Et, m, n):
    integrand = (rho_e[: , m, n] + rho_h[:, m, n]-1) * F_array_plus[: , m, n] * Dr
    return np.trapezoid(integrand, Et)

@njit
def f_moins_njit(rho_e, rho_h, F_array_moins, Dr, Et, m, n):
    integrand = (rho_e[:, m, n] + rho_h[:, m, n]-1) * F_array_moins[:, m, n] * Dr
    return np.trapezoid(integrand, Et)

@njit
def f_njit(rho_e, rho_h, F_array, Dr, Et, m, n):
    integrand = (rho_e[:, m, n] + rho_h[:, m, n]-1) * F_array[: , m, n] * Dr
    return np.trapezoid(integrand, Et)

@njit
def rho_calcul_njit(rho_e, rho_h, F_array, E_value, kappa, dt, tau_c, n, m):
    interaction = np.real(np.conj(E_value) * F_array[:, m, n])
    drho = kappa * (1-rho_e[: , m, n-1] - rho_h[:, m, n-1]) * interaction
    rho_e[:, m, n] = rho_e[:, m, n-1] + dt * (drho - 1/tau_c * rho_e[:, m, n-1])
    rho_h[:, m, n] = rho_h[:, m, n-1] + dt * (drho - 1/tau_c * rho_h[:, m, n-1])
    rho_e[:, m, n] = np.clip(rho_e[: , m, n], 0.0, 0.5)
    rho_h[:, m, n] = np. clip(rho_h[:, m, n], 0.0, 0.5)
    return rho_e, rho_h



# --------------------
# WORKER FUNCTION (Exécutée sur chaque cœur)
# --------------------

def process_thickness(L_curr):
    """
    Fonction autonome qui effectue tout le calcul pour une épaisseur L_curr donnée.
    Elle recrée ses propres grilles temporelles et tableaux pour éviter les conflits mémoire.
    """
    r_in_0, t_in_0, r_out_0, t_out_0, cos_theta_gaas_complex = get_multistack_coeffs_njit(couches_lineaires, theta_inc_rad)
    cos_theta_gaas_real = np.real(cos_theta_gaas_complex)
    if np.abs(cos_theta_gaas_real) < 1e-12:
        raise ValueError("cos(theta) dans le substrat est trop proche de 0 : incidence quasi rasante non supportée")
    L_normal = L_curr * 1e-9
    k0z = k0 * cos_theta_gaas_real

    # 1. Recalcul de la grille temporelle et spatiale spécifique à cette épaisseur
    # On discrétise l'épaisseur réelle (normale à la couche)
    # et on adapte le temps via la vitesse projetée sur z.
    dz = L_normal / (Nz-1)
    v_g_z = v_g * cos_theta_gaas_real
    dt = dz / v_g_z
    
    t = np.arange(Nt) * dt
    alpha = np.exp(dt*(-1j*(Et)/h_bar-gamma)) # Recalcul de alpha car dt a changé

    sin_theta_gold = (1.0 * np.sin(theta_inc_rad)) / n_gold
    cos_theta_gold = np.sqrt(1 - sin_theta_gold**2 + 0j)

    num_r = n0 * cos_theta_gaas_complex - n_gold * cos_theta_gold
    den_r = n0 * cos_theta_gaas_complex + n_gold * cos_theta_gold
    r_gold_val = num_r / den_r
    
    # Paramètres de pulse
    FWHM = 100e-15
    sigma = FWHM / (2*np.sqrt(2*np.log(2)))
    tau0 = t[2500] 
    
    # 2. Allocation locale des tableaux (remplace reset_arrays)
    # Ces variables n'existent que dans ce processus
    E_plus = np.zeros((Nz, Nt), dtype=complex)
    E_moins = np.zeros((Nz, Nt), dtype=complex)
    E = np.zeros((Nz, Nt), dtype=complex)
    rho_e = np.zeros((NEt, Nz, Nt))
    rho_h = np.zeros((NEt, Nz, Nt))
    F_array = np.zeros((NEt, Nz, Nt), dtype=complex)
    F_array_plus = np.zeros((NEt, Nz, Nt), dtype=complex)
    F_array_moins = np.zeros((NEt, Nz, Nt), dtype=complex)

    # --- Fonction de propagation locale ---
    # Cette fonction a accès aux tableaux locaux définis ci-dessus via la closure,
    # ou on peut réinitialiser les tableaux à chaque appel. 
    # Pour simplifier, on écrit une routine qui prend 'source_input' et reset les tableaux locaux.
    
    def run_simulation_internal(source_pulse_input):
        # Reset des tableaux locaux à zéro
        E_plus.fill(0)
        E_moins.fill(0)
        E.fill(0)
        rho_e.fill(0)
        rho_h.fill(0)
        F_array.fill(0)
        F_array_plus.fill(0)
        F_array_moins.fill(0)
        
        # Boucle temporelle
        for n in range(1, Nt-1):
            E_plus[0, n] = t_in_0 * source_pulse_input[n] + r_out_0 * E_moins[0, n-1]
            
            # Propagation avant
            for m in range(0, Nz-1):
                # Appel direct aux fonctions numba avec les tableaux locaux
                F_plus_njit(F_array_plus, E_plus, alpha, gamma, Et, h_bar, m, n)
                F_moins_njit(F_array_moins, E_moins, alpha, gamma, Et, h_bar, m, n)
                F_njit(F_array, F_array_plus, F_array_moins, k0z, dz, m, n)
                
                E_combined = E_plus[m, n-1]*np.exp(-1j*k0z*dz*m) + E_moins[m, n-1]*np.exp(1j*k0z*dz*m)
                
                rho_calcul_njit(rho_e, rho_h, F_array, E_combined, kappa, dt, tau_c, n, m)
                
                # Calcul polarisation et step spatial
                pol_dynamique = f_plus_njit(rho_e, rho_h, F_array_plus, Dr, Et, m, n)
                E_plus[m+1, n] = E_plus[m, n-1] + dz * g * (pol_dynamique)
                
            m_back = Nz-1
            F_plus_njit(F_array_plus, E_plus, alpha, gamma, Et, h_bar, m_back, n)
            F_moins_njit(F_array_moins, E_moins, alpha, gamma, Et, h_bar, m_back, n)
            F_njit(F_array, F_array_plus, F_array_moins, k0z, dz, m_back, n)
            
            # 2. Calcul du champ combiné au fond (basé sur n-1 pour interaction causale)
            E_combined_back = E_plus[m_back, n-1]*np.exp(-1j*k0z*dz*m_back) + E_moins[m_back, n-1]*np.exp(1j*k0z*dz*m_back)
            
            # 3. Mise à jour des populations (rho) au fond pour l'instant n
            rho_calcul_njit(rho_e, rho_h, F_array, E_combined_back, kappa, dt, tau_c, n, m_back)
            
            # Condition limite arrière

            L_total = (Nz-1)*dz
            dephasage = np.exp(-2j*k0z*L_total)
            E_moins[-1, n] = r_gold_val * E_plus[-1, n] * dephasage
            
            # Propagation arrière
            for m in range(Nz-1, 0, -1):
                pol_dynamique = f_moins_njit(rho_e, rho_h, F_array_moins, Dr, Et, m, n)
                E_moins[m-1, n] = E_moins[m, n-1] + dz * g * (pol_dynamique)
            
            # Interface avant et coefficients dynamiques
            E[0, n] = E_plus[0, n] + E_moins[0, n]
            
            # MaJ champs complets (optionnel pour la réflectivité seule mais présent dans le code original)
            # for m in range(1, Nz): ... (omis pour gain de perf si pas nécessaire)

        # Calcul réflectivité
        E_reflected = r_in_0 * source_pulse_input + t_out_0 * E_moins[0, :]
        return E_reflected

    # --- Séquence de tirs (Shots) ---
    amplitude_pump_fixed = amplitude
    amplitude_probe_fixed = amplitude_pump_fixed * 0.005
    
    # 1. Tir Sonde Seule (Référence Linéaire)
    pulse_probe_ref = amplitude_probe_fixed * np.exp(-((t-tau0)**2)/(2*sigma**2))
    E_refl_ref = run_simulation_internal(pulse_probe_ref)
    
    Fluence_in_ref = np.trapezoid(np.abs(pulse_probe_ref)**2, t)
    Fluence_refl_ref = np.trapezoid(np.abs(E_refl_ref)**2, t)
    R0_curr = Fluence_refl_ref / Fluence_in_ref
    
    # 2. Tir Pompe Seule
    pulse_pump = amplitude_pump_fixed * np.exp(-((t-tau0)**2)/(2*sigma**2))
    E_refl_pump = run_simulation_internal(pulse_pump)
    
    # 3. Scan Temporel (Delays)
    delays_scan = delays_ps * 1e-12
    R_values_temp = []
    
    for delay in delays_scan:
        tau_probe_delayed = tau0 + delay
        pulse_probe_delayed = amplitude_probe_fixed * np.exp(-((t-tau_probe_delayed)**2)/(2*sigma**2))
        pulse_total = pulse_pump + pulse_probe_delayed
        
        E_refl_total = run_simulation_internal(pulse_total)
        
        # Extraction signal sonde par soustraction
        E_probe_refl = E_refl_total - E_refl_pump
        Fluence_refl_probe = np.trapezoid(np.abs(E_probe_refl)**2, t)
        
        R_curr = Fluence_refl_probe / Fluence_in_ref
        R_values_temp.append(R_curr)
        
    # Résultats finaux pour cette épaisseur
    R_max = np.max(R_values_temp)
    Delta_R = R_max - R0_curr
    Delta_R_sur_R_max = Delta_R / R_max if R_max > 1e-10 else 0
    Delta_R_sur_R = np.max((np.array(R_values_temp)-R0_curr)/np.array(R_values_temp))
    
    return L_curr, R_max, R0_curr, Delta_R_sur_R_max, Delta_R_sur_R


# --------------------
# MAIN EXECUTION
# --------------------
if __name__ == "__main__":
    
    print("\n" + "="*50)
    print(f" PARTIE C : SCAN EN ÉPAISSEUR PARALLÉLISÉ ({NBR_COEUR} Cœurs)")
    print("="*50)

    # Listes pour stocker les résultats triés
    results_map = {}

    print(f"Scan sur {len(epaisseurs_nm)} épaisseurs...")
    start_time = time.time()

    # Exécution parallèle
    with concurrent.futures.ProcessPoolExecutor(max_workers=NBR_COEUR) as executor:
        # On lance les tâches
        futures = {executor.submit(process_thickness, L): L for L in epaisseurs_nm}
        
        # On récupère au fur et à mesure
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            L_val, R_max, R0_curr, dR_Rmax, dR_R = future.result()
            
            # Stockage dans un dictionnaire pour remettre dans l'ordre après
            results_map[L_val] = (R_max, R0_curr, dR_Rmax, dR_R)
            
            elapsed = time.time() - start_time
            print(f"   [{i+1}/{len(epaisseurs_nm)}] Terminé L={L_val:.1f}nm | R_max={R_max*100:.2f}% | Temps écoulé: {elapsed:.1f}s")

    print("\nScan terminé.")

    # Reconstitution des listes ordonnées
    R_max_list = []
    R0_list = []
    Delta_R_sur_R_max_list = []
    Delta_R_sur_R_list = []

    for L in epaisseurs_nm:
        res = results_map[L]
        R_max_list.append(res[0])
        R0_list.append(res[1])
        Delta_R_sur_R_max_list.append(res[2])
        Delta_R_sur_R_list.append(res[3])

    # --- TRACÉS ---
    fig3, ax3 = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    ax3[0].plot(epaisseurs_nm, np.array(R_max_list)*100, 'b-o', linewidth=2, label='$R_{max}$ (saturé)')
    ax3[0].plot(epaisseurs_nm, np.array(R0_list)*100, 'g--s', linewidth=2, label='$R_0$ (linéaire)')
    ax3[0].set_ylabel('Réflectivité (%)', fontsize=12)
    ax3[0].set_title('Réflectivité en fonction de l\'épaisseur de GaAs')
    ax3[0].legend()
    ax3[0].grid(True, alpha=0.3)

    ax3[1].plot(epaisseurs_nm, (np.array(R_max_list) - np.array(R0_list))*100, 'm-^', linewidth=2)
    ax3[1].set_ylabel(r'$\Delta R$ (%)', fontsize=12)
    ax3[1].set_title('Modulation absolue')
    ax3[1].grid(True, alpha=0.3)

    ax3[2].plot(epaisseurs_nm, np.array(Delta_R_sur_R_max_list)*100, 'r-s', linewidth=2)
    ax3[2].set_xlabel('Épaisseur de GaAs (nm)', fontsize=12)
    ax3[2].set_ylabel(r'$\Delta R / R_{max}$ (%)', fontsize=12)
    ax3[2].set_title('Modulation relative')
    ax3[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.plot()

    plt.savefig(f'{results_folder}figure/scanepaisseur_{amplitude:.2e}.png')
    print(f"\nGraphique sauvegardé sous '{results_folder}figure/scanepaisseur_{amplitude:.2e}.png'")

    plt.show()

    # Save results to CSV file
    results_df = pd.DataFrame({
        'Thickness (nm)': epaisseurs_nm,
        'R_max':   R_max_list,
        'R_linear (pump off)': R0_list,
        'dR/Rmax': Delta_R_sur_R_max_list,
        'max(dR/R)': Delta_R_sur_R_list
    })

    csv_filename = f'{results_folder}data/donnees_simulation_L_parallel_{amplitude:.2e}.csv'
    results_df.to_csv(csv_filename, index=False, sep=';', decimal=',')
    print(f"Données sauvegardées sous : {csv_filename}")