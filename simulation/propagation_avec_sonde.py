import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------
# PARAMÈTRES UTILISATEURS A MODIFIER
# --------------------

#Renseigner ci-dessous les valeurs de la simulation à effectuer
#L'unité nécessaire pour le bon fonctionnement du code est indiquée entre crochet

#Dossier des résultats [doit être créé au préalable !]
results_folder = "results/" #Dossier de sauvegarde des résultats, doit se terminer par "/"

#Paramètres du matériau
n0 = 3.7 #[adim]
wavelength = 780e-9 # [m]
gamma = 1/(50e-15) # [s^-1]
tau_c = 0.5e-12 # [s]
mr = 5.36e-32 #[kg] masse réduite effective
Egap = 2.272e-19 #[J]
d_cv = (1.12e-28)**2 #[C².m²] moment dipolaire, indiquer |d_cv|²

#Paramètre de simulation
L_objectif = 401e-9 #[m] épaisseur de semiconducteur à simuler
n_gold = 0.15 - 1j*4.74 #[adim] indice de réfraction du miroir d'or
couches_lineaires = [] #Liste des tuples (indice, epaisseur) des couches linéaires précédent le semiconducteur, laisser vide si seulement air
delays_ps = np.linspace(-0.25, 3, 35) #[ps] Délais à simuler pour le scan temporel, autour du pic du pulse d'entrée
amplitudes_scan = np.geomspace(1e6, 3e9, 15) #[V/m] Amplitudes de pompe à simuler pour le scan en fluence, couvrant une plage large pour observer la saturation

#Paramètres de discrétisation
NEt = 100 # nombre d’échantillons énergie 
Nt = 10000 # nombre de pas en temps
Nz = 11 # nombre de pas en espace
Etmax = 8.01e-21 #[J] borne supérieure de l'intégrale de polarisation

#Paramètres du pulse d'entrée
FWHM = 100e-15 #[s] durée à mi-hauteur du pulse d'entrée
tau0 = 1e-12 #[s] temps de pic du pulse d'entrée
amplitude = 3.2e7 #[V/m] amplitude de la pompe utilisée dans le code
R = 7.5e-6  #[m] Rayon du faisceau
F_repetition = 80e6 #[Hz] Fréquence de répétition du laser

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

#Tableaux des coefficients de Fresnel à l'entrée
t_in_array = np.zeros(Nt, dtype=complex)   # in : Air → GaAs
t_out_array = np.zeros(Nt, dtype=complex)  # out : GaAs → Air
r_in_array = np.zeros(Nt,dtype=complex)
r_out_array = np.zeros(Nt,dtype=complex)

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

g = (d_cv*omega0)/(4*n0*c*epsilon0*h_bar) #[adim]
kappa = d_cv/(2*h_bar**2)  # force du couplage champ-populations utilisé dans l'évolution des populations
D0 = 1/(2*np.pi**2)*(2*mr/(h_bar)**2)**(3/2)

dt = L_objectif/(v_g*(Nz-1))  # pas en temps
dz = v_g*dt # pas en espace 

Et0 = h_bar*omega0-Egap

t = np.arange(Nt) * dt
z = np.arange(Nz) * dz 
Et = np.linspace(-Et0,Etmax*2,NEt)

a = 0.64/q #Paramètre de non-parabolicité
Dr = D0*np.sqrt((Et+Et0)*(1+a*(Et+Et0)))*(1+2*a*(Et+Et0)) #Densité d'états en 3D (il s'agit d'un tableau en Et)

#On précalcule le coefficient alpha utilisée dans le calcul du produit de convolution pour gagner du temps d'exécution.
alpha = np.exp(dt*(-1j*(Et)/h_bar-gamma))

sigma = FWHM / (2*np.sqrt(2*np.log(2)))

# --------------------
# DEFINITION DU PULSE D'ENTRÉE
# --------------------
base_pulse = np.exp(-((t-tau0)**2)/(2*sigma**2))
source_pulse = np.zeros((Nt), dtype=complex)
source_pulse = amplitude * base_pulse

# --------------------
# FONCTIONS 
# --------------------
def get_multistack_coeffs(layers_list):
    """
    Calcule les coefficients de réflexion et de transmission d'une structure stratifiée à partir de la matrice de transfert.
    La structure est définie par une liste de tuples (n, d) où n est l'indice de réfraction de la couche et d son épaisseur. La dernière interface est celle entre la dernière couche et un milieu effectif d'indice n_eff.
    Les coefficients sont calculés pour une incidence normale.

    Parameters
    ----------
    layers_list : list of tuples
        Liste des couches linéaires précédant le semiconducteur, sous la forme [(n1, d1), (n2, d2), ...].
    """
    M = np.eye(2, dtype=complex)
    n_curr = 1.0 #Indice de départ : Air
    
    for (n_next, d_next) in layers_list:
        #Première interface
        r = (n_curr-n_next)/(n_curr+n_next)
        t = 2*n_curr/(n_curr+n_next)
        #Mise à jour de la matrice M
        M_int = 1/t*np.array([[1,r],[r,1]], dtype=complex)
        M = np.dot(M,M_int)
        #Propagation dans la couche
        phi = 2*np.pi*n_next*d_next/wavelength
        M_prop = np.array([[np.exp(-1j*phi),0],[0,np.exp(1j*phi)]], dtype=complex)
        M = np.dot(M,M_prop)
        
        n_curr = n_next
        
    #Pour la dernière couche :
    r = (n_curr-n0)/(n_curr + n0)
    t = 2*n_curr / (n_curr+n0)
    M_int = 1/t*np.array([[1,r],[r,1]], dtype=complex)
    M = np.dot(M,M_int)
    
    #Extraction des coefficients à renvoyer
    det_M = M[0,0]*M[1,1] - M[0,1]*M[1,0]

    r_in = M[1,0] / M[0,0]
    t_in = 1.0 / M[0,0]
    r_out = -M[0,1] / M[0,0]
    t_out = det_M / M[0,0] 
    
    return r_in, t_in, r_out, t_out

def F_moins(m,n):
    F_array_moins[:, m,n] = alpha*F_array_moins[:,m,n-1] + 1/(gamma+(Et)/h_bar*1j)*(1-alpha)*E_moins[m,n-1]
    return

def F_plus(m,n):
    F_array_plus[:, m,n] = alpha*F_array_plus[:,m,n-1] + 1/(gamma+(Et)/h_bar*1j)*(1-alpha)*E_plus[m,n-1]
    return

def F(m, n):
    """
    Calcule la convolution récursive spectrale F(n) = α·F(n-1) + (1-α)/(γ + iE_t+E_t0/ℏ)·E(n-1).
    
    Parameters
    ----------
    m : int
        Index spatial.
    n : int
        Index temporel.
    """    
    F_array[:, m,n] = F_array_plus[:,m,n]*np.exp(-1j*k0*dz*m) + F_array_moins[:,m,n]*np.exp(1j*k0*dz*m)
    return
 
def f(m, n):
    """
    Intègre la polarisation P = ∫ (ρₑ + ρₕ - 1)·F(E,t)·D(E) dE.
    ATTENTION : ce n'est en réalité pas une polarisation car il manque le facteur multiplicatif en i|d_cv|**2/(2h_barGamma)
    Celui-ci est déjà comptabilisé dans le gain g0 mais pour calculer n_eff, il ne faut pas oublier de remettre ce facteur.
    
    Parameters
    ----------
    m : int
        Index spatial.
    n : int
        Index temporel.
    
    Returns
    -------
    complex
        Polarisation au point (m, n).
    """
    polarisation[m,n] = np.trapezoid((rho_e[:, m, n] + rho_h[:, m, n]-1) * F_array[:, m,n]*Dr,Et)
    return

def f_plus(m,n):
    polarisation_plus[m,n] = np.trapezoid((rho_e[:, m, n] + rho_h[:, m, n]-1) * F_array_plus[:, m,n]*Dr,Et)
    return polarisation_plus[m,n]

def f_moins(m,n):
    polarisation_moins[m,n] = np.trapezoid((rho_e[:, m, n] + rho_h[:, m, n]-1) * F_array_moins[:, m,n]*Dr,Et)
    return polarisation_moins[m,n]

def rho_calcul(n, m, E_value):
    """
    Résout dρ/dt = κ(1-ρₑ-ρₕ)|E·F| - ρ/τ_c avec clipping ρ ∈ [0, 0.5].
    
    Parameters
    ----------
    n : int
        Index temporel.
    m : int
        Index spatial.
    E_value : complex
        Champ électrique combiné.
    """
    
    interaction = np.real(np.conj(E_value) * F_array[:, m,n])
    drho =  kappa * (1-rho_e[:, m, n-1] - rho_h[:, m, n-1]) * interaction
    rho_e[:, m, n] = rho_e[:, m, n-1] + dt * (drho - 1/tau_c * rho_e[:, m, n-1])
    rho_h[:, m, n] = rho_h[:, m, n-1] + dt * (drho - 1/tau_c * rho_h[:, m, n-1])
    rho_e[:, m, n] = np.clip(rho_e[:, m, n], 0, 0.5)
    rho_h[:, m, n] = np.clip(rho_h[:, m, n], 0, 0.5)
    return


def propagation_croissante(n, m):
    """
    Propage l'onde progressive : E⁺(m+1, n) = E⁺(m, n-1) + dz·g·P+(m, n).
    
    Parameters
    ----------
    n : int
        Index temporel.
    m : int
        Index spatial.
    """
    
    pol_dynamique = f_plus(m, n)   
    E_plus[m+1, n] = E_plus[m, n-1] + dz * g * (pol_dynamique)  
    return 

def propagation_decroissante(n, m):
    """Propage l'onde progressive : E-(m-1, n) = E-(m, n-1) + dz·g·P-(m, n).
    
    Parameters
    ----------
    n : int
        Index temporel.
    m : int
        Index spatial.
    """
    
    pol_dynamique = f_moins(m, n)  
    E_moins[m-1, n] = E_moins[m, n-1] + dz * g * (pol_dynamique)  
    return

def get_linear_susceptibility():
    """Retourne la susceptibilité linéaire χ_lin = (i|d_cv|)/(2ℏε₀)·∫ D(E)/(γ + iE/ℏ) dE.
    
    Returns
    -------
    complex
        Susceptibilité linéaire du matériau.
    """
    
    chi_lin = 0j
    response_per_energy = 1 / (gamma + 1j * (Et) / h_bar)
    integral = np.trapezoid(-Dr * response_per_energy, Et)
    
    norm_factor = (1j * d_cv) / (2*h_bar*epsilon0)
    
    chi_lin = norm_factor * integral
    
    return chi_lin

#Susceptibilité approchée loin de la saturation
chi_linear_material = get_linear_susceptibility()
n_lin_complex = np.sqrt(n0**2 + chi_linear_material)

def calculate_total_neff(E_field, n, m):
    """Retourne l'indice effectif complexe n_eff = √(n₀² + χ_dyn(ρₑ, ρₕ)).
    
    Parameters
    ----------
    E_field : complex
        Champ électrique au point considéré.
        Polarisation au point considéré.
    n : int
        Index temporel.
    m : int
        Index spatial.
    
    Returns
    -------
    complex
        Indice effectif saturé, avec validations numériques.
    """

    response_per_energy = 1 / (gamma + 1j *(Et) / h_bar)
    
    # Facteur de Pauli DYNAMIQUE (avec saturation)
    pauli_factor =  (rho_e[:, m, n] + rho_h[:, m, n] - 1) 
    
    # Intégrer avec le facteur de Pauli dynamique
    integral_dynamic = np.trapezoid(pauli_factor * Dr * response_per_energy, Et)
    
    # Appliquer le facteur de normalisation
    norm_factor = (1j* d_cv) / (2*h_bar * epsilon0)
    
    # Susceptibilité résonante
    chi_resonant = norm_factor * integral_dynamic

    # Racine carrée
    n_eff =np.sqrt(n0**2+chi_resonant)
    
    #On prend toujours la racine avec une partie imaginaire positive. 
    if np.imag(n_eff) < 0:
        n_eff = np.conj(n_eff)

    
    # Sécurité basique contre les divergences extrêmes
    if np.isnan(n_eff) or np.isinf(n_eff):
        return n_lin_complex
    
    # Vérifier que les valeurs restent physiquement raisonnables
    if abs(n_eff.imag) > 1.0 or abs(n_eff.real) > 5.0 or abs(n_eff.real) < 1.0:
        return n_lin_complex

    return n_eff


def propagation():
    """Résout la propagation couplant Maxwell et Bloch optiques sur la grille (z,t).
    """
    
    #On initialise les différents coefficients à l'aide de l'indice linéaire complexe et de la TMM.
    r_eff_air_gaas, t_eff_air_gaas, r_eff_gaas_air, t_eff_gaas_air = get_multistack_coeffs(couches_lineaires)


    # On stocke les coefficients qui seront utilisés au pas de temps n=1
    r_out_array[0] = r_eff_gaas_air
    t_in_array[0] = t_eff_air_gaas
    r_in_array[0] = r_eff_air_gaas
    t_out_array[0] = t_eff_gaas_air
    
    for n in range(1,Nt-1):
        #Condition limite à l'interface Air/GaAs
        E_plus[0,n] = t_in_array[n-1] * source_pulse[n] + r_out_array[n-1] * E_moins[0,n-1]
        
        #Propagation croissante    
        for m in range(0,Nz-1):
            F_plus(m,n)
            F_moins(m,n)
            F(m,n)
            E_combined = E_plus[m,n-1]*np.exp(-1j*k0*dz*m) + E_moins[m,n-1]*np.exp(1j*k0*dz*m)
            rho_calcul(n, m, E_combined)
            propagation_croissante(n,m)
            
        m_back = Nz-1
            
        F_plus(m_back, n)
        F_moins(m_back, n)
        F(m_back, n)
        
        # 2. Calcul du champ combiné au fond (basé sur n-1 pour interaction causale)
        E_combined_back = E_plus[m_back, n-1]*np.exp(-1j*k0*dz*m_back) + E_moins[m_back, n-1]*np.exp(1j*k0*dz*m_back)
        
        # 3. Mise à jour des populations (rho) au fond pour l'instant n
        rho_calcul(n, m_back, E_combined_back)

        # Coefficient de réflexion GaAs → Or
        r_gold = (n0 - n_gold) / (n_gold + n0)
        
        L = (Nz-1)*dz
        dephasage = np.exp(-2j*k0*L)
        
        
        # Condition limite arrière réaliste
        E_moins[-1, n] = r_gold * E_plus[-1, n] * dephasage
        
        #Propagation décroissante
        for m in range(Nz-1,0,-1):
            f(m,n) #On met à jour la polarisation totale simplement pour nos analyses sur la SVEA
            propagation_decroissante(n,m) 
  
    
        E[0, n] = E_plus[0,n]+E_moins[0,n]
        
        F(0,n)
        
        n_eff = calculate_total_neff(E[0,n], n, 0)
            
        # Stockage de n_eff
        n_eff_array[n] = n_eff
        
        # Calcul du coefficient d'absorption alpha = 4*pi*k/lambda = 2*omega*Im(n)/c
        alpha_array[n] = 2 * omega0 * np.imag(n_eff) / c
    
    
        r_eff_air_gaas, t_eff_air_gaas, r_eff_gaas_air, t_eff_gaas_air = get_multistack_coeffs(couches_lineaires)
        
        t_in_array[n] = t_eff_air_gaas
        t_out_array[n] = t_eff_gaas_air 
        r_in_array[n] = r_eff_air_gaas
        r_out_array[n] = r_eff_gaas_air
        
        N_entree[n] = np.trapezoid(rho_e[:, 0, n] * Dr, Et)
        
        #Calcul de E total
        for m in range(1,Nz):
           E[m,n] = E_plus[m,n]*np.exp(-1j*k0*dz*m) + E_moins[m,n]*np.exp(1j*k0*dz*m)
    return

# --------------------
# RUN
# --------------------

def reset_arrays():
    """Réinitialise tous les tableaux globaux à zéro avant une nouvelle propagation."""
    global E_plus, E_moins, E, rho_e, rho_h, F_array, F_array_plus, F_array_moins
    global polarisation, polarisation_plus, polarisation_moins, n_eff_array, alpha_array
    global t_in_array, t_out_array, r_in_array, r_out_array, N_entree, source_pulse
    
    E_plus.fill(0)
    E_moins.fill(0)
    E.fill(0)
    rho_e.fill(0)
    rho_h.fill(0)
    F_array.fill(0)
    F_array_plus.fill(0)
    F_array_moins.fill(0)
    polarisation.fill(0)
    polarisation_plus.fill(0)
    polarisation_moins.fill(0)
    n_eff_array.fill(0)
    alpha_array.fill(0)
    t_in_array.fill(0)
    t_out_array.fill(0)
    r_in_array.fill(0)
    r_out_array.fill(0)
    N_entree.fill(0)

def run_shot(input_pulse):
    """
    Lance une propagation et retourne :
    - E_reflected : Champ réfléchi vers l'air (z=0)
    - E_transmitted : Champ transmis après le miroir/fond (z=L)
    """
    global source_pulse 
    
    reset_arrays()
    source_pulse = input_pulse # Mise à jour de la source globale
    propagation() # Lancement du calcul physique
    
    E_reflected = r_in_array * source_pulse + t_out_array * E_moins[0, :]

    return E_reflected

# ==============================================================================
# PARTIE A : SCAN TEMPOREL (DYNAMIQUE)
# ==============================================================================
print("\n" + "="*50)
print(" PARTIE A : SCAN TEMPOREL (DYNAMIQUE)")
print("="*50)

# Paramètres Scan Temporel
delays = delays_ps * 1e-12

ratio_probe = 0.005 
amplitude_pump_ref = amplitude
amplitude_probe_ref = amplitude_pump_ref * ratio_probe

R_values = [] 
T_values = [] 

print("Calibration (Linéaire)...")

# Tir Sonde Seule (Référence linéaire)
pulse_probe_calib = amplitude_probe_ref * base_pulse
E_refl_calib= run_shot(pulse_probe_calib)

Fluence_in_calib = np.trapezoid(np.abs(pulse_probe_calib)**2, t)
Fluence_refl_calib = np.trapezoid(np.abs(E_refl_calib)**2, t)

R0 = Fluence_refl_calib / Fluence_in_calib

print(f" -> R0 : {R0*100:.2f} %")

# Tir Pompe Seule (Bruit de fond)
pulse_pump_only = amplitude_pump_ref * base_pulse
E_refl_pump = run_shot(pulse_pump_only)

print("Lancement du scan temporel...")
for i, delay in enumerate(delays):
    # Sonde décalée
    tau_probe = tau0 + delay
    pulse_probe_delayed = amplitude_probe_ref * np.exp(-((t-tau_probe)**2)/(2*sigma**2))
    
    pulse_total = pulse_pump_only + pulse_probe_delayed
    
    E_refl_total = run_shot(pulse_total)
    
    E_probe_refl = E_refl_total - E_refl_pump
    
    Fluence_refl = np.trapezoid(np.abs(E_probe_refl)**2, t)
    
    R_values.append(Fluence_refl / Fluence_in_calib)

R_values = np.array(R_values)
T_values = np.array(T_values)
Delta_R_sur_R = (R_values - R0) / np.max(R_values)

# --- Plot Temporel ---
fig1, ax1 = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
ax1[0].plot(delays_ps, Delta_R_sur_R * 100, 'b-o', label='Reflexion')
ax1[0].set_ylabel(r'$\Delta R / R_{max}$ (%)')
ax1[0].set_title('Dynamique de Relaxation')
ax1[0].grid(True, alpha=0.3)

ax1[1].plot(delays_ps, R_values * 100, 'g-s', label='Reflexion')
ax1[1].set_ylabel(r'$R$ (%)')
ax1[1].set_xlabel('Délai (ps)')
ax1[1].set_title('Dynamique de Relaxation')
ax1[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.plot()

results_time = pd.DataFrame({
    'delay_ps': delays_ps,
    'R': R_values,
})
csv_filename = f'{results_folder}data/temporal_scan_{amplitude:.2e}.csv'
results_time.to_csv(csv_filename, index=False, sep=';', decimal=',')
print(f"Données sauvegardées sous : {csv_filename}")

plt.savefig(f'{results_folder}figure/timescan_{amplitude:.2e}.png')
print(f"\nGraphique sauvegardé sous '{results_folder}figure/timescan_{amplitude:.2e}.png'")

# ==============================================================================
# PARTIE B : SCAN EN FLUENCE (COURBE DE SATURATION)
# ==============================================================================
print("\n" + "="*50)
print(" PARTIE B : SCAN EN FLUENCE (COURBE DE SATURATION)")
print("="*50)

# On se place au délai optimal (là où la modulation est maximale dans le scan précédent)
idx_max_mod = np.argmax(np.abs(Delta_R_sur_R))
delay_optimal = 0 #delays[idx_max_mod]
print(f"Délai fixé pour le scan : {delay_optimal*1e12:.2f} ps (Pic de modulation)")

fluences_input = [] # Pour l'axe X
max_delta_R_values = [] # Pour l'axe Y (Modulation)
R_sat_curr_list = []

print("Lancement du scan en puissance...")

for i, amp_pump_val in enumerate(amplitudes_scan):
    current_pump = amp_pump_val * base_pulse
    S_t = epsilon0 * c * np.abs(current_pump)**2
    fluence_curr = np.trapezoid(S_t,t)
    fluences_input.append(fluence_curr)
    
    E_refl_pump_curr = run_shot(current_pump)
    
    amp_probe_curr = amp_pump_val * 0.005 
    current_probe = amp_probe_curr * np.exp(-((t-(tau0+delay_optimal))**2)/(2*sigma**2))
    
    pulse_tot_curr = current_pump + current_probe
    E_refl_tot_curr = run_shot(pulse_tot_curr)
    
    E_probe_extracted = E_refl_tot_curr - E_refl_pump_curr
    Fluence_refl_probe = np.trapezoid(np.abs(E_probe_extracted)**2, t)
    
    Fluence_in_probe_curr = np.trapezoid(np.abs(current_probe)**2, t)
    
    R_sat_curr = Fluence_refl_probe / Fluence_in_probe_curr
    
    R_sat_curr_list.append(R_sat_curr)
    
    print(f"   Step {i+1}/{len(amplitudes_scan)} : Fluence={fluence_curr:.2e}", end="\r")

print("\nScan terminé.")

# --- Plot Saturation ---
fig2, ax2 = plt.subplots(figsize=(8, 6))

x_axis = np.array(fluences_input)
y_axis = np.array(R_sat_curr_list) * 100 # en %

ax2.semilogx(x_axis*1e3, y_axis, 'r-o', linewidth=2, markersize=8)
ax2.set_xlabel('Fluence d\'entrée (mJ/m²)')
ax2.set_ylabel(r'Réflectivité (%)')
ax2.set_title(f'Courbe de Saturation du SESAM (Délai fixé à {delay_optimal*1e12:.1f} ps)')
ax2.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.plot()

fig2, ax2 = plt.subplots(figsize=(8, 6))

x_axis_2 = x_axis * F_repetition * (np.pi * R**2)*0.5

ax2.semilogx(x_axis_2*1e3, y_axis, 'r-o', linewidth=2, markersize=8)
ax2.set_xlabel('Puissance d\'entrée moyenne (mW)')
ax2.set_ylabel(r'Réflectivité (%)')
ax2.set_title(f'Courbe de Saturation du SESAM (Délai fixé à {delay_optimal*1e12:.1f} ps)')
ax2.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.plot()

plt.savefig(f'{results_folder}figure/ampscan_{amplitude:.2e}.png')
print(f"\nGraphique sauvegardé sous '{results_folder}figure/ampscan_{amplitude:.2e}.png'")

plt.show()


# Enregistrement des résultats dans un fichier CSV
results_df = pd.DataFrame({
    'pump power': x_axis_2*1e3,
    'fluence': x_axis,
    'R':y_axis,
})

csv_filename = f'{results_folder}data/pump_probe_simulation_{amplitude:.2e}.csv'
results_df.to_csv(csv_filename, index=False, sep=';', decimal=',')
print(f"Données sauvegardées sous : {csv_filename}")
