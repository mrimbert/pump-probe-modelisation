import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Dossier des résultats [doit être créé au préalable !]
results_folder = "results/" #Dossier de sauvegarde des résultats, doit se terminer par "/"

csv_simulation_filename = f'{results_folder}data/pump_probe_simulation_3.20e+07.csv'
data_simulation = pd.read_csv(csv_simulation_filename, sep=';', decimal=',')

csv_experience_filename = 'comparaison/data_experimental/401ExperimentalPoint.csv'
data_experience = pd.read_csv(csv_experience_filename, sep=';', decimal=',')

Ep = data_experience['Pump power']/80e6
F_experience_5 = Ep/(np.pi*5e-6**2)
F_experience_10 = Ep/(np.pi*10e-6**2)
F_center = (F_experience_10 + F_experience_5) / 2
F_err = (F_experience_5 - F_experience_10) / 2


plt.figure(figsize=(3.37, 2.5), constrained_layout=True)

plt.semilogx(data_simulation['fluence']*1e3,data_simulation['R']/100, 
             linewidth=2, 
             label="Simulation\ndata")



plt.errorbar(
    F_center,                               
    data_experience['Reflectivity'],
    xerr=F_err,                             
    fmt='o',
    markersize=0,
    elinewidth=1.5,
    capsize=2,
    capthick=1,
    label="Experimental\ndata"
)
plt.xscale('log')

plt.legend(fontsize=9, loc="lower right")
plt.xlabel('Pump fluence (mJ/m²)', fontsize=9)
plt.ylabel(r'Maximum reflectivity', fontsize=9)
plt.grid(True, which="both", ls="-", alpha=0.3)

plt.savefig(f"{results_folder}comparaison/RoverPump.png", dpi=600)

plt.show()