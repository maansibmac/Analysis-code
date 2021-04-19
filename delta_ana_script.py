# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from classes_text import *
import os 
from classes_tharnal import * 


use_data = ['test_19032021_2']


folder_name = 'test_19032021_2'
pattern = f'mol_trial.*\.hdf5$'

patternc = re.compile(pattern)
names = []

for filename in os.listdir(f'/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/{folder_name}/videos/'):
    if patternc.match(filename):
        name, form = filename.split('.')
        names.append(name)
    else:
        continue

names.sort(key=natural_keys)
    
delta_list = []

for i, n in enumerate(names):
    dat_im = ReAnRaw(f'/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/{folder_name}/videos/{n}')
    dat_im.datatoDic()
    dat_im.extractMeans()
    baseline_means = dat_im.means
        
    dat_im.extractOpenClose('stimulus')
    baseline = np.mean(dat_im.means[:(dat_im.open[0] + 1)])

    dat_im.extractMeans(name_coor='diff_coor')
    diff_means = dat_im.means
    threshold = dat_im.means[-1] 


    delta_indv = baseline - threshold

    if delta_indv > 0.2:
        delta_list.append(delta_indv)
            

print(len(delta_list))

# %%
yax = np.arange(1, len(delta_list) + 1, 1)

fig, ax = plt.subplots()

ax.plot(yax, delta_list, marker = '.', markerfacecolor = 'k', markeredgecolor = 'k', color = 'steelblue')
    
ax.set_ylim(0, 2)

ax.set_xlabel('MoL Trials', fontsize = 14)
ax.set_ylabel('Delta T (Degrees Celsius)', fontsize = 14)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# %%
