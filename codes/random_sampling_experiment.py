#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 06:43:55 2024

@author: rickerr2

The purpose of this script is to imitate how our technology would be used in real-life;
i.e. we wouldn't be taking 400 spectra per patient, we would take like 10 and then use 
those 10 spectra to determine which virus the patient had.
"""

#%%

# import libraries

import os
import pandas as pd # v 1.3.5
import numpy as np # v 1.21.2

# python version 3.7.11

#%%

# load data
# load the class probabilities produced by the model.predict on the test class
folder = '/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/results'

h05_532 = pd.DataFrame(pd.read_csv(os.path.join(folder,'532nm_h05_class_probabilities.csv'), index_col = 0, encoding= 'unicode_escape'))
h10_532 = pd.DataFrame(pd.read_csv(os.path.join(folder,'532nm_h10_class_probabilities.csv'), index_col = 0, encoding= 'unicode_escape'))
h05_785 = pd.DataFrame(pd.read_csv(os.path.join(folder,'785nm_h05_class_probabilities.csv'), index_col = 0, encoding= 'unicode_escape'))
h10_785 = pd.DataFrame(pd.read_csv(os.path.join(folder,'785nm_h10_class_probabilities.csv'), index_col = 0, encoding= 'unicode_escape'))

n05_532 = pd.DataFrame(pd.read_csv(os.path.join(folder,'532nm_n05_class_probabilities.csv'), index_col = 0, encoding= 'unicode_escape'))
n10_532 = pd.DataFrame(pd.read_csv(os.path.join(folder,'532nm_n10_class_probabilities.csv'), index_col = 0, encoding= 'unicode_escape'))
n05_785 = pd.DataFrame(pd.read_csv(os.path.join(folder,'785nm_n05_class_probabilities.csv'), index_col = 0, encoding= 'unicode_escape'))
n10_785 = pd.DataFrame(pd.read_csv(os.path.join(folder,'785nm_n10_class_probabilities.csv'), index_col = 0, encoding= 'unicode_escape'))
  

#%%

# put files into lists to run through

files785 = [n05_785, n10_785, h05_785, h10_785]
files532 = [n05_532, n10_532, h05_532, h10_532]
sample_names = ['H1N1 - 2.33E07 copies/mL', 'H1N1 - 9.91E06 copies/mL',
                'H3N2 - 1.67E06 copies/mL', 'H3N2 - 3.36E05 copies/mL']

#%%

# For the 785 Raman spectrometer

simulation_predictions = [] # make empty list
for file in files785: # for each of the 4 files in the 785 nm list
    h1n1 = 0 # set to 0
    h3n2 = 0 # set to 0
    for i in range(0,100): # go through 100 iterations
        s = file.sample(n=10,random_state=i) # sample 10 random spectral probabilities that were made from the trained CNN
        pred = s.mean() # average the mean probability
        if pred[0] > 0.5: # if the average probability is >0,5
            h1n1 = h1n1 + 1 # classify it at h1n1 and add 1 to the counter
        else:
            h3n2 = h3n2 + 1 # else, classify it as h3n2 and add 1 to the counter
    simulation_predictions.append([h1n1, h3n2]) 
simulations785 = np.array(sum(simulation_predictions,[])).reshape((4,2)) # sum the predictions 
simulations785 = pd.DataFrame(simulations785)
simulations785.columns = ['# Simulations Predicted as H1N1', '# of Simulations Predicted as H3N2']
simulations785['Sample'] = ['H1N1', 'H1N1', 'H3N2', 'H3N2']
simulations785['Viral Copies/mL'] = ['2.33E07', '9.91E06',
                '1.67E06', '3.36E05']

print(simulations785)  

# save the features importances
#simulations785.to_csv('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /results/785_sample_simulation_results.csv')


