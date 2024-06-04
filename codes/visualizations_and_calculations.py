#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:31:36 2024

@author: rickerr2

Title: Visualizations and Spectral Correlation/Mean Comparison


This script is used to make figures of the spectra, training loss, feature importances, 
and the probability distributions of the testing sets as well as run some of the stats


"""


#%%

# import libraries
import os
import pandas as pd # v 1.3.5
import numpy as np # v 1.21.2
import matplotlib.pyplot as plt # v 3.1.0
from matplotlib.pyplot import cm
import seaborn as sns # v 0.11.2
from scipy import stats # v 1.7.3

# python version 3.7.11


#%%

'''
### SPECTRA FIGURES ###
'''

#%%

# load the nebraska 785 nm
folder = '/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/data/785nm/preprocessed/nebraska'
files = os.listdir(folder)
nebraska785_files = [] # blank list
nebraska785_names = []
counter = 0

for file in sorted(files):
    if file.endswith('.txt'): # only get the text files
        df = pd.DataFrame(pd.read_csv(os.path.join(folder,file), index_col = 0, encoding= 'unicode_escape'))
        nebraska785_files.append(df.iloc[:,:])
        nebraska785_names.append(file[9:-17])
        print(file)
        counter = counter + 1
print('')
print('The numer of samples in the list is', counter)

#%%

# load the hawaii 785 nm
folder = '/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/data/785nm/preprocessed/hawaii'
files = os.listdir(folder)
hawaii785_files = [] # blank list
hawaii785_names = []
counter = 0

for file in sorted(files):
    if file.endswith('.txt'): # only get the text files
        df = pd.DataFrame(pd.read_csv(os.path.join(folder,file), index_col = 0))
        hawaii785_files.append(df.iloc[:,:])
        hawaii785_names.append(file[7:-17])
        print(file)
        counter = counter + 1
print('')
print('The numer of samples in the list is', counter)

#%%

# load the nebraska 532 nm
folder = '/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/data/532nm/preprocessed/nebraska'
files = os.listdir(folder)
nebraska532_files = [] # blank list
nebraska532_names = []
counter = 0

for file in sorted(files):
    if file.endswith('.txt'): # only get the text files
        df = pd.DataFrame(pd.read_csv(os.path.join(folder,file), index_col = 0, encoding= 'unicode_escape'))
        nebraska532_files.append(df.iloc[:,:])
        nebraska532_names.append(file[9:-17])
        print(file)
        counter = counter + 1
print('')
print('The numer of samples in the list is', counter)

#%%

# load the hawaii 532 nm
folder = '/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/data/532nm/preprocessed/hawaii'
files = os.listdir(folder)
hawaii532_files = [] # blank list
hawaii532_names = []
counter = 0

for file in sorted(files):
    if file.endswith('.txt'): # only get the text files
        df = pd.DataFrame(pd.read_csv(os.path.join(folder,file), index_col = 0))
        hawaii532_files.append(df.iloc[:,:])
        hawaii532_names.append(file[7:-17])
        print(file)
        counter = counter + 1
print('')
print('The numer of samples in the list is', counter)

#%%

# get mean of each sybtype of each laser wavelength

nebraska785_concat = pd.concat(nebraska785_files, axis = 1, ignore_index=True)
hawaii785_concat = pd.concat(hawaii785_files, axis = 1, ignore_index=True)
nebraska532_concat = pd.concat(nebraska532_files, axis = 1, ignore_index=True)
hawaii532_concat = pd.concat(hawaii532_files, axis = 1, ignore_index=True)


#%%

# calculate the correlation coefficients for each laser wavelength

pearsons_785 = stats.pearsonr(nebraska785_concat.mean(axis=1), hawaii785_concat.mean(axis=1))
pearsons_532 = stats.pearsonr(nebraska532_concat.mean(axis=1), hawaii532_concat.mean(axis=1))
pearsons_785
# (0.8312471983927399, 1.555695874627832e-138)
pearsons_532
# (0.968605179735789, 1.919655032377444e-284)

spearman_785 = stats.spearmanr(nebraska785_concat.mean(axis=1), hawaii785_concat.mean(axis=1))
spearman_532 = stats.spearmanr(nebraska532_concat.mean(axis=1), hawaii532_concat.mean(axis=1))
spearman_785
# SpearmanrResult(correlation=0.8723956694689, pvalue=1.914065403280365e-168)
spearman_532
# SpearmanrResult(correlation=0.948920466640378, pvalue=4.288439133248499e-236)


#%%

# perform t test between the all the samples collected at 785 nm at each wavenumber

l = nebraska785_concat.shape[0]
ps = [] # list of p values
a = 0 # number of significant wavenumbers
for i in range(nebraska785_concat.shape[0]):
    s,p = stats.ttest_ind(nebraska785_concat.iloc[i,:], hawaii785_concat.iloc[i,:], equal_var=False)
    if p < (0.05/l):
        a = a + 1

    
# proportion of statistically significant wavenumbers
n = a/l
print('The proportion of statistically significant wavenumbers at 785 nm is: ', n)
# The proportion of statistically significant wavenumbers at 785 nm is:  0.9273743016759777

#%%

# perform t test between the all the samples collected at 532 nm at each wavenumber


l = nebraska532_concat.shape[0]
a = 0 # number of significant wavenumbers
for i in range(nebraska532_concat.shape[0]):
    s,p = stats.ttest_ind(nebraska532_concat.iloc[i,:], hawaii532_concat.iloc[i,:], equal_var=False)
    if p < (0.05/l): # by dividing 0.05 by the length, this is a Bonferroni correction to account for many t-tests.
        a = a + 1
        
# proportion of statistically significant wavenumbers
n = a/l
print('The proportion of statistically significant wavenumbers at 532 nm is: ', n)
# The proportion of statistically significant wavenumbers at 532 nm is:  0.837953091684435      
 


#%%

# get the descriptive statistics (mean and stnd dev) of each subtype at each laser wavelength

nebraska785_concat['mean'] = nebraska785_concat.mean(axis=1)
nebraska785_concat['std'] = nebraska785_concat.std(axis=1)

hawaii785_concat['mean'] = hawaii785_concat.mean(axis=1)
hawaii785_concat['std'] = hawaii785_concat.std(axis=1)

nebraska532_concat['mean'] = nebraska532_concat.mean(axis=1)
nebraska532_concat['std'] = nebraska532_concat.std(axis=1)

hawaii532_concat['mean'] = hawaii532_concat.mean(axis=1)
hawaii532_concat['std'] = hawaii532_concat.std(axis=1)
#%%

# calculate the average mean between the sets of spectra on each spectrometer

#%%


# plot the average of the positive and the negative
 
# plot the trimmed spectra
#plt.figure(figsize=(10,10))

fig = plt.figure(figsize=(5,3))

gs = fig.add_gridspec(2,4,height_ratios=[2,1])


plt.subplots_adjust(hspace=1, wspace = 1)

ax1 = fig.add_subplot(gs[0,:2,])
palette = sns.husl_palette(2)
plt.plot(nebraska785_concat.index, nebraska785_concat.loc[:,'mean']+0.25, label = 'A/H1N1', color = palette[1])
#plt.fill_between(nebraska785_concat.index, nebraska785_concat.loc[:,'mean']+0.5-nebraska785_concat.loc[:,'std'], nebraska785_concat.loc[:,'mean']+0.5+nebraska785_concat.loc[:,'std'], color = palette[1], alpha=0.2)
plt.plot(hawaii785_concat.index, hawaii785_concat.loc[:,'mean'], label = 'A/H3N2', color = palette[0])
#plt.fill_between(hawaii785_concat.index, hawaii785_concat.loc[:,'mean']-hawaii785_concat.loc[:,'std'], hawaii785_concat.loc[:,'mean']+hawaii785_concat.loc[:,'std'], color = palette[0], alpha=0.2)
#plt.text(-0.1, 1.4, 'A', size = 10, weight = 'bold')
plt.legend()
plt.tick_params(left = True, right = False , labelleft = True , 
                labelbottom = True, bottom = True) 


plt.yticks(fontname= "Arial", fontsize=8)
plt.ylim(None, 1.25)
plt.xticks(fontname= "Arial", fontsize=8)
plt.ylabel("Normalized Intensity", fontname= "Arial", fontsize=10, labelpad=0.5)
plt.xlabel('Wavenumber ($cm^{-1}$)', fontsize=10, fontname="Arial", labelpad=0.5)

plt.legend(loc='upper left', frameon=False, fontsize=8, ncol=1)#bbox_to_anchor = (1.02,0.1), loc = 'upper left', frameon=False, fontsize=10)#fontname= "Arial", fontsize=14)
sns.despine()

plt.subplots_adjust(hspace=1, wspace = 0.2)

# get the means of each awvenumber for each sample
nebraska785_means = []

for file in nebraska785_files:
    file['mean'] = file.mean(axis=1)
    nebraska785_means.append(file)
    
hawaii785_means = []

for file in hawaii785_files:
    file['mean'] = file.mean(axis=1)
    hawaii785_means.append(file)
    
ax2 = fig.add_subplot(gs[1,0])  

color = iter(cm.GnBu(np.linspace(0,1,10)))
counter = 0
for i in range(len(nebraska785_means)):
    c = next(color)
    plt.plot(nebraska785_means[i].index, nebraska785_means[i].loc[:,'mean'], label = nebraska785_names[counter], c=c)
    counter = counter + 1
plt.tick_params(left = False, right = False , labelleft = False, 
                labelbottom = False, bottom = False) 
#plt.yticks(fontname= "Arial", fontsize=8)
plt.ylabel("Normalized\nIntensity", fontname= "Arial", fontsize=8)
plt.xticks(fontname= "Arial", fontsize=8)
plt.xlabel('Wavenumber', fontsize=8, fontname="Arial")

sns.despine()


ax3 = fig.add_subplot(gs[1,1])

color = iter(cm.RdPu(np.linspace(0,1,10)))

counter = 0
for i in range(len(hawaii785_means)):
    c = next(color)
    plt.plot(hawaii785_means[i].index, hawaii785_means[i].loc[:,'mean'], label = hawaii785_names[counter], c=c)
    counter = counter + 1
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

plt.yticks(fontname= "Arial", fontsize=8)
plt.xticks(fontname= "Arial", fontsize=8)
plt.xlabel('Wavenumber', fontsize=8, fontname="Arial")

#plt.ylabel("Normalized Intensity", fontname= "Arial", fontsize=8)

sns.despine()



#fig.tight_layout(pad=0.3)


# SECOND COLUMNS - 532 nm

ax1 = fig.add_subplot(gs[0,2:])
palette = sns.husl_palette(2)
plt.plot(nebraska532_concat.index, nebraska532_concat.loc[:,'mean']+0.25, label = 'A/H1N1', color = palette[1])
#plt.fill_between(nebraska532_concat.index, nebraska532_concat.loc[:,'mean']+0.5-nebraska532_concat.loc[:,'std'], nebraska532_concat.loc[:,'mean']+0.5+nebraska532_concat.loc[:,'std'], color = palette[1], alpha=0.2)
plt.plot(hawaii532_concat.index, hawaii532_concat.loc[:,'mean'], label = 'A/H3N2', color = palette[0])
#plt.fill_between(hawaii532_concat.index, hawaii532_concat.loc[:,'mean']-hawaii532_concat.loc[:,'std'], hawaii532_concat.loc[:,'mean']+hawaii532_concat.loc[:,'std'], color = palette[0], alpha=0.2)
#plt.text(-0.1, 1.4, 'B', size = 10, weight = 'bold')

plt.legend()
plt.tick_params(left = False, right = False , labelleft = False, 
                labelbottom = True, bottom = True) 


#plt.yticks(fontname= "Arial", fontsize=8)
plt.ylim(None, 1.25)
plt.xticks(fontname= "Arial", fontsize=8)
#plt.ylabel("Normalized Intensity", fontname= "Arial", fontsize=10, labelpad=0.5)
plt.xlabel('Wavenumber ($cm^{-1}$)', fontsize=10, fontname="Arial", labelpad=0.5)
plt.legend(loc='upper left', frameon=False, fontsize=8, ncol=1)#bbox_to_anchor = (1.02,0.1), loc = 'upper left', frameon=False, fontsize=10)#fontname= "Arial", fontsize=14)
sns.despine()



# get the means of each awvenumber for each sample
nebraska532_means = []
for file in nebraska532_files:
    file['mean'] = file.mean(axis=1)
    nebraska532_means.append(file)    
hawaii532_means = []
for file in hawaii532_files:
    file['mean'] = file.mean(axis=1)
    hawaii532_means.append(file)
    
ax2 = fig.add_subplot(gs[1,2])  



color = iter(cm.GnBu(np.linspace(0,1,10)))
counter = 0
for i in range(len(nebraska532_means)):
    c = next(color)
    plt.plot(nebraska532_means[i].index, nebraska532_means[i].loc[:,'mean'], label = nebraska532_names[counter], c=c)
    counter = counter + 1
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
#plt.yticks(fontname= "Arial", fontsize=8)
#plt.ylabel("Normalized\nIntensity", fontname= "Arial", fontsize=8)
plt.xticks(fontname= "Arial", fontsize=8)
plt.xlabel('Wavenumber', fontsize=8, fontname="Arial")
sns.despine()


ax3 = fig.add_subplot(gs[1,3])

color = iter(cm.RdPu(np.linspace(0,1,10)))

counter = 0
for i in range(len(hawaii532_means)):
    c = next(color)
    plt.plot(hawaii532_means[i].index, hawaii532_means[i].loc[:,'mean'], label = hawaii532_names[counter], c=c)
    counter = counter + 1
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
plt.xticks(fontname= "Arial", fontsize=6)
plt.xticks(fontname= "Arial", fontsize=8)
plt.xlabel('Wavenumber', fontsize=8, fontname="Arial")

#plt.xlabel('Wavenumber ($cm^{-1}$)', fontsize=8, fontname="Arial")
#plt.ylabel("Normalized Intensity", fontname= "Arial", fontsize=8)

# plot the sub figure portion and the laser wavelength

fig.text(0, 0.96,  
         'A',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10) 

fig.text(0.5, 0.96,  
         'B',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10)
fig.text(0, 0.35,  
         'C',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10) 

fig.text(0.5, 0.35,  
         'D',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10)

fig.text(0.25, 0.93,  
         '785 nm',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10) 

fig.text(0.67, 0.93,  
         '532 nm',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10)


sns.despine()

#plt.savefig('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /figures/Spectra.png', dpi=1000, bbox_inches = "tight") 
#plt.savefig('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /figures/Spectra.pdf', dpi=1000, bbox_inches = "tight") 



plt.show()




#%%

# find the peak maximums of the virus positive and virus negative
# for virus positive
print('The wavenumber at which the max peak occurs for H1N1 (Nebraska) at 785 nm is: ',nebraska785_concat.index[np.argmax(nebraska785_concat.loc[:,'mean'])])
# 1298.42
print('The wavenumber at which the max peak occurs for H3N2 (Hawaii) at 785 nm is: ',hawaii785_concat.index[np.argmax(hawaii785_concat.loc[:,'mean'])])
# 1062.13
print('The wavenumber at which the max peak occurs for H1N1 (Nebraska) at 532 nm is: ',nebraska532_concat.index[np.argmax(nebraska532_concat.loc[:,'mean'])])
# 1590.94
print('The wavenumber at which the max peak occurs for H3N2 (Hawaii) at 532 nm is: ',hawaii532_concat.index[np.argmax(hawaii532_concat.loc[:,'mean'])])
# 1590.94

#%%

'''
### LOSS FIGURES ###
'''

#%%

# load data
# each column is a fold
train785_loss = pd.DataFrame(pd.read_csv('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/results/785nm_training_loss.csv', index_col = 0, encoding= 'unicode_escape'))
train532_loss = pd.DataFrame(pd.read_csv('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/results/532nm_training_loss.csv', index_col = 0, encoding= 'unicode_escape'))

#%%

# plot the training loss for each fold at each wavelength
x = np.arange(1,101,1)

fig = plt.figure(figsize=(4,1.25))

gs = fig.add_gridspec(2,2)
#plt.subplots_adjust(left=0.125)
#
ax1 = fig.add_subplot(gs[:,0])
color = iter(cm.ocean(np.linspace(0,0.8,8)))
counter = 0
for i in range(8):
    c = next(color)
    plt.plot(x, train785_loss.iloc[:,i], label = 'Fold '+str(i+1), c=c)
    counter = counter + 1
plt.ylim(0,0.8)


#
#plt.text(0.9, 1.4, 'D', size = 10, weight = 'bold')
#plt.legend()
plt.xticks(fontname= "Arial", fontsize=8)
plt.yticks(fontname= "Arial", fontsize=8)
plt.xlabel('Epochs', fontsize=10, fontname="Arial")
plt.ylabel("Loss", fontname= "Arial", fontsize=10)
#plt.text(-0.5, 0.9, 'A', size = 10, weight = 'bold')
#plt.gcf().text(-0.01, 1.0, 'A', fontsize=10, weight='bold')

plt.subplots_adjust(wspace=0.3)

ax2 = fig.add_subplot(gs[:,1])
color = iter(cm.ocean(np.linspace(0,0.8,8)))
counter = 0
for i in range(8):
    c = next(color)
    ax2.plot(x, train532_loss.iloc[:,i], label = 'Fold '+str(i+1), c=c)
    counter = counter + 1
ax2.legend()
plt.xticks(fontname= "Arial", fontsize=8)
plt.yticks(fontname= "Arial", fontsize=8)
plt.xlabel('Epochs', fontsize=10, fontname="Arial")
#plt.ylabel("Loss", fontname= "Arial", fontsize=10)
#plt.gcf().text(0.45, 1.0, 'B', fontsize=10, weight='bold')

ax2.legend(loc='right', bbox_to_anchor = (1.6,0.5),frameon=False, fontsize=7, ncol=1)#bbox_to_anchor = (1.02,0.1), loc = 'upper left', frameon=False, fontsize=10)#fontname= "Arial", fontsize=14)

plt.ylim(0,0.8)

fig.text(0.23, 0.93,  
         '785 nm',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10) 

fig.text(0.68, 0.93,  
         '532 nm',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10)

sns.despine()

#plt.savefig('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /figures/TrainingLoss.png', dpi=1000, bbox_inches = "tight") 
#plt.savefig('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /figures/TrainingLoss.pdf', dpi=1000, bbox_inches = "tight") 



plt.show()


#%%
'''
### CLASSIFICATION PROBABILITY DISTRIBUTIONS PER SAMPLE ###
'''

#%%
# load the data
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
# round the probabilities to the nearest percentage

# round the probabilities to the nearest percentage
files = [h05_532, h10_532, h05_785, h10_785,
         n05_532, n10_532, n05_785, n10_785] 

 


#%%

fig = plt.figure(figsize=(6,2))

fig.tight_layout(pad=2.5)

plt.subplots_adjust(hspace=0.6)

plt.subplot(2,4,1)
arr1 = plt.hist(n05_785.iloc[:,0], weights = (np.ones(n05_785.shape[0])/n05_785.shape[0]),bins = 10, color='teal', edgecolor="navy")
plt.xticks(fontname= "Arial", fontsize=8)
plt.yticks(fontname= "Arial", fontsize=8)
#plt.xlabel('Probability of Belonging\nto Class = H1N1', fontsize=10, fontname="Arial")
plt.ylabel("Proportion", fontname= "Arial", fontsize=8)
plt.title('2.33E+07 copies/mL', fontsize=8, fontname="Arial")#, weight='bold')
plt.ylim(0,1.0)

# get the proportions with less than 10 percent prob and more than 90
less_10_h1n1_n5_785 = arr1[0][0]
# 0.09000000000000004
more_90_h1n1_n5_785 = arr1[0][-1]
# 0.8374999999999931

#for i in range(10):
#    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))


plt.subplot(2,4,2)
arr2 = plt.hist(n10_785.iloc[:,0], weights = (np.ones(n10_785.shape[0])/n10_785.shape[0]),bins = 10, color='teal', edgecolor="navy")
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = True, bottom = True) 
plt.xticks(fontname= "Arial", fontsize=8)
#plt.yticks(fontname= "Arial", fontsize=8)
#plt.xlabel('Probability of Belonging\nto Class = H1N1', fontsize=10, fontname="Arial")
#plt.ylabel("Frequency", fontname= "Arial", fontsize=10)
plt.title('9.91E+06 copies/mL', fontsize=8, fontname="Arial")#, weight='bold')
plt.ylim(0,1.0)

# get the proportions with less than 10 percent prob and more than 90
less_10_h1n1_n10_785 = arr2[0][0]
# 0.19250000000000012
more_90_h1n1_n10_785 = arr2[0][-1]
# 0.657499999999997



plt.subplot(2,4,3)
arr3 = plt.hist(h05_785.iloc[:,0], weights = (np.ones(h05_785.shape[0])/h05_785.shape[0]),bins = 10, color='rebeccapurple', edgecolor="navy")
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = True, bottom = True) 
plt.xticks(fontname= "Arial", fontsize=8)
plt.yticks(fontname= "Arial", fontsize=8)
#plt.xlabel('Probability of Belonging\nto Class = H1N1', fontsize=10, fontname="Arial")
#plt.ylabel("Frequency", fontname= "Arial", fontsize=10)
plt.title('1.67E+06 copies/mL', fontsize=8, fontname="Arial")#, weight='bold')
plt.ylim(0,1.0)

# get the proportions with less than 10 percent prob and more than 90
less_10_h3n2_h5_785 = arr3[0][0]
# 0.7199999999999956
more_90_h3n2_h5_785 = arr3[0][-1]
# 0.20000000000000012

plt.subplot(2,4,4)
arr4 = plt.hist(h10_785.iloc[:,0], weights = (np.ones(h10_785.shape[0])/h10_785.shape[0]),bins = 10, color='rebeccapurple', edgecolor="navy")
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = True, bottom = True) 
plt.xticks(fontname= "Arial", fontsize=8)
plt.yticks(fontname= "Arial", fontsize=8)
#plt.xlabel('Probability of Belonging\nto Class = H1N1', fontsize=10, fontname="Arial")
#plt.ylabel("Frequency", fontname= "Arial", fontsize=10)
plt.title('3.36E+05 copies/mL', fontsize=8, fontname="Arial")#, weight='bold')
plt.ylim(0,1.0)

# get the proportions with less than 10 percent prob and more than 90
less_10_h3n2_h10_785 = arr4[0][0]
# 0.7724999999999945
more_90_h3n2_h10_785 = arr4[0][-1]
# 0.11500000000000006


### BOTTOM ROW

plt.subplot(2,4,5)
arr5 = plt.hist(n05_532.iloc[:,0], weights = (np.ones(n05_532.shape[0])/n05_532.shape[0]),bins = 10, color='teal', edgecolor="navy")
plt.xticks(fontname= "Arial", fontsize=8)
plt.yticks(fontname= "Arial", fontsize=8)
plt.xlabel('Probability of \nClass = H1N1', fontsize=8, fontname="Arial")
plt.ylabel("Frequency", fontname= "Arial", fontsize=8)
#plt.title('H1N1 - 2.33E+07 copies/mL', fontsize=10, fontname="Arial")
plt.ylim(0,1.0)

# get the proportions with less than 10 percent prob and more than 90
less_10_h1n1_n5_532 = arr5[0][0]
# 0.1725000000000001
more_90_h1n1_n5_532 = arr5[0][-1]
# 0.6449999999999972



plt.subplot(2,4,6)
arr6 = plt.hist(n10_532.iloc[:,0], weights = (np.ones(n10_532.shape[0])/n10_532.shape[0]),bins = 10, color='teal', edgecolor="navy")
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = True, bottom = True) 
plt.xticks(fontname= "Arial", fontsize=8)
plt.yticks(fontname= "Arial", fontsize=8)
plt.xlabel('Probability of \nClass = H1N1', fontsize=8, fontname="Arial")
#plt.ylabel("Frequency", fontname= "Arial", fontsize=10)
#plt.title('H1N1 - 9.91E+06 copies/mL', fontsize=10, fontname="Arial")
plt.ylim(0,1.0)

sns.despine()

# get the proportions with less than 10 percent prob and more than 90
less_10_h1n1_n10_532 = arr6[0][0]
# 0.24250000000000016
more_90_h1n1_n10_532 = arr6[0][-1]
# 0.5824999999999986


plt.subplot(2,4,7)
arr7 = plt.hist(h05_532.iloc[:,0], weights = (np.ones(h05_532.shape[0])/h05_532.shape[0]),bins = 10, color='rebeccapurple', edgecolor="navy")
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = True, bottom = True) 
plt.xticks(fontname= "Arial", fontsize=8)
plt.yticks(fontname= "Arial", fontsize=8)
plt.xlabel('Probability of \nClass = H1N1', fontsize=8, fontname="Arial")
#plt.ylabel("Frequency", fontname= "Arial", fontsize=10)
#plt.title('H3N2 - 1.67E+06 copies/mL', fontsize=10, fontname="Arial")
plt.ylim(0,1.0)

# get the proportions with less than 10 percent prob and more than 90
less_10_h3n2_h5_532 = arr7[0][0]
# 0.515
more_90_h3n2_h5_532 = arr7[0][-1]
# 0.2575000000000002

plt.subplot(2,4,8)
arr8 = plt.hist(h10_532.iloc[:,0], weights = (np.ones(h10_532.shape[0])/h10_532.shape[0]),bins = 10, color='rebeccapurple', edgecolor="navy")
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = True, bottom = True) 
plt.xticks(fontname= "Arial", fontsize=8)
plt.yticks(fontname= "Arial", fontsize=8)
plt.xlabel('Probability of \nClass = H1N1', fontsize=8, fontname="Arial")
#plt.ylabel("Frequency", fontname= "Arial", fontsize=10)
#plt.title('H3N2 - 3.36E+05 copies/mL', fontsize=10, fontname="Arial")
plt.ylim(0,1.0)


sns.despine()

# get the proportions with less than 10 percent prob and more than 90
less_10_h3n2_h10_532 = arr8[0][0]
# 0.937499999999991
more_90_h3n2_h10_532 = arr8[0][-1]
# 0.029999999999999995

#plt.text(1, 0, 'Class = H1N1', fontsize=12, fontname='Arial', ha='center', va='top')

fig.text(0.25, 0.98,  
         'Class = H1N1',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10) 

fig.text(0.65, 0.98,  
         'Class = H3N2',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10) 

fig.text(0.02, 0.64,  
         '785nm',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10,
         rotation=90) 
fig.text(0.02, 0.23,  
         '532nm',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10,
         rotation=90) 

#plt.savefig('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /figures/predicted_probabilities.png', dpi=1000, bbox_inches = "tight") 
#plt.savefig('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /figures/predicted_probabilities.pdf', dpi=1000, bbox_inches = "tight") 

#%%%

# calculate average correct classification with greater than 90% probability
# note that when the class is h1n1, use the more_90 and when class is h3n2 use less_10,
# as the prediction probability is based upon it belonging to class h1n1 

correct_and_high_prob_785 = (more_90_h1n1_n5_785+more_90_h1n1_n10_785+less_10_h3n2_h5_785+less_10_h3n2_h10_785)/4
# 0.7468749999999951
correct_and_high_prob_532 = (more_90_h1n1_n5_532+more_90_h1n1_n10_532+less_10_h3n2_h5_532+less_10_h3n2_h10_532)/4
# 0.6699999999999967
#%%%

'''
### FEATURE IMPORTANCE FIGURES ###
'''

#%%

# load data

rf_features_785 = pd.DataFrame(pd.read_csv('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/results/785nm_randomforest_feature_importances.csv', index_col = 0, encoding= 'unicode_escape'))
rf_features_532 = pd.DataFrame(pd.read_csv('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/results/532nm_randomforest_feature_importances.csv', index_col = 0, encoding= 'unicode_escape'))

#%%

# get the 60 highest peaks and then find each of the clusters of points

# get the max peak points of feature importance
most_important_features_785 = rf_features_785.iloc[:,0].nlargest(n=60).index
most_important_features_785
#Float64Index([1291.81, 1313.83, 1294.02, 1302.83, 1456.91, 1285.19, 1311.63,
#              1307.23, 1289.61, 1316.02, 1305.03,  1287.4, 1309.43, 1282.98,
#              1296.22, 1454.77, 1298.43, 1670.81, 1444.06, 1664.64, 1300.63,
#              1672.86, 1662.58, 1411.79, 1459.05, 1441.91, 1666.69, 1452.63,
#              1668.75, 1431.18, 1409.64, 1660.52, 1674.91, 1471.87, 1433.33,
#               1474.0, 1685.17, 1280.78, 1318.22,  1446.2, 1681.07, 1429.03,
#              1687.22, 1658.46, 1420.42, 1326.99, 1329.18, 1476.13, 1422.57,
#              1683.12, 908.309, 1278.56, 1271.93, 1711.75, 910.671, 1426.87,
#              1713.79, 1337.94, 1276.35, 1641.95],
#             dtype='float64', name='0')

#%%
most_important_features_532 = rf_features_532.iloc[:,0].nlargest(n=60).index
most_important_features_532
#Float64Index([1120.65, 1125.82, 1128.41, 1123.23, 1639.65, 1642.08, 1130.99,
#              1118.06, 1138.75, 1649.36, 1141.33, 1115.47, 1136.16, 1112.88,
 #              1644.5, 1651.78, 1646.93, 798.693, 1156.81, 1133.58, 1154.23,
 #             796.003, 1637.22, 1151.65,  1524.7, 1099.91, 1654.21, 1659.05,
 #              1663.9, 1470.32, 1661.47, 1143.91,  1627.5, 1032.16, 1110.28,
 #             1527.16, 739.302, 1728.99, 1656.63, 1612.89, 1608.02,  742.01,
 #             1047.84, 1625.07, 1149.07, 1467.84, 1629.93, 1726.59, 1146.49,
 #             1622.63, 1546.84, 1522.23, 1610.46, 1632.36, 1034.78, 1617.76,
 #             1159.38, 1598.26, 1544.38, 1050.46],
 #            dtype='float64', name='0')

#%%
 
 # the clusters of feature importance for 785 nm were located around
 # 1291.81
 # 1454.77
 # 1670.81
 # 910.671
 
 # the clusters of feature importance for the 532 nm were located around
 
# 1120.65
# 1639.65
# 798.693
# 1524.7
# 1470.32
# 739.302

#%%

# plot the important features in which the clusters were found

#fig, ax1, ax3 = plt.subplots(1,2,1)
fig, (ax1,ax3) = plt.subplots(1,2, figsize=(7,2))

#fig = plt.figure(figsize=(6,2))
plt.subplots_adjust(wspace=0.6)
#fig.tight_layout(pad=1.0)


#  785 nm plot

x = np.asarray(rf_features_785.index,float)

ax2 = ax1.twinx()


ax2.plot(x, (nebraska785_concat.loc[:,'mean']+0.5), label = 'A/H1N1', color = 'dimgray')
ax2.plot(x, (hawaii785_concat.loc[:,'mean']+0.5), label = 'A/H3N2', color = 'darkgrey')
ax2.set_ylabel('Normalized Intensity', fontsize=10, fontname='Arial')

ax1.bar(x, rf_features_785.iloc[:,0], color='black', edgecolor="black")

ax1.set_ylabel('Decrease in Impurity', fontsize=10, fontname='Arial')
ax1.set_xlabel('Wavenumber ($cm^{-1}$)', fontsize=10, fontname="Arial")

ax1.set_ylim(0,0.03)
ax2.set_ylim(0,1.5)

ax1.xaxis.set_tick_params(labelsize=7)
ax1.yaxis.set_tick_params(labelsize=7)
ax2.yaxis.set_tick_params(labelsize=7)

sns.despine(top=True, right=False, left=False, bottom=False)

plt.grid(False)

# now plot the wavenumber peaks

ymin = 0.8
ymax = 0.85
fontsize = 8
rotation = 90
ax1.axvline(x = 1291.81, ymin = ymin, ymax = ymax, color='black')#, linewidth = 3)
ax1.text(x=1291.81, y = 0.029, s = "1292", fontname = "Arial", fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center', rotation=rotation)

ax1.axvline(x = 1454.77, ymin = ymin, ymax = ymax, color='black')#, linewidth = 3)
ax1.text(x=1454.77, y = 0.029, s = "1455", fontname = "Arial", fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center', rotation=rotation)

ax1.axvline(x = 1670.81, ymin = ymin, ymax = ymax, color='black')#, linewidth = 3)
ax1.text(x=1670.81, y = 0.029, s = "1671", fontname = "Arial", fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center', rotation=rotation)

ax1.axvline(x = 910.671, ymin = ymin, ymax = ymax, color='black')#, linewidth = 3)
ax1.text(x=910.671, y = 0.029, s = "911", fontname = "Arial", fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center', rotation=rotation)




# now 532 nm plot

x2 = np.asarray(rf_features_532.index,float)

ax4 = ax3.twinx()


ax4.plot(x2, (nebraska532_concat.loc[:,'mean']+0.5), label = 'A/H1N1', color = 'dimgray')
ax4.plot(x2, (hawaii532_concat.loc[:,'mean']+0.5), label = 'A/H3N2', color = 'darkgrey')
ax4.set_ylabel('Normalized Intensity', fontsize=10, fontname='Arial')

ax3.bar(x2, rf_features_532.iloc[:,0], color='black', edgecolor="black")

ax3.set_ylabel('Decrease in Impurity', fontsize=10, fontname='Arial')
ax3.set_xlabel('Wavenumber ($cm^{-1}$)', fontsize=10, fontname="Arial")

ax3.set_ylim(0,0.03)
ax4.set_ylim(0,1.5)

ax3.xaxis.set_tick_params(labelsize=7)
ax3.yaxis.set_tick_params(labelsize=7)
ax4.yaxis.set_tick_params(labelsize=7)


ymin = 0.8
ymax = 0.85
fontsize = 8
rotation = 90

ax3.axvline(x = 1120.65, ymin = ymin, ymax = ymax, color='black')#, linewidth = 3)
ax3.text(x=1120.65, y = 0.029, s = "1121", fontname = "Arial", fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center', rotation=rotation)

ax3.axvline(x = 1639.65, ymin = ymin, ymax = ymax, color='black')#, linewidth = 3)
ax3.text(x=1639.65, y = 0.029, s = "1640", fontname = "Arial", fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center', rotation=rotation)

ax3.axvline(x = 798.693, ymin = ymin, ymax = ymax, color='black')#, linewidth = 3)
ax3.text(x=798.693, y = 0.029, s = "799", fontname = "Arial", fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center', rotation=rotation)

ax3.axvline(x = 1470.32, ymin = ymin, ymax = ymax, color='black')#, linewidth = 3)
ax3.text(x=1470.32, y = 0.029, s = "1470", fontname = "Arial", fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center', rotation=rotation)

ax3.axvline(x = 1524.7, ymin = ymin, ymax = ymax, color='black')#, linewidth = 3)
ax3.text(x=1524.7, y = 0.029, s = "1525", fontname = "Arial", fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center', rotation=rotation)

ax3.axvline(x = 739.302, ymin = ymin, ymax = ymax, color='black')#, linewidth = 3)
ax3.text(x=739.302, y = 0.029, s = "739", fontname = "Arial", fontsize = fontsize, horizontalalignment = 'center', verticalalignment = 'center', rotation=rotation)



sns.despine(top=True, right=False, left=False, bottom=False)

plt.grid(False)



fig.text(0.25, 0.98,  
         '785 nm',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10) 

fig.text(0.73, 0.98,  
         '532 nm',  
         fontname = 'Arial',
         weight = 'bold',
         fontsize = 10) 

#sns.despine()


#plt.savefig('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /figures/rf_feature_importance.png', dpi=1000, bbox_inches = "tight") 
#plt.savefig('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /figures/rf_feature_importance.pdf', dpi=1000, bbox_inches = "tight") 


plt.show()
