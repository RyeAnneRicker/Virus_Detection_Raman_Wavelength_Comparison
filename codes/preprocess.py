#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 09:09:35 2023

Author: RyeAnne Ricker

Title: Raman Preprocessing

This preprocessing stript allows the user to see how each of the major steps 
of preprocessing affect the data. 

"""

#%%
# libraries
import pandas as pd # v 1.3.5
import numpy as np # v 1.21.2
import matplotlib.pyplot as plt # v 3.1.0
import scipy as sp # v 1.7.3
import pybaselines # v 0.8.0
import scipy
import os
from sklearn.preprocessing import MinMaxScaler # v 0.0

# python version 3.7.11

#%%

def preprocess(sample):
    """This function performs preprocessing of raman files
    This includes a median filter to remove cosmic spikes, an SG filter for smoothing
    and assymmetric least squares baseline correction
    
    Input is a pandas dataframes with the index representing the wavenumber and each column is a sample"""

    # Remove spikes in data - median filter
    medfilt1d = pd.DataFrame(sample)
    medfilt1d.index = sample.index # place the wavenumbers into the index
    for i in range(3,sample.shape[1]-3): # go through all the columns and apply a median filter
        medfilt1d.iloc[3:-3,i] = sp.signal.medfilt(sample.iloc[3:-3,i],kernel_size=3)
    
    # sg filter - smoothing
    sgfilt = pd.DataFrame(np.random.randint(0,10, size=(sample.shape[0],sample.shape[1]))) # again make new df for filtered values
    sgfilt.index = medfilt1d.index 
    for i in range(medfilt1d.shape[1]): # go through all the columns and apply a median filter
        sgfilt.iloc[:,i] = sp.signal.savgol_filter(medfilt1d.iloc[:,i],window_length = 5, polyorder=3)

    # assymetric least squares to get baseline
    baseline = pd.DataFrame(np.random.randint(0,10, size=(sample.shape[0],sample.shape[1]))) # again make new df for filtered values
    baseline.index = medfilt1d.index 
    for i in range(sgfilt.shape[1]): # go through all the columns and fine baseline
        baseline.iloc[:,i] = pybaselines.whittaker.asls(sgfilt.iloc[:,i], lam=1e5)[0]
    corrected = sgfilt.iloc[:,:] - baseline.iloc[:,:] # subtract the baseline off the sample
    corrected.index = sample.index
    return corrected


#%%
# Load the data
folder = '/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/data/785nm/raw/nebraska_positive/100x_5s_400' # path
files = os.listdir(folder) # list of files at the path location
data_files = [] # list to store all ibv file dataframes in
names = [] # list to store the sample number - this is in the same order as the ibvfiles
counter = 0 # start counter at 0

for file in sorted(files): # go through each file and
    if file.endswith('.txt'): # only process .txt files
        df = pd.DataFrame(pd.read_csv(os.path.join(folder,file),
                        sep="\s+", skiprows=19, # the first set of lines are not part of the data
                        index_col=None, header=None, 
                        encoding= 'unicode_escape')) # loads data and skips the rows with run info
        data_files.append(df.iloc[:,:]) # add the file into the list of sample files 
        names.append(file[:-4]) # add the name to list of names, without the .txt part
        print(file) # print file name so we can see all the files we used
        print(df.shape)
        counter = counter + 1 # gives tally of files added
print('') 
print('The number of samples in list is:', counter)


#%%

# check a sample
data_files[0].head()


#%%

# find the mean spectra of each sample and view it

for i in range(len(data_files)):
    data_files[i]['mean'] = data_files[i].iloc[:,1:].mean(axis=1)
    

# plot spectra
plt.figure(figsize=(15,8))
counter = 0
for i in range(len(data_files)):
    plt.plot(data_files[i].iloc[:,0], data_files[i].loc[:,'mean'], label = names[counter])
    counter = counter + 1
plt.xlabel('Wavenumber')
plt.ylabel('Intensity')
plt.title('Raw Spectra')
plt.legend()
plt.show() 


#%%

# remove the 'mean' column from the data
final_files = []
for file in data_files:
    file = file.iloc[:,:-1]
    final_files.append(file)
data_files = final_files


#%%
# Truncate the data from 600 cm-1 to 1800 cm-1

i_wn = 599.5  # initial wavenumber
o_wn = 1800.5 # outer edge wavenumber
trimmed_files = []
for file in data_files: # go through each file
    df = []
    for i in range(file.shape[0]): # go through the rows
        if file.iloc[i,0] > i_wn and file.iloc[i,0] < o_wn: # trim to lengths
            df.append(file.iloc[i,:]) # add to list
    dff = pd.DataFrame(df)
    trimmed_files.append(dff)
    


#%%
    
for i in range(len(trimmed_files)):
    trimmed_files[i]['mean'] = trimmed_files[i].iloc[:,1:].mean(axis=1) # get the average spectrum after trimming
    
    
# plot the trimmed spectra
plt.figure(figsize=(15,8))
counter = 0
for i in range(len(trimmed_files)):
    plt.plot(trimmed_files[i].iloc[:,0], trimmed_files[i].loc[:,'mean'], label = names[counter])
    counter = counter + 1
plt.xlabel('Wavenumber')
plt.ylabel('Intensity')
plt.title('Raw-Trimmed Spectra')
plt.legend()
plt.show()

#%%
# remove the 'mean' column from the data
final_files = []
for file in trimmed_files:
    file = file.iloc[:,:-1]
    final_files.append(file)
trimmed_files = final_files


#%%

# set the first column as the index so that the preprocessing function works

ready_files = []
for file in trimmed_files:
    df = file.set_index([0])
    ready_files.append(df)



#%%

# #### PREPROCESS
    
# remove spikes, smooth the data, and correct the baseline (see function up above)
counter = 0 # start counter at 
preprocessed_files = [] # make an empty list to store the preprocessed files in
for file in ready_files: # go through all the files in the list
    preprocessed = preprocess(file) # preprocess them
    preprocessed_files.append(preprocessed) # add the preprocessed files to the list
    counter += 1 # add 1 to the counter so that you can track how many files you have gone through
    print(counter) # print the number of files completed so far

#%%
    
# Normalization
    
# loop through and normalize each file
scaler = MinMaxScaler() # initiate the min max scaler
normalized_files = []
for file in preprocessed_files:
    x_scaled = scaler.fit_transform(file) # fit the data to the scaler
    x = pd.DataFrame(x_scaled)
    x.index = file.index
    normalized_files.append(x)   
    
#%%
    
# Look at the normalized data
    
# get the mean spectra of the preprocessed and normalized data
for i in range(len(normalized_files)):
    normalized_files[i]['mean'] = normalized_files[i].iloc[:,:].mean(axis=1)
    
# plot the normalized spectra
plt.figure(figsize=(15,8))
counter = 0
for i in range(0,len(normalized_files)):
    plt.plot(normalized_files[i].index, normalized_files[i].loc[:,'mean'], label = names[counter])
    counter = counter + 1
plt.xlabel('Wavenumber')
plt.ylabel('Intensity')
plt.title('Preprocessed and Normalized Data')
plt.legend(loc='upper right')
plt.ylim(0.0,1.6)

#plt.savefig('a', dpi=1000) 

plt.show()

#%%

# remove the 'mean' column from the data frames before saving
final_files = []
for file in normalized_files:
    file2 = file.iloc[:,:-1]
    final_files.append(file2)
    
#%%
    
# save the normalized data
save_folder = '/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/data/785nm/preprocessed/nebraska' # path to save file in
name = 0 # position in the name files to grab for saving the file, start with 0
for file in final_files: # go through the list of all the fully processed files
    file.to_csv(os.path.join(save_folder,'hawaii_'+names[name]+'_preprocessed.txt'), index=True) # save the file to the path with the name of the original file plus _preprocessed on it
    print(names[name]) # print the name of the file completed
    name = name + 1 # go to the next position for naming
