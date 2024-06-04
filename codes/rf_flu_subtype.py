#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:00:29 2023

Title: Random Forest Classifier for Influenza A subtypes - Binary

This script constructs a RF classifier to classify Raman spectra of viruses
and then uses the mean decrease in impurity to determine feature importance

@author: rickerr2
"""

#%%
# import libraries
import pandas as pd # v 1.3.5
import numpy as np # v 1.21.2
import matplotlib.pyplot as plt # v 3.1.0
import seaborn as sns # v 0.11.2
import os
from numpy.random import seed 
import random as rn 
from sklearn.ensemble import RandomForestClassifier # v 0.0
from matplotlib import pyplot
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint 
import time 
from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics

# python version 3.7.11

#%%
    
  

# load the hawaii data

folder = '/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/data/785nm/preprocessed/hawaii' # path
files = os.listdir(folder) # list of files at the path location
data_files = [] # list to store all ibv file dataframes in
names = [] # list to store the sample number - this is in the same order as the ibvfiles
counter = 0 # start counter at 0

for file in sorted(files): # go through each file and
    if file.endswith('.txt'): # only process .txt files
        df = pd.DataFrame(pd.read_csv(os.path.join(folder,file), index_col=0))
        data_files.append(df.iloc[:,:]) # add the file into the list of sample files 
        names.append(file[:-17]) # add the name to list of names, without the .txt part
        print(file) # print file name so we can see all the files we used
        counter = counter + 1 # gives tally of files added
print('') 
print('The number of samples in list is:', counter)
# get the nebraska data


folder = '/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/data/785nm/preprocessed/nebraska' # path
files = os.listdir(folder) # list of files at the path location
#data_files = [] # list to store all ibv file dataframes in
#names = [] # list to store the sample number - this is in the same order as the ibvfiles
counter = 0 # start counter at 0

for file in sorted(files): # go through each file and
    if file.endswith('.txt'): # only process .txt files
        df = pd.DataFrame(pd.read_csv(os.path.join(folder,file), index_col=0))
        data_files.append(df.iloc[:,:]) # add the file into the list of sample files
        names.append(file[:-17]) # add the name to list of names, without the .txt part
        print(file) # print file name so we can see all the files we used
        counter = counter + 1 # gives tally of files added
print('') 
print('The number of samples in list is:', counter)

#%%

# set reproducability

seed(10) # set numpy seed
#tensorflow.random.set_seed(10) # set tensorflow seed
os.environ['PYTHONHASHSEED'] = '10'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
rn.seed(130)
np.random.seed(42)



#%%


# transpose each of the dataframes

transposed = []

for file in data_files:
    f = file.T # transpose
    transposed.append(f)



#%%
    
    
# add the labels to each sample
    
counter= 0
for file in transposed:
    if 'hawaii' in names[counter]:
        file['Label'] = 0
        file['Sample'] = str('h'+names[counter][-2:])
    if 'nebraska' in names[counter]:
        file['Label'] = 1
        file['Sample'] = str('n'+names[counter][-2:])
    counter = counter + 1

#%%
    
    
# put the dataframes together
    
df = pd.concat(transposed)

data_for_model = df

#%%

# plot the data

df_grouped = df.iloc[:,:-1].groupby("Label").mean()

grouped = df_grouped.T
grouped

#%%

names = list(grouped.columns)

#%%

grouped['wavenumbers'] = grouped.index
df=grouped.reset_index()
df = df.iloc[:,1:]
#df.reset_index(inplace=True)

#%%

plt.figure(figsize=(10,6))

plt.ylabel('Normalized Intensity')
plt.xlabel('Wavenumber')
pyplot.plot(np.asarray(df['wavenumbers'],float), df[0], label = 'Nebraska H1N1',color='blue')
pyplot.plot(np.asarray(df['wavenumbers'],float), df[1], label = "Hawaii H3N2", color='orange')
plt.legend(frameon = False, loc='lower center')
plt.grid(False)
ax = plt.axes()
ax.set_facecolor('white')
rr = plt.legend(frameon=False, loc='upper right')

#plt.savefig('/Users/rickerr2/Documents/local_data/CSF/JC culture media mean Spectra.png', dpi=1000, bbox_extra_artists=(rr)) 


pyplot.show()


#%%
# define the inputs, targets, and groups (biological replicate)
inputs = data_for_model.iloc[:,:-2] # these are all the features
inputs.reset_index() # reset index
targets = data_for_model.iloc[:,-2] # these are the targets
targets.reset_index()
targets.columns = ['class'] # the target column is called 'class'
groups = data_for_model.iloc[:,-1] # these are the arbitrary sample numbers
groups.reset_index()
groups.columns = ['sample'] # column name of patients

#%%
# split the data into trainn and testing, making sure to split so that groups aren't in both the training and the testing
train_inds, test_inds = next(GroupShuffleSplit(test_size=0.5, random_state=50).split(inputs, targets, groups)) # split the training set into testing and training set making sure all spectra from one patient is only in one set or the other
x_train, x_test, y_train, y_test = inputs.iloc[train_inds], inputs.iloc[test_inds], targets.iloc[train_inds], targets.iloc[test_inds]
    

#%%

# define the hyperparameters to tune

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 25, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

#%%

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 25, verbose=2, cv=8, random_state=42, 
                               n_jobs = -1, return_train_score=True)
# Fit the random search model
rf_random.fit(x_train, y_train)


#%%

# get the best parameters

print(" Results from Random Search " )
print("\n The best estimator across ALL searched params:\n", rf_random.best_estimator_)
print("\n The best score across ALL searched params:\n", rf_random.best_score_)
#  The best score across ALL searched params:
# 0.827
print("\n The best parameters across ALL searched params:\n", rf_random.best_params_)

rf_best = rf_random.best_params_

#%%

# fit the model with the best features
model = RandomForestClassifier(**rf_best)
#%%
model.fit(x_train,y_train)


#%%
 # now test the best cross-validated model on the test set    
yhat_probabs = model.predict(x_test) # predict probabilities for test set
    # predict classes for test set
yhat_classes = np.round(yhat_probabs) # I didn't use argmax because I don't have probs for 2 classes, I instead it as a prob between 0 and 1, with >0.5 being class 1
#yhat_probs = yhat_probabs[:, 0] # reduce to 1d array
    
fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat_probabs, pos_label=1)
auc = metrics.auc(fpr,tpr)
probability_threshold = thresholds[np.argmax(tpr - fpr)]
print('ROC AUC: %f' % auc)

#ROC AUC: 0.820000



#%%
# Get the feature importance

# The impurity-based feature importances.

# The higher, the more important the feature. 
# The importance of a feature is computed as the (normalized) total reduction of the criterion 
# brought by that feature. It is also known as the Gini importance.

# The values of this array sum to 1, unless all trees are single node trees consisting of only the root node, 
# in which case it will be an array of zeros.

start_time = time.time()
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
#Elapsed time to compute the importances: 0.466 seconds

feature_names = [f"feature {i}" for i in range(x_train.shape[1])]
forest_importances = pd.Series(importances, index=feature_names)
forest_importances.index = data_files[0].index

#%%

# save the features importances
#forest_importances.to_csv('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength/results/600to1800/785nm_randomforest_feature_importances.csv')
    

#%%

plt.figure(figsize=(10,6))

plt.title("Feature Importance Using Mean Decrease in Impurity", Fontsize=20)
plt.ylabel('Decrease in Impurity', fontsize=16)
plt.xlabel('Wavenumber ($cm^{-1}$)', fontsize=16)
pyplot.bar(np.asarray(forest_importances.index,float), forest_importances, color='rebeccapurple', edgecolor="navy")

#plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.fill_between(x=np.asarray(data_files[0].index, float), y1=0, y2=max(forest_importances), where=(forest_importances < 1.25) & (forest_importances > 0.002),alpha=0.5, color='lightgreen')


plt.grid(False)



sns.despine()



pyplot.show()
