#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:53:34 2024

Title: CNN for Classifying Influenza A virus

This model takes in influenza a raman spectra and trains and cnn to classify the spectra
The models are optimized and cross validated

@author: rickerr2
"""



#%%
# import libraries
import pandas as pd # v 1.3.5
import numpy as np # v 1.21.2
from sklearn.model_selection import StratifiedGroupKFold # sklearn v 0.0
import matplotlib.pyplot as plt # v 3.1.0
import os
from sklearn.model_selection import GroupShuffleSplit
import keras_tuner # v 1.1.0
import tensorflow as tf # v 2.11.0
from sklearn.metrics import accuracy_score
from sklearn import metrics
#import shap
from tensorflow import keras
from keras_tuner import RandomSearch
from tensorflow.keras import activations
from numpy.random import seed
import tensorflow
import random as rn
from matplotlib import pyplot

# python version 3.7.11
    
#%%
# function to build the optimized cnn model

def build_model(hp):
    # create model object
    model = keras.Sequential([
    #adding first convolutional layer    
    keras.layers.Conv1D(
        #adding filter 
        filters=hp.Int('conv_1_filter', min_value=5, max_value=50, step=5),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [3,6,9]),
        #activation function
        activation=activations.elu,
        input_shape=(x_train.shape[1],1)
    ),
        #CNN layer with Max pooling
    keras.layers.MaxPool1D(
        pool_size=(3,), strides=2, padding='same'
    ),
    keras.layers.Conv1D(
        #adding filter 
        filters=hp.Int('conv_2_filter', min_value=5, max_value=50, step=5),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_2_kernel', values = [3,6,9]),
        #activation function
        activation=activations.elu,
        input_shape=(x_train.shape[1],1)
    ),
    #keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.0001)),
    #CNN layer with Max pooling
    keras.layers.MaxPool1D(
        pool_size=(3,), strides=2, padding='same'
    ),
    # adding second convolutional layer 
    keras.layers.Conv1D(
        #adding filter 
        filters=hp.Int('conv_3_filter', min_value=30, max_value=60, step=5),
        #adding filter size or kernel size
        kernel_size=hp.Choice('conv_3_kernel', values = [3,6,9]),
        #activation function
        activation=activations.elu
    ),
    #keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.0001)),
    #CNN layer with Max pooling
    keras.layers.MaxPool1D(
        pool_size=(3,), strides=2, padding='same'
    ),
    # adding third convolutional layer 
    keras.layers.Conv1D(
        #adding filter 
        filters=hp.Int('conv_4_filter', min_value=70, max_value=120, step=5),
        #adding filter size or kernel size
        kernel_size=hp.Choice('conv_4_kernel', values = [3,6,9,12,15]),
        #activation function
        activation=activations.elu,
    ),
    #keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.0001)),

    #CNN layer with Max pooling
    keras.layers.MaxPool1D(
        pool_size=(3,), strides=2, padding='same'
    ),
    keras.layers.Dropout(0.5),
    # adding flatten layer    
    keras.layers.Flatten(),
    # adding dense layer   
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=60, max_value=150, step=10),
        activation=activations.elu
    ),
    #keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.0001)),

    # adding a 2nd dense layer   
    keras.layers.Dense(
        units=hp.Int('dense_2_units', min_value=120, max_value=240, step=10),
        activation=activations.elu
    ),
    keras.layers.Dense(
        units=hp.Int('dense_3_units', min_value=120, max_value=240, step=10),
        activation=activations.elu
    ),
    keras.layers.Dense(
        units=hp.Int('dense_4_units', min_value=120, max_value=240, step=10),
        activation=activations.elu
    ),
    keras.layers.Dense(
        units=hp.Int('dense_5_units', min_value=120, max_value=240, step=10),
        activation=activations.elu
    ),
    keras.layers.Dense(
        units=hp.Int('dense_6_units', min_value=120, max_value=240, step=10),
        activation=activations.elu
    ),
    #keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.0001)),

    # output layer    
    keras.layers.Dense(units=1, activation='sigmoid')
    ])
    #compilation of model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
              loss='binary_crossentropy',
              #metrics=tf.keras.metrics.AUC())
              metrics=['accuracy', tf.keras.metrics.AUC()])
    return model
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
        data_files.append(df.iloc[:,:]) # add the file into the list of sample files and remove the x,y coordinates located in the first two columns
        names.append(file[:-17]) # add the name to list of names, without the .txt part
        print(file) # print file name so we can see all the files we used
        counter = counter + 1 # gives tally of files added
print('') 
print('The number of samples in list is:', counter)

#%%

# get the nebraska data


folder = '/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison/data/785nm/preprocessed/nebraska' # path
files = os.listdir(folder) # list of files at the path location
#data_files = [] # list to store all ibv file dataframes in
#names = [] # list to store the sample number - this is in the same order as the ibvfiles
counter = 0 # start counter at 0

for file in sorted(files): # go through each file and
    if file.endswith('.txt'): # only process .txt files
        df = pd.DataFrame(pd.read_csv(os.path.join(folder,file), index_col=0))
        data_files.append(df.iloc[:,:]) # add the file into the list of sample files and remove the x,y coordinates located in the first two columns
        names.append(file[:-17]) # add the name to list of names, without the .txt part
        print(file) # print file name so we can see all the files we used
        counter = counter + 1 # gives tally of files added
print('') 
print('The number of samples in list is:', counter)

#%%

# set reproducability

seed(10) # set numpy seed
tensorflow.random.set_seed(10) # set tensorflow seed
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
training = []
training_names = []
testing = []
testing_names = []
    
counter= 0
for file in transposed:
    if 'hawaii' in names[counter]:
        file['Label'] = 0
        file['Sample'] = str('h'+names[counter][-2:])
        
    if 'nebraska' in names[counter]:
        file['Label'] = 1
        file['Sample'] = str('n'+names[counter][-2:])
    
    # get samples 5 and 10 for each flu subtype - these will be used as testing data

    if (counter == 4):
        testing.append(file)
        testing_names.append(names[counter])
    elif (counter == 9):
        testing.append(file)
        testing_names.append(names[counter])
    elif (counter == 14):
        testing.append(file)
        testing_names.append(names[counter])
    elif (counter == 19):
        testing.append(file)
        testing_names.append(names[counter])
    else:
        training.append(file)
        training_names.append(names[counter])
    counter = counter + 1

#%%
    
    
# put the dataframes together
    
train = pd.concat(training)
final_test = pd.concat(testing)

df = pd.concat(training)

data_for_model = train

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
pyplot.plot(np.asarray(df['wavenumbers'],float), df[0], label = 'FluA/H3N2/Hawaii',color='blue')
pyplot.plot(np.asarray(df['wavenumbers'],float), df[1], label = "FluA/H1N1/Nebraska", color='orange')
plt.legend(frameon = False, loc='lower center')
plt.grid(False)
ax = plt.axes()
ax.set_facecolor('white')
rr = plt.legend(frameon=False, loc='upper right')

#plt.savefig('/Users/rickerr2/Documents/local_data/CSF/JC culture media mean Spectra.png', dpi=1000, bbox_extra_artists=(rr)) 


pyplot.show()


#%%
# separate data by inputs, targets, and groups (biological replicates)
inputs = data_for_model.iloc[:,:-2] # these are all the features
inputs.reset_index() # reset index
targets = data_for_model.iloc[:,-2] # these are the targets
targets.reset_index()
targets.columns = ['class'] # the target column is called 'class'
groups = data_for_model.iloc[:,-1] # these are the arbitrary patient numbers
groups.reset_index()
groups.columns = ['sample'] # column name of patients


#%%
# this is the loop that actually does the model optimization and cross validation

acc_per_fold = [] # stores accuracy per fold
auc_per_fold = [] # stores aauc per fold
train_loss_per_fold = [] # stores loss per fold
val_loss_per_fold = []
sample_acc = []
models = [] # store each folds model so that after the CV, we use the best one to evaluate on the test set
thresholds_per_fold = [] # store the probability threshold for each fold


batch_size = 32
# K-fold Cross Validation model evaluation
fold_no = 1
groupkfold = StratifiedGroupKFold(n_splits=8)


for train, test in groupkfold.split(inputs, targets, groups):

    #for train, test in kfold.split(inputs, targets, groups):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no}')
    train_inds, val_inds = next(GroupShuffleSplit(test_size=0.25, random_state=42).split(inputs.iloc[train], targets.iloc[train], groups.iloc[train])) # split the training set into testing and training set making sure all spectra from one patient is only in one set or the other
    x_train, x_val, y_train, y_val = inputs.iloc[train_inds], inputs.iloc[val_inds], targets.iloc[train_inds], targets.iloc[val_inds]
    
    #x_train, x_val, y_train, y_val = train_test_split(inputs.iloc[train], targets.iloc[train], test_size = 0.2, random_state = 42)
    
    #creating randomsearch object
    tuner_all = RandomSearch(build_model,
                    objective=keras_tuner.Objective('val_auc', direction='max'),
                    overwrite=True, max_trials = 25
                    ) # overwrite is important so that it makes a new model and doesn't just load the new one
    # search best parameter
    tuner_all.search(x_train,y_train,epochs=20, validation_data=(x_val, y_val))

  # Define the model architecture
    model_all=tuner_all.get_best_models(num_models=1)[0]

  # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

  # Fit data to model
    history = model_all.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size= batch_size,
              epochs=100)#,
              #verbose=verbosity)
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    train_loss_per_fold.append(train_loss)
    val_loss_per_fold.append(val_loss)
  # Generate generalization metrics
    scores = model_all.evaluate(inputs.iloc[test], targets.iloc[test], verbose=0)
    
    yhat_probabs = model_all.predict(inputs.iloc[test], verbose=0) # predict probabilities for test set
    # predict classes for test set
    yhat_classes = np.round(yhat_probabs) # I didn't use argmax because I don't have probs for 2 classes, I instead it as a prob between 0 and 1, with >0.5 being class 1
    yhat_probs = yhat_probabs[:, 0] # reduce to 1d array
    
    fpr, tpr, thresholds = metrics.roc_curve(targets.iloc[test], yhat_probs, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    probability_threshold = thresholds[np.argmax(tpr - fpr)]
    #thresholds.append(probability_threshold)
    print('ROC AUC: %f' % auc)
    
    accuracy = accuracy_score(targets.iloc[test], yhat_classes)
    print('Accuracy: %f' % accuracy)
    
    print(f'Score for fold {fold_no}: {model_all.metrics_names[0]} of {scores[0]}; accuracy of ', accuracy, 'auc of ', auc)
    acc_per_fold.append(accuracy*100)
    auc_per_fold.append(auc * 100)
    #loss_per_fold.append(scores[0])
    thresholds_per_fold.append(probability_threshold)
    models.append(model_all)
    

  # Increase fold number
    fold_no = fold_no + 1
    
#%%
 # get the average auc of all the folds   
cv_auc_all = pd.DataFrame(auc_per_fold).mean()
print('The cross validation auc is:', cv_auc_all)
# get the average accuracy of all the folds
cv_acc_all = pd.DataFrame(acc_per_fold).mean()
print('The cross validation accuracy is:', cv_acc_all)

#pd.DataFrame(auc_per_fold).to_csv(os.path.join('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /results/785nm_AUC_per_fold.csv'))
#pd.DataFrame(acc_per_fold).to_csv(os.path.join('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /results/785nm_Acc_per_fold.csv'))


# 785 nm: 
#auc_per_fold
#Out[267]: 
#[99.78687500000001,
# 94.27875,
# 100.0,
# 77.7865625,
# 92.1575,
# 100.0,
# 98.0275,
# 96.14562500000001]

#cv_auc_all
#Out[266]: 
#0    94.772852
#dtype: float64



# 532 nm:
#auc_per_fold 
#Out[4]: 
#[99.94437500000001,
# 98.05312500000001,
# 100.0,
# 93.854375,
# 50.0,
# 50.0,
# 96.856875,
# 96.046875]

#cv_auc_all
#Out[5]: 
#0    85.594453
#dtype: float64
#%%
# get the best model

m = pd.Series(auc_per_fold).idxmax() # get the fold with the highest auc
model = models[m] # grab the model from the fold with the highest auc

yhat_probabs = model.predict(final_test.iloc[:,:-2], verbose=0) # predict probabilities for test set
    # predict classes for test set
yhat_classes = np.round(yhat_probabs) # I didn't use argmax because I don't have probs for 2 classes, I instead it as a prob between 0 and 1, with >0.5 being class 1
yhat_probs = yhat_probabs[:, 0] # reduce to 1d array
    
fpr, tpr, thresholds = metrics.roc_curve(final_test.iloc[:,-2], yhat_probs, pos_label=1)

i = np.arange(len(tpr)) 
roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thresholds' : pd.Series(thresholds, index=i)})
roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
probability_threshold = list(roc_t['thresholds'])
print('The probability threshold is: ',probability_threshold[0])
probability_threshold = probability_threshold[0]

auc = metrics.auc(fpr,tpr)
#probability_threshold = thresholds[np.argmax(tpr - fpr)]
print('Testing set AUC: ', auc)
accuracy = accuracy_score(final_test.iloc[:,-2], yhat_classes)
print('Testing set Accuracy: ', accuracy)


# results for 785 nm:
#Testing set AUC:  0.9035578124999999
#Testing set Accuracy:  0.8075
#Testing set AUC:  0.8963828125000001
#Testing set Accuracy:  0.811875
#Testing set AUC:  0.8946312500000001
#Testing set Accuracy:  0.8025
#Testing set AUC:  0.91856640625
#Testing set Accuracy:  0.834375
#Testing set AUC:  0.9367671875
#Testing set Accuracy:  0.856875
#Testing set AUC:  0.8977703125000001
#Testing set Accuracy:  0.805
#Testing set AUC:  0.91626875
#Testing set Accuracy:  0.82
#Testing set AUC:  0.92304375
#Testing set Accuracy:  0.83125

# results for 532 nm:
#Testing set AUC:  0.81407734375
#Testing set Accuracy:  0.7225
#Testing set AUC:  0.833821875
#Testing set Accuracy:  0.740625
#Testing set AUC:  0.8454320312500001
#Testing set Accuracy:  0.7525
#Testing set AUC:  0.8231085937500001
#Testing set Accuracy:  0.721875
#Testing set AUC:  0.5
#Testing set AUC:  0.5
#Testing set Accuracy:  0.5
#Testing set AUC:  0.83921484375
#Testing set Accuracy:  0.75625
#Testing set AUC:  0.842253125
#Testing set Accuracy:  0.7625


sample_acc = [] # list to store sample metrics in

sample_groups = final_test.groupby(by = ['Sample']) # group by patients so that i can test on each patient individually
g = sample_groups.groups.keys() # get the keys to call each patient separately
sample_groups_list = list(g) # get list of samples to call
for i in range(len(sample_groups_list)):  # go through each patient in the group
    df = sample_groups.get_group(sample_groups_list[i]) # call the group
            #scores2 = model.evaluate(df.iloc[:,:-2], df.iloc[:,-2], verbose=0) # get an accuracy for each patient, later we will use the score and set a threshold to actually classify the patient
    pred = model.predict(df.iloc[:,:-2])


    df = pd.DataFrame(pred)
    df.columns = ['Probability of Belonging to Class 1 (Nebraska)']
    #df.to_csv(os.path.join('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /results/'+'785nm_'+sample_groups_list[i]+'_class_probabilities.csv'))
    
    
#%%
    
# get the test set AUC for each folds model
for i in range(len(models)):
    model = models[i] # grab the model from the fold with the highest auc

    yhat_probabs = model.predict(final_test.iloc[:,:-2], verbose=0) # predict probabilities for test set
    # predict classes for test set
    yhat_classes = np.round(yhat_probabs) # I didn't use argmax because I don't have probs for 2 classes, I instead it as a prob between 0 and 1, with >0.5 being class 1
    yhat_probs = yhat_probabs[:, 0] # reduce to 1d array
    
    fpr, tpr, thresholds = metrics.roc_curve(final_test.iloc[:,-2], yhat_probs, pos_label=1)

    auc = metrics.auc(fpr,tpr)
#probability_threshold = thresholds[np.argmax(tpr - fpr)]
    print('Testing set AUC for model from fold ', i+1,' : ', auc)


# results for 785 nm:   
#Testing set AUC for model from fold  1  :  0.9035578124999999
#Testing set AUC for model from fold  2  :  0.8963828125000001
#Testing set AUC for model from fold  3  :  0.8946312500000001
#Testing set AUC for model from fold  4  :  0.91856640625
#Testing set AUC for model from fold  5  :  0.9367671875
#Testing set AUC for model from fold  6  :  0.8977703125000001
#Testing set AUC for model from fold  7  :  0.91626875
#Testing set AUC for model from fold  8  :  0.92304375
    
# results for 532 nm:
#Testing set AUC for model from fold  1  :  0.81407734375
#Testing set AUC for model from fold  2  :  0.833821875
#Testing set AUC for model from fold  3  :  0.8454320312500001
#Testing set AUC for model from fold  4  :  0.8231085937500001
#Testing set AUC for model from fold  5  :  0.5
#Testing set AUC for model from fold  6  :  0.5
#Testing set AUC for model from fold  7  :  0.83921484375
#Testing set AUC for model from fold  8  :  0.842253125
    
#%%
 
auc_each_fold_model = [] 
acc_each_fold_model = []
    
for m in range(len(models)):
    # Save the entire model as a `.keras` zip archive.
    model = models[m]
    l = m+1
    # make sure to save this 532 nm for the other laser
    model.save('model_785nm_600to1800_fold' + str(l)+'.keras') # save the model
    
    yhat_probabs = model.predict(final_test.iloc[:,:-2], verbose=0) # predict probabilities for test set
    # predict classes for test set
    yhat_classes = np.round(yhat_probabs) # I didn't use argmax because I don't have probs for 2 classes, I instead it as a prob between 0 and 1, with >0.5 being class 1
    yhat_probs = yhat_probabs[:, 0] # reduce to 1d array
    
    fpr, tpr, thresholds = metrics.roc_curve(final_test.iloc[:,-2], yhat_probs, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    probability_threshold = thresholds[np.argmax(tpr - fpr)]
    print('Testing set AUC: ', auc)
    accuracy = accuracy_score(final_test.iloc[:,-2], yhat_classes)
    print('Testing set Accuracy: ', accuracy)
    
    auc_each_fold_model.append(auc)
    acc_each_fold_model.append(accuracy)
    
#%%
    
# save training and validation loss
dfs = []   
for file in train_loss_per_fold:
    f = pd.DataFrame(file)
    dfs.append(f)
train_loss = pd.concat(dfs, axis=1)  

dfs = []   
for file in val_loss_per_fold:
    f = pd.DataFrame(file)
    dfs.append(f)
val_loss = pd.concat(dfs, axis=1)  

#train_loss.to_csv('/Users/rickerr2/Library/CloudStorage/OneDrive-NationalInstitutesofHealth/Documents/LAB_STUFF/PEOPLE/rye/Publications/Optica Biomedical Optics Express - Raman Wavelength Comparison /results/785nm_training_loss.csv')
   
