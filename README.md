# Virus_Detection_Raman_Wavelength_Comparison

> ##### This project tests the incident wavelength of Raman spectroscopy (785 nm vs 532) to detect viruses
> ##### Additionally, this project tests the robustness of our previously published platform by introducing biological replicates
> ##### (having separate growth cultures), variation between runs, variation in the CNT platform, and finally, 
> ##### demonstrates how this technology would word in the real-world by performing a data simulation

> ##### This project is for the paper: Rapid and label-free Influenza A subtyping using surface-enhanced Raman spectroscopy with incident-wavelength analysis
> ###### RYEANNE RICKER1,2, NESTOR PEREA3,  MURRAY LOEW2, AND ELODIE GHEDIN1,*



## This folder contains all the data, code, and results

### Data
* ##### Collected at 100x magnification for 5 seconds with 400 spectra collected per biological replicate 
* ##### 40 files in total, each with 400 spectra
>> ##### 785 nm - 20 files
>>> ######     Raw H1N1 Samples - 10 files/10 biological replicates, each one containing 400 spectra
>>> ######     Raw H3N2 Samples - 10 files/10 biological replicates, each one containing 400 spectra
>> ##### 532 nm - 20 files
>>> ###### - Raw H1N1 Samples - 10 files/10 biological replicates, each one containing 400 spectra
>>> ###### - Raw H3N2 Samples - 10 files/10 biological replicates, each one containing 400 spectra


### Codes
> #### Run in this order:
>> ###### 1) preprocess.py - preprocessed the data
>> ###### 2) cnn_h1n1_vs_h3n2_cv_plus_test.py - CNN model that optimizes the hyperparameters and performs cross validation + testing
>> ###### 3) rf_flu_subtype.py - Random Forest model that is optimized, performs CV, and extracts feature importance
>> ###### 4) random_sampling_experiment.py - real world simulation
>> ###### 5) visualizations_and_calculations.py - visualizations and calculations/statistics

### Results
>> ###### AUC and accuracies
>> ###### t-tests
>> ###### Training Losses
>> ###### Simulation Results
>> ###### Feature Importance
>> ###### Class Probabilities of Predictions on the Testing Samples
