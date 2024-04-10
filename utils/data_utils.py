# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 11:02:08 2020

@author: Valentin Kapitany
"""
#%%
import os
import numpy as np
from matplotlib.pyplot import cm
from scipy.signal import medfilt
from scipy.io import loadmat
from scipy.io import loadmat
#%%

def load(name):
    """
    Load data for different experiments based on the provided name.

    Args:
    - name (str): Name of the experiment to load.

    Returns:
    - hr_int (numpy.ndarray): High-resolution intensity image data.
    - hr_tau (numpy.ndarray): High-resolution lifetime image data.
    - hr_int_mask (numpy.ndarray): Mask for the high-resolution intensity image.
    - tau_limits (list): List containing the lower and upper bounds for lifetime values.
    - intensfactor (float): Intensity factor for the experiment.
    """
    if name=='TRIMSCOPE_FLIPPER':
        file = loadmat('data/raw/TRIMSCOPE_FLIPPER.mat')

        hr_int = file['hr_int']
        hr_tau = file['hr_tau']
        hr_int_mask = np.ones_like(hr_int)
        hr_int_enh = hr_int.copy()

        tau_limits = [0,3.5]
        intensfactor = 10
        intensfactor_A = 17

    if name=='TRIMSCOPE_Rac_Raichu':
        file = loadmat('data/raw/TRIMSCOPE_Rac_Raichu.mat')
        
        hr_int = file['hr_int']
        hr_tau = file['hr_tau']
        hr_int_mask = np.ones_like(hr_int)
        hr_int_enh = loadmat("data/intermediate/TRIMSCOPE_Rac_Raichu/intensity_enhanced.mat")['Int_output3']
        
        tau_limits = [1,2.5]
        intensfactor = 3
        intensfactor_A = 2


    if name=='Flimera_Convallaria_Acridine_Orange':
        file = loadmat('data/raw/Flimera_Convallaria_Acridine_Orange.mat')
        
        hr_int = file['hr_int']
        hr_tau = file['hr_tau']
        hr_int_mask = np.ones_like(hr_tau)
        hr_int_enh = loadmat("data/intermediate/Flimera_Convallaria_Acridine_Orange/intensity_enhanced.mat")['Int_output3']

        tau_limits = [0.1, 0.8]        
        intensfactor = 2
        intensfactor_A = 1

    if name=='Flimera_Rac_Raichu':
        file = loadmat('data/raw/Flimera_Rac_Raichu.mat')
        
        hr_int = file['hr_int']
        hr_tau = file['hr_tau']
        hr_int_mask = np.ones_like(hr_int)
        hr_int_enh = loadmat("data/intermediate/Flimera_rac_Raichu/intensity_enhanced.mat")['Int_output']
        
        tau_limits = [0.3,1.4]
        intensfactor = 1.5
        intensfactor_A = 1


    if name=='TRIMSCOPE_FLIPPER_2':
        file = loadmat('data/raw/TRIMSCOPE_FLIPPER_2.mat')
        hr_int = file['hr_int']
        hr_tau = file['hr_tau']
        hr_int_mask = np.ones_like(hr_int)
        hr_int_enh = hr_int.copy()
        
        tau_limits = [0,3.5]
        intensfactor = 7
        intensfactor_A = 25

    return hr_int, hr_tau, hr_int_mask, hr_int_enh, tau_limits, intensfactor, intensfactor_A
    