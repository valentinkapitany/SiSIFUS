#%%
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg'
import matplotlib.pyplot as plt
from utils import data_utils, data_vis_utils, ir_utils #run help(utils) to see contents
import tensorflow as tf
from scipy.io import loadmat, savemat
from skimage.metrics import structural_similarity as ssim
import os
import importlib
importlib.reload(ir_utils)

#%%
"""
Using WSL 2, with mamba installed.

Optional (visualisation only) - PyQt5 throws errors without LibGL1, fix with:
sudo apt-get update && sudo apt-get install libgl1

Required:
mamba create -n sisifus-tf python=3.11
mamba activate sisifus-tf
mamba install numpy matplotlib scikit-image scikit-learn tqdm scipy ipykernel
pip3 install tensorflow[and-cuda] opencv-python-headless
pip install lpips
"""
#%% Load data
%matplotlib qt
# Set constants
cmap = plt.cm.jet
epochs = 150
importlib.reload(data_utils)
# Load data using data_utils.load function
data_str = 'TRIMSCOPE_FLIPPER_2' #TRIMSCOPE_FLIPPER, TRIMSCOPE_FLIPPER_2, Flimera_Convallaria_Acridine_Orange, TRIMSCOPE_Rac_Raichu, Flimera_Rac_Raichu, TRIMSCOPE_FLIPPER <- used only for validation
hr_int, hr_tau, hr_int_mask, hr_int_enh, tau_limits, intensfactor, intensfactor_A = data_utils.load(data_str)

# Visualize data
fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# Plot high-resolution tau
plt.sca(axs[0])
plt.imshow(hr_tau, vmin=tau_limits[0], vmax=tau_limits[1], cmap=cmap)
plt.colorbar()

# Plot high-resolution intensity
plt.sca(axs[1])
plt.imshow(hr_int, vmin=0, vmax=np.max(hr_int) / intensfactor, cmap='gray')
plt.colorbar()

# Plot intensity-weighted tau
plt.sca(axs[2])
colors, _ = data_vis_utils.intensity_weighting(hr_tau, hr_int_enh, tau_limits, cmap, intensfactor_A)
plt.imshow(colors, vmin=tau_limits[0], vmax=tau_limits[1], cmap=cmap)
plt.colorbar()

# Show the plot
plt.show(block=False)
plt.pause(0.01)

#%% SiSIFUS pipeline
import importlib
importlib.reload(ir_utils)

# For validation, we downsample the high-resolution FLIM ground truth image.
# For testing, load the low-resolution FLIM image directly
df = 16
lr_tau = hr_tau[::df, ::df]

# Initialize SiSIFUS object with low-resolution tau and high-resolution intensity data
sisifus = ir_utils.SiSIFUS(lr_tau, hr_int, tau_limits=tau_limits)

# # # Generate priors and use them to perform ADMM
local_prior = sisifus.local_pipeline_segment(window_size=(5,5))
global_prior, history = sisifus.global_pipeline_segment(epochs=epochs,patch_size=(13,13))
hr_tau_estimate = sisifus.admm_loop(local_prior,global_prior,ADMM_iter = 20)

# # Data can also be loaded in from repositories
# local_prior = loadmat(os.path.join('data/intermediate', data_str, 'local_prior_upsampling_{}x{}.mat'.format(df,df)))['local_prior']
# global_prior = loadmat(os.path.join('data/intermediate', data_str, 'global_prior_upsampling_{}x{}.mat'.format(df,df)))['tau_mean']
# hr_tau_estimate = loadmat(os.path.join('data/processed','{}_upsampling_{}x{}.mat'.format(data_str,df,df)))['hr_tau_estimate']
#%% Visualise results

colors_hr, _ = data_vis_utils.intensity_weighting(hr_tau, hr_int_enh, tau_limits, cmap, intensfactor_A)
colors_local, _ = data_vis_utils.intensity_weighting(local_prior, hr_int_enh, tau_limits, cmap, intensfactor_A)
colors_global, _ = data_vis_utils.intensity_weighting(np.pad(global_prior,(6,6),'constant',constant_values=0), hr_int_enh, tau_limits, cmap, intensfactor_A)
colors_sisifus, _ = data_vis_utils.intensity_weighting(hr_tau_estimate, hr_int_enh, tau_limits, cmap, intensfactor_A)
bilinear = tf.squeeze(tf.compat.v1.image.resize(lr_tau[tf.newaxis,...,tf.newaxis],hr_int.shape,'bilinear',align_corners=False,)).numpy()
colors_bilinear, _ = data_vis_utils.intensity_weighting(bilinear, hr_int_enh, tau_limits, cmap, intensfactor_A)
nearest = tf.squeeze(tf.compat.v1.image.resize(lr_tau[tf.newaxis,...,tf.newaxis],hr_int.shape,'nearest',align_corners=False,)).numpy()
colors_nearest, _ = data_vis_utils.intensity_weighting(nearest, hr_int_enh, tau_limits, cmap, intensfactor_A)

image_options = {'GT': colors_hr, 
                 'SiSIFUS': colors_sisifus,
                 'Local': colors_local,
                 'Global': colors_global,
                 'Bilinear': colors_bilinear,
                 'Nearest': colors_nearest}

# Initial images
image1_key = 'Local'
image2_key = 'Global'

globals().update(data_vis_utils.two_image_slider(image_options,image1_key,image2_key))
plt.show(block=False)

# Crop extremes (if not, values do not change much)
hr_tau_estimate[hr_tau_estimate<tau_limits[0]] = tau_limits[0]
hr_tau[hr_tau<tau_limits[0]] = tau_limits[0]
bilinear[bilinear<tau_limits[0]] = tau_limits[0]
hr_tau_estimate[hr_tau_estimate>tau_limits[1]] = tau_limits[1]
hr_tau[hr_tau>tau_limits[1]] = tau_limits[1]
bilinear[bilinear>tau_limits[1]] = tau_limits[1]

lpips_sisifus = ir_utils.perceptual_loss(colors_sisifus,colors_hr)
psnr_sisifus = ir_utils.psnr(hr_tau_estimate,hr_tau,tau_limits)
ssim_sisifus = ssim(hr_tau_estimate,hr_tau,win_size=25,data_range=tau_limits[1] - tau_limits[0])

lpips_bilinear = ir_utils.perceptual_loss(colors_bilinear,colors_hr)
psnr_bilinear = ir_utils.psnr(bilinear,hr_tau,tau_limits)
ssim_bilinear = ssim(bilinear,hr_tau,win_size=25,data_range=tau_limits[1] - tau_limits[0])

print('SiSIFUS: LPIPS',lpips_sisifus,'SSIM',ssim_sisifus,'PSNR',psnr_sisifus)
print('Bilinear: LPIPS',lpips_bilinear,'SSIM',ssim_bilinear,'PSNR',psnr_bilinear)
