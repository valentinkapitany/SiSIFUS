#%%
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg'
import matplotlib.pyplot as plt
from utils import data_utils, data_vis_utils, ir_utils #run help(utils) to see contents
import tensorflow as tf
from scipy.io import loadmat, savemat
from scipy.stats import gmean
plt.rcParams.update(data_vis_utils.my_vis_dict)
# #%% SF-7

cmap = plt.cm.jet
lpips_window_array = np.zeros([4, 7, 4])

# Iterate over different datasets
for i, data_str in enumerate(['Flimera_Convallaria_Acridine_Orange', 'TRIMSCOPE_Rac_Raichu', 'Flimera_Rac_Raichu', 'TRIMSCOPE_FLIPPER']):
    print('\n', data_str, '\n')
    
    # Load data for the current dataset
    hr_int, hr_tau, hr_int_mask, hr_int_enh, tau_limits, intensfactor, intensfactor_A = data_utils.load(data_str)
    
    # Iterate over different window sizes
    for j, ws in enumerate([2, 3, 4, 5, 6, 7, 8]):
        
        # Iterate over different downsampling factors
        for k, df in enumerate([2, 4, 8, 16]):
            lr_tau = hr_tau[::df, ::df]

            # Apply SiSIFUS algorithm
            sisifus = ir_utils.SiSIFUS(lr_tau, hr_int, tau_limits=tau_limits)
            
            # Perform local pipeline segmentation
            local_prior = sisifus.local_pipeline_segment(window_size=(ws, ws))
            
            # Compute intensity weighting
            colors_hr, _ = data_vis_utils.intensity_weighting(hr_tau, hr_int_enh, tau_limits, cmap, intensfactor_A)
            colors_local, _ = data_vis_utils.intensity_weighting(local_prior, hr_int_enh, tau_limits, cmap, intensfactor_A)
            
            # Crop extrema
            local_prior[local_prior<tau_limits[0]] = tau_limits[0]
            hr_tau[hr_tau<tau_limits[0]] = tau_limits[0]
            local_prior[local_prior>tau_limits[1]] = tau_limits[1]
            hr_tau[hr_tau>tau_limits[1]] = tau_limits[1]

            # Compute LPIPS
            lpips_sisifus = ir_utils.perceptual_loss(colors_local, colors_hr)
            lpips_window_array[i, j, k] = lpips_sisifus

            # Uncomment below lines to plot the local colors
            # plt.figure()
            # plt.imshow(colors_local)
            # plt.pause(1)

# %%
%matplotlib inline
# Calculate the geometric mean of LPIPS values across datasets and downsampling factors
gmean_lpips = gmean(lpips_window_array, axis=(0, 2))

# Calculate the arithmetic mean of LPIPS values across datasets and downsampling factors
mean_lpips = np.mean(lpips_window_array, axis=(0, 2))

fig, ax = plt.subplots(figsize=(6,4))
ax2 = plt.twinx(ax)
ax.plot(range(2,9), mean_lpips, label='arithmetic mean', c='b')
ax.tick_params(axis='y',labelcolor='b')
ax.set_ylabel('LPIPS (arithmetic mean)',c='b')
ax2.plot(range(2,9), gmean_lpips, label='geometric mean', c='r')
ax2.tick_params(axis='y',labelcolor='r')
ax2.set_ylabel('LPIPS (geometric mean)',c='r')
ax.set_xlabel('window size')
ax.set_xticks(range(2,9),range(2,9))
plt.savefig('results/sf1/fig.svg',,bbox_inches='tight')
savemat('results/sf1/fig_data.mat',{'x_data':range(2,9),'mean_lpips':mean_lpips,'gmean_lpips':gmean_lpips})
plt.show()
#%%
#%% SF-7
# Initialize an array to store LPIPS values
lpips_func_array = np.zeros([4, 7, 4])
mae_func_array = np.zeros([4, 7, 4])
my_funcs = ['linear_spline','quadratic_spline','cubic_spline','nearest_interp','linear_interp','cubic_interp','rbf_gp']
# Iterate over different datasets
for i, data_str in enumerate(['Flimera_Convallaria_Acridine_Orange', 'TRIMSCOPE_Rac_Raichu', 'Flimera_Rac_Raichu', 'TRIMSCOPE_FLIPPER']):
    print('\n', data_str, '\n')
    
    # Load data for the current dataset
    hr_int, hr_tau, hr_int_mask, hr_int_enh, tau_limits, intensfactor, intensfactor_A = data_utils.load(data_str)
    ws = 5
    # Iterate over different window sizes
    for j, ft in enumerate(my_funcs):
        
        # Iterate over different downsampling factors
        for k, df in enumerate([2, 4, 8, 16]):
            lr_tau = hr_tau[::df, ::df]

            # Apply SiSIFUS algorithm
            sisifus = ir_utils.SiSIFUS(lr_tau, hr_int, tau_limits=tau_limits)
            
            # Perform local pipeline segmentation
            local_prior = sisifus.local_pipeline_segment(func_type=ft)

            # Crop extrema
            local_prior[local_prior<tau_limits[0]] = tau_limits[0]
            hr_tau[hr_tau<tau_limits[0]] = tau_limits[0]
            local_prior[local_prior>tau_limits[1]] = tau_limits[1]
            hr_tau[hr_tau>tau_limits[1]] = tau_limits[1]
            
            # Compute intensity weighting
            colors_hr, _ = data_vis_utils.intensity_weighting(hr_tau, hr_int_enh, tau_limits, cmap, intensfactor_A)
            colors_local, _ = data_vis_utils.intensity_weighting(local_prior, hr_int_enh, tau_limits, cmap, intensfactor_A)
            
            # Compute LPIPS and MAE
            lpips_local = ir_utils.perceptual_loss(colors_local, colors_hr)
            mae_local = np.mean(np.abs(hr_tau - local_prior))
            lpips_func_array[i, j, k] = lpips_local
            mae_func_array[i, j, k] = mae_local

            # Uncomment below lines to plot the local colors
            # plt.figure()
            # plt.imshow(colors_local)
            # plt.pause(1)

#%%
# mae_func_array = np.load('mae_func_array.npy')
# lpips_func_array = np.load('lpips_func_array.npy')
x_ticks = ['linear\nspline',
           'quad\nspline',
           'cubic\nspline',
           'nearest\ninterp',
           'linear\ninterp',
           'cubic\ninterp',
           'rbf\ngp']

fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = plt.twinx(ax1)

ax1.plot(x_ticks,np.mean(mae_func_array,axis=(0,2)),c='b')
ax2.plot(x_ticks,np.mean(lpips_func_array,axis=(0,2)),c='r')

ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.tick_params(axis='y', which='both', colors='b')
ax2.tick_params(axis='y', which='both', colors='r')
ax1.set_ylabel('MAE (ns)', color='b')
ax2.set_ylabel('LPIPS (a.u.)', color='r')

plt.savefig('results/sf2/fig.svg',bbox_inches='tight')
savemat('results/sf2/fig_data.mat',{'x_data':x_ticks,'mae_data':np.mean(mae_func_array,axis=(0,2)),'lpips_data':np.mean(lpips_func_array,axis=(0,2))})
plt.show()
