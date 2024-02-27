# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 11:02:08 2020

@author: Valentin Kapitany
"""
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.util import view_as_windows
from tqdm import tqdm

from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from scipy import interpolate
import cv2
import tensorflow as tf

from utils import cnn_utils
import torch
from lpips import LPIPS
from matplotlib.colors import Normalize
#%% 
class SiSIFUS(object):
    """
    SiSIFUS: Single-Sample Image-Fusion Upsampling

    This class implements the SiSIFUS algorithm for super-resolution imaging via image fusion.
    It provides methods for pre-processing data, generating local and global priors, applying the ADMM optimization, and more.

    Attributes:
    - lr_tau (numpy.ndarray): Low-resolution lifetime image data.
    - hr_int (numpy.ndarray): High-resolution intensity image data.
    - lr_tau_mask (numpy.ndarray): Mask for the low-resolution lifetime image.
    - hr_int_mask (numpy.ndarray): Mask for the high-resolution intensity image.
    - tau_limits (tuple): Tuple containing lower and upper bounds for lifetime values.
    - upsampling_factor (tuple): Tuple indicating the upsampling factor along the y and x axes.
    - local_window_size (tuple): Size of the window for local prior generation.
    - global_patch_size (tuple): Size of the patches for global prior generation.

    Methods:
    - __init__(self, lr_tau, hr_int, lr_tau_mask=None, hr_int_mask=None, tau_limits=None): 
      Initializes the SiSIFUS object with the provided data and optional parameters.
    - data_clean(self): Cleans the data by filling missing values and applying masks.
    - local_data_prep(self, window_size=(5, 5)): Prepares data for local prior generation based on the specified window size.
    - global_data_prep(self, patch_size=(13, 13), edge_mode=None, pad=None): 
      Prepares data for global prior generation based on specified patch size and padding.
    - local_prior(self, lr_tau_winds, lr_int_winds, loo_cv=False, func_type='linear_interp'): 
      Computes the local prior based on low-resolution tau and intensity windows using interpolation techniques.
    - local_fit(self, X, y, Xtest, func_type='linear_interp'): 
      Fits local models to the data points and predicts values at test points using various interpolation techniques.
    - global_prior(self, lr_int_patches, lr_tau_centres, hr_int_patches, model_v='v6', epochs=150, augment=True): 
      Generates the global prior based on the provided low-resolution intensity patches and corresponding tau centres.
    - local_pipeline_segment(self): End-to-end pipeline to create the local prior.
    - global_pipeline_segment(self, pad=None, epochs=150): End-to-end pipeline to create the global prior.
    - admm_loop(self, local_prior, global_prior): 
      Applies the Alternating Direction Method of Multipliers (ADMM) to optimize the reconstruction.

    Note: Detailed descriptions of parameters, return values, and functionality are provided within each method's docstring.
    """

    def __init__(self, lr_tau, hr_int, lr_tau_mask=None, hr_int_mask=None, tau_limits = None):
        """Pre-processes data to be used for upsampling a mono-exponential lifetime estimate using an intensity image guide"""
        assert(len(lr_tau.shape)==2 and len(hr_int.shape)==2)
        assert(hr_int.shape[0]%lr_tau.shape[0]==0 and hr_int.shape[1]%lr_tau.shape[1]==0),'expect an integer difference between the low-resolution lifetime and high-resolution intensity sizes'
        if lr_tau_mask is not None: 
            assert(lr_tau_mask.shape==lr_tau.shape),'Lifetime mask expected to have same shape as lifetime image'
        else:
            self.lr_tau_mask = np.ones_like(lr_tau)
        if hr_int_mask is not None: 
            assert(hr_int_mask.shape==hr_int_mask.shape),'Intensity mask expected to have same shape as intensity image'
        else:
            self.hr_int_mask = np.ones_like(hr_int)
        if tau_limits is not None: 
            assert(len(tau_limits)==2),'Expect tau_limits to be a tuple of floats/ints'
            self.tau_limits = tau_limits
        elif tau_limits is None: 
            self.tau_limits = (np.min(lr_tau),np.max(lr_tau))
        self.lr_tau = lr_tau.astype('float64')
        self.hr_int = hr_int.astype('float64')
        self.upsampling_factor = (int(hr_int.shape[0]/lr_tau.shape[0]),int(hr_int.shape[1]/lr_tau.shape[1])) #y,x

    def data_clean(self):
        """"Clean the data by filling missing values and applying masks."""
        self.lr_tau = data_fill(self.lr_tau)
        self.lr_tau = self.lr_tau_mask*self.lr_tau
        if 'tau_limits' in self.__dict__:
            self.lr_tau[self.lr_tau<0]=0
        self.hr_int = data_fill(self.hr_int)
        self.hr_int = self.hr_int_mask*self.hr_int
    
    def local_data_prep(self, window_size=(5,5)):
        """
        Prepare data for local prior generation based on the specified window size.

        Parameters:
        - window_size (tuple): Size of the window for local prior generation.

        Returns:
        - lr_tau_winds (numpy.ndarray): Low-resolution tau windows.
        - lr_int_winds (numpy.ndarray): Low-resolution intensity windows.

        This method prepares data for generating the local prior used in subsequent processing steps.
        It divides the low-resolution tau and intensity data into windows of the specified size.

        The resulting data include:
        - lr_tau_winds: Low-resolution tau windows obtained through windowing_func.
        - lr_int_winds: Low-resolution intensity windows obtained through windowing_func.

        Note: some differences exist between the outputs of this code and the published data, due to floating point differences. 
        The published data converted data into float16s, then to float64s, then back to float16s. We just keep it at float64s.
        """

        self.local_window_size = window_size #add to instance variables for local prior generation

        lr_tau_winds = windowing_func(self.lr_tau,window_size) 
        lr_int = self.hr_int[::self.upsampling_factor[1],::self.upsampling_factor[0]]   
        lr_int_winds = windowing_func(lr_int,window_size)

        pos = np.indices(self.hr_int.shape)
        lr_pos_ = pos[:,::self.upsampling_factor[0],::self.upsampling_factor[1]]
        self.lr_pos = np.zeros([2] + list(lr_int_winds.shape))
        self.lr_pos[0] = windowing_func(lr_pos_[0],window_size)
        self.lr_pos[1] = windowing_func(lr_pos_[1],window_size) 
        return lr_tau_winds, lr_int_winds
        
    def global_data_prep(self, patch_size=(13, 13), pad=None):
        """
        Prepare data for global prior generation based on specified patch size and padding.

        Parameters:
        - patch_size (tuple): Size of the patches for windowing the intensity.
        - pad (int): Amount of padding to apply around each flim sample for window selection.

        Returns:
        - lr_int_patches (numpy.ndarray): Low-resolution intensity patches for training inputs.
        - lr_tau_centres (numpy.ndarray): Low-resolution tau values corresponding to training labels.
        - hr_int_patches (numpy.ndarray): High-resolution intensity patches for test inputs.

        This method prepares data for generating the global prior used in subsequent processing steps.
        It divides the high-resolution intensity into patches of the specified size, considering padding.
        The function selects appropriate windows containing flim samples and extracts corresponding intensity patches.
        It ensures that the number of training inputs and labels are equal.

        The resulting data include:
        - lr_int_patches: Low-resolution intensity patches corresponding to the selected windows.
        - lr_tau_centres: Low-resolution tau values corresponding to the flim samples.
        - hr_int_patches: High-resolution intensity patches for test inputs.

        Note: The function ensures consistency between the number of test inputs, window centres, and intensity patches.
        """
        if pad is None:
            if min(self.upsampling_factor)<=6: 
                pad = 1
            else: 
                pad = 3

        self.global_patch_size = patch_size  # Add to instance variables for global prior generation

        intensity = self.hr_int/np.max(self.hr_int)
        hr_int_patches = view_as_windows(intensity[:, :], window_shape=patch_size, step=1)
        hr_int_centres = hr_int_patches[:, :, int(patch_size[0] / 2), int(patch_size[1] / 2)]

        # Selecting appropriate windows with a lifetime sample at the center
        x_indices = np.arange(0, intensity.shape[0], 1)
        y_indices = np.arange(0, intensity.shape[1], 1)
        xv, yv = np.meshgrid(y_indices, x_indices)
        zv = np.stack([xv, yv]).transpose(1, 2, 0)
        zv_windows = view_as_windows(zv, window_shape=(patch_size[0], patch_size[1], 1), step=1)
        window_centers = zv_windows[:, :, :, int(patch_size[0] / 2), int(patch_size[1] / 2), 0]  # Center positions of the windows
        indices = zv[::self.upsampling_factor[0], ::self.upsampling_factor[1]]  # Indices of the flim samples
        if pad > 1:
            # Apply padding around each flim sample for window selection
            padded_samples = np.zeros(np.append(indices.shape, (pad, pad)))
            for i in range(pad):
                for j in range(pad):
                    padded_samples[:, :, 0, j, i] = indices[:, :, 0] + i - int(np.floor(pad / 2))
                    padded_samples[:, :, 1, j, i] = indices[:, :, 1] + j - int(np.floor(pad / 2))
            indices = padded_samples.transpose(0, 3, 1, 4, 2).reshape(indices.shape[0] * pad, indices.shape[1] * pad, 2)

        comp_samples = indices[:, :, 0] + 1j * indices[:, :, 1]
        comp_centers = window_centers[:, :, 0] + 1j * window_centers[:, :, 1]
        valid_flim_samples = np.isin(comp_samples, comp_centers)
        valid_lr_windows = np.isin(comp_centers, comp_samples)
        flim_limits = np.where(valid_flim_samples)
        flim_shape = (flim_limits[0][-1] - flim_limits[0][0] + 1, flim_limits[1][-1] - flim_limits[1][0] + 1)

        lr_int_patches = hr_int_patches[valid_lr_windows].reshape((flim_shape[0], flim_shape[1], patch_size[0], patch_size[1]))
        assert (hr_int_patches.shape[:2] == window_centers.shape[:2] == hr_int_centres.shape), 'Expect equal number of test inputs'

        lr_tau_centres = np.repeat(np.repeat(self.lr_tau, pad, axis=0), pad, axis=1)
        lr_tau_centres = lr_tau_centres[valid_flim_samples].reshape(flim_shape)
        assert (lr_tau_centres.shape == lr_int_patches.shape[:2]), 'Expect equal number of training inputs and labels'
        
        return lr_int_patches, lr_tau_centres, hr_int_patches

    def local_prior(self, lr_tau_winds, lr_int_winds, loo_cv = False, func_type = 'linear_interp'):
        """
        Compute the local prior based on low-resolution tau and intensity windows using interpolation techniques.

        Parameters:
        - lr_tau_winds (numpy.ndarray): Array of low-resolution tau windows.
        - lr_int_winds (numpy.ndarray): Array of low-resolution intensity windows.
        - loo_cv (bool): Flag indicating whether to perform leave-one-out cross-validation.
        - func_type (str): Type of interpolation function to be used.

        Returns:
        - hr_tau_mu (numpy.ndarray): Computed local prior for high-resolution tau values.

        This method computes the local prior based on the provided low-resolution tau and intensity windows.
        It iterates through the windows, performs interpolation based on the specified function type, and computes the local prior.
        The function type determines the interpolation method used, such as linear, cubic spline, or radial basis function Gaussian process.
        
        If leave-one-out cross-validation (loo_cv) is enabled, it generates the local prior with cross-validation.
        Otherwise, it computes the local prior using all available data.

        If the resulting local prior is contains NaN or infinity values, they are subsequently filled using neighbouring correct values.
        A copy of the computed local prior is stored in the instance.

        Note: The choice of interpolation method depends on the data characteristics and the desired interpolation behavior.
        """

        self.local_exceptions = 0
        print(self.hr_int.shape,self.hr_int.dtype)
        if loo_cv:
            i_ = j_ = 0 #used to address the 'central' position of even window shape. element [0,0] of 2x2 window, [1,1] of 4x4 (like 3x3), [2,2] of 6x6 (like 5x5), etc.
            hr_tau_mu = np.zeros_like(self.lr_tau)
        else: 
            hr_tau_mu = np.ones_like(self.hr_int)
            
        for i in tqdm(range(lr_int_winds.shape[0]+self.local_window_size[0]%2),desc='row number'):
            for j in range(lr_int_winds.shape[1]+self.local_window_size[1]%2):
    
                if loo_cv:                    
                    test_ind = np.array([self.upsampling_factor[0]*min(lr_int_winds.shape[0]-1,i)+self.upsampling_factor[0]*i_,self.upsampling_factor[1]*min(lr_int_winds.shape[1]-1,j)+self.upsampling_factor[1]*j_]).reshape(2,1,1)
                    arr = self.lr_pos_winds[:,min(lr_int_winds.shape[0]-1,i),min(lr_int_winds.shape[1]-1,j)]
                    train_ind = np.not_equal(arr,test_ind).any(axis=0)
                    lr_int_data = lr_int_winds[min(lr_int_winds.shape[0]-1,i),min(lr_int_winds.shape[1]-1,j)][train_ind]
                    lr_tau_data = lr_tau_winds[min(lr_int_winds.shape[0]-1,i),min(lr_int_winds.shape[1]-1,j)][train_ind]
                else:
                    lr_int_data = lr_int_winds[min(lr_int_winds.shape[0]-1,i),min(lr_int_winds.shape[1]-1,j)]
                    lr_tau_data = lr_tau_winds[min(lr_int_winds.shape[0]-1,i),min(lr_int_winds.shape[1]-1,j)]

                z = np.asarray(list(set(sorted(zip(lr_int_data.flatten(),lr_tau_data.flatten()))))) #works for both odd and even window shapes # note: sorted(set()) 
                
                X = z[:,0]
                y = z[:,1]
                Xmax = np.max(X)
                X = X/np.max((Xmax,1e-6)) #avoid div by 0 

                if loo_cv: 
                    Xtest = self.hr_int[test_ind[0],test_ind[1]]/np.max((Xmax,1e-6))
                else:
                    #Works for both odd and even window shapes
                    x_l = max(0,self.upsampling_factor[0]*i-int((self.local_window_size[0]%2)*np.floor(self.upsampling_factor[0]/2)))
                    x_h = min(self.hr_int.shape[0],self.upsampling_factor[0]*i+self.upsampling_factor[0]-int((self.local_window_size[0]%2)*np.floor(self.upsampling_factor[0]/2)))
                    y_l = max(0,self.upsampling_factor[1]*j-int((self.local_window_size[1]%2)*np.floor(self.upsampling_factor[0]/2)))
                    y_h = min(self.hr_int.shape[1],self.upsampling_factor[1]*j+self.upsampling_factor[1]-int((self.local_window_size[1]%2)*np.floor(self.upsampling_factor[0]/2)))
                
                    Xtest = (self.hr_int[x_l:x_h,y_l:y_h]).reshape(-1)
                    Xtest = Xtest/np.max((Xmax,1e-6))

                mu = self.local_fit(X,y,Xtest,func_type)
                
                if loo_cv:
                    hr_tau_mu[min(lr_int_winds.shape[0]-1,i),min(lr_int_winds.shape[1]-1,j),i_*((self.local_window_size[0]+1)%2+1)+j_] = mu
                else:
                    hr_tau_mu[x_l:x_h,y_l:y_h] = mu[:].reshape((x_h-x_l,y_h-y_l))
                
        hr_tau_mu = nan_fill(hr_tau_mu)
        hr_tau_mu = inf_fill(hr_tau_mu)
        
        self.local_prior = hr_tau_mu.copy() #keep a copy in the instance

        return hr_tau_mu
    
    def local_fit(self,X,y,Xtest,func_type = 'linear_interp'):
        """
        Fit local models to the data points and predict values at test points using various interpolation techniques.

        Parameters:
        - X (numpy.ndarray): Array of input data points.
        - y (numpy.ndarray): Array of target values corresponding to the input data points.
        - Xtest (numpy.ndarray): Array of test data points where predictions are to be made.
        - func_type (str): Type of interpolation function to be used.

        Returns:
        - mu (numpy.ndarray): Predicted values at the test points.

        This method fits local models to the data points and predicts values at test points using various interpolation techniques.
        It supports different types of interpolation functions specified by 'func_type', including linear, quadratic, and cubic spline interpolations,
        nearest neighbor interpolation, and radial basis function (RBF) Gaussian process interpolation.

        If 'spline' interpolation is chosen, it uses spline transformation with Ridge regression as the model.
        If 'interp' interpolation is chosen, it uses either nearest neighbor, linear or cubic interpolation functions from SciPy.
        If 'rbf_gp' interpolation is chosen, it uses radial basis function Gaussian process interpolation.

        If an exception occurs during the fitting process, the method increments the 'local_exceptions' counter and returns zero-filled predictions.

        Note: The choice of interpolation method depends on the data characteristics and the desired interpolation behavior.
        """

        try:
            # ================================================================== ===========
            if 'spline' in func_type:
                X = X.reshape(-1,1)
                y = y.reshape(-1,1)
                Xtest = Xtest.reshape(-1,1)
                if 'linear' in func_type: # Suboptimal to define pipeline within the for loop
                    model = make_pipeline(SplineTransformer(n_knots=len(X), knots='quantile', degree=1,extrapolation='constant'), Ridge(alpha=1e-2))
                if 'quadratic' in func_type:    
                    model = make_pipeline(SplineTransformer(n_knots=len(X), knots='quantile', degree=2,extrapolation='constant'), Ridge(alpha=1e-2))
                if 'cubic' in func_type:
                    model = make_pipeline(SplineTransformer(n_knots=len(X), knots='quantile', degree=3,extrapolation='constant'), Ridge(alpha=1e-2))
                model.fit(X, y)
                mu = model.predict(Xtest)
            # =============================================================================
            elif 'interp' in func_type:
                if 'nearest' in func_type:
                    f = interpolate.interp1d(X, y, kind='nearest',fill_value='extrapolate')
                if 'linear' in func_type:
                    f = interpolate.interp1d(X, y, kind='linear',fill_value='extrapolate')
                elif 'cubic' in func_type:
                    f = interpolate.interp1d(X, y, kind='cubic',fill_value='extrapolate')
                mu = f(Xtest)
                mu[Xtest.flatten()>np.max(X)] = y[np.argmax(X)]
                mu[Xtest.flatten()<np.min(X)] = y[np.argmin(X)]
            # =============================================================================                            
            elif func_type=='rbf_gp':                                
                X = X.reshape(-1,1)
                Xtest = Xtest.reshape(-1,1)
                y = y.astype('float64')
                s = 0.01
                N = len(X)                                               
                K = kernel(X, X)
                L = np.linalg.cholesky(K + (s+1e-6)*np.eye(N)) # gaussian prior          
                K_prime = kernel(X,Xtest)     
                Lk = np.linalg.solve(L, K_prime) 
                Ly = np.linalg.solve(L, y)
                mu = np.matmul(Lk.T, Ly).flatten()
                mu[Xtest.flatten()>np.max(X)] = y[np.argmax(X)]
                mu[Xtest.flatten()<np.min(X)] = y[np.argmin(X)]

        except:
            self.local_exceptions += 1
            mu = np.zeros_like(Xtest)
        return mu

    def global_prior(self, lr_int_patches, lr_tau_centres, hr_int_patches, model_v='v6', epochs = 150, augment=True):
        """
        Generate the global prior based on the provided low-resolution intensity patches and corresponding tau centres.

        Parameters:
        - lr_int_patches (numpy.ndarray): Low-resolution intensity patches.
        - lr_tau_centres (numpy.ndarray): Tau centres corresponding to the low-resolution intensity patches.
        - hr_int_patches (numpy.ndarray): High-resolution intensity patches.
        - model_v (str): Version of the model architecture to use.
        - epochs (int): Number of training epochs.
        - augment (bool): Whether to perform data augmentation.

        Returns:
        - prediction (numpy.ndarray): Predicted global prior.

        This method generates the global prior using the provided low-resolution intensity patches and corresponding tau centres.
        It constructs the training and testing datasets, normalizes the intensity values, and performs data augmentation if specified.
        The data is then trained using the specified deep learning model architecture.
        The trained model is used to predict the global prior for the high-resolution intensity patches.

        The predicted global prior is clipped to ensure it falls within the specified tau limits.
        """

        train_windowed_intens = lr_int_patches.reshape(-1,self.global_patch_size[0],self.global_patch_size[1])#.reshape(-1,window_shape[0]*window_shape[1])
        test_windowed_intens = hr_int_patches.reshape(-1,self.global_patch_size[0],self.global_patch_size[1])#.reshape(-1,window_shape[0]*window_shape[1])

        x_train = np.stack([train_windowed_intens,train_windowed_intens],axis=-1,dtype='float16')
        x_test = np.stack([test_windowed_intens,test_windowed_intens],axis=-1,dtype='float16')

        # Normalise
        c_ = np.max(train_windowed_intens,axis=(1,2),keepdims=True)
        c_[c_==0] = 1e-6
        x_train[:,:,:,0] = x_train[:,:,:,0]/c_
        
        c_ = np.max(test_windowed_intens,axis=(1,2),keepdims=True)
        c_[c_==0] = 1e-6
        x_test[:,:,:,0] = x_test[:,:,:,0]/c_

        lr_tau_centres = lr_tau_centres.flatten()

        # Data augmentation
        if augment==True:
            x_train_aug = np.stack([x_train]*8,axis=1)
            for j in range(4):
                x_train_aug[:,j,...] = np.rot90(x_train,axes=(1,2),k=j)
            for j in range(4):
                x_train_aug[:,j+4,...] = np.flip(np.rot90(x_train,axes=(1,2),k=j),axis=1)
            x_train_aug = x_train_aug.reshape(-1,x_train.shape[1],x_train.shape[2],x_train.shape[3])
            x_train = x_train_aug
            lr_tau_centres = np.repeat(lr_tau_centres,8,0)

        y_train = lr_tau_centres.reshape(-1,1)

        # Train the model
        encoder = eval('cnn_utils.Architectures(self.global_patch_size).{}()'.format(model_v))
        encoder.compile(loss='mae',
                        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4))
        # history = encoder.fit(x=x_train, y = y_train, epochs = epochs, shuffle=True, batch_size=100)
        
        # # Make predictions
        # prediction = encoder.predict(x_test,batch_size=1000)
        # prediction = prediction.reshape(hr_int_patches.shape[0],hr_int_patches.shape[1])
        # prediction[prediction<0] = 0

        # self.global_prior=prediction.copy()
        # return prediction, history.history
        
        run_matrix = np.zeros((512,512,151))
        class TestCallback(tf.keras.callbacks.Callback):
            def __init__(self, test_dataset):
                super().__init__()
                self.test_dataset = test_dataset
            def on_epoch_begin(self,epoch,logs=None):
                prediction = self.model.predict(x_test,batch_size=1000)
                prediction = prediction.reshape(hr_int_patches.shape[0],hr_int_patches.shape[1])
                prediction[prediction<0] = 0
                run_matrix[6:-6,6:-6,epoch]=prediction
        
        history = encoder.fit(x=x_train, y = y_train, epochs = epochs, shuffle=True, batch_size=100,callbacks=[TestCallback(x_test)])
        
        # Make predictions
        prediction = encoder.predict(x_test,batch_size=1000)
        prediction = prediction.reshape(hr_int_patches.shape[0],hr_int_patches.shape[1])
        prediction[prediction<0] = 0
    
        np.save(r'E:\OneDrive - University of Glasgow\mega-FLIM\FLIPPER\gp_frames.npy',run_matrix)
        
        self.global_prior=prediction.copy()
        return prediction, None#history.history
       
    def admm_loop(self,local_prior,global_prior, ADMM_iter = 20):
        """
        Apply the Alternating Direction Method of Multipliers (ADMM) to optimize the reconstruction.

        Parameters:
        - local_prior (numpy.ndarray): Local prior data.
        - global_prior (numpy.ndarray): Global prior data.

        Returns:
        - g0 (numpy.ndarray): Optimized reconstruction.

        This method applies the Alternating Direction Method of Multipliers (ADMM) to optimize the reconstruction.
        It pads the global prior to match the size of the local prior and creates a mask to inform the ADMM that
        the padded regions do not contribute to the loss.

        The optimization involves several steps:
        1. Initialization of variables and parameters.
        2. Iterative optimization using gradient descent.
        3. Primal update for z via soft thresholding.
        4. Dual update for y.

        The optimization proceeds for a fixed number of ADMM iterations.

        The final optimized reconstruction 'g0' is returned.
        """

        # Pad global prior to match size of local prior. We create a mask to inform the ADMM that the padded regions do not contribute to the ADMM loss
        if 'global_patch_size' not in self.__dict__:
            global_patch_size = (local_prior.shape[0]-global_prior.shape[0]+1,local_prior.shape[1]-global_prior.shape[1]+1) #Calculate patch size to allow ADMM to be used flexibly on different data
        else:
            global_patch_size = self.global_patch_size
        mask = np.ones_like(global_prior)
        mask = np.pad(mask,(int(global_patch_size[0]/2),int(global_patch_size[1]/2)),'constant',constant_values = (0,0))
        global_prior = np.pad(global_prior,(int(global_patch_size[0]/2),int(global_patch_size[1]/2)),'constant',constant_values = (0,0))
        
        # Ensure non-negativity of local and global priors
        local_prior[local_prior<0] = 0
        global_prior[global_prior<0] = 0

        # Upsampling factor
        k = self.upsampling_factor

        # Dimensions of the high-resolution image
        row, col = (self.hr_int).shape

        # Define the downsampling function
        def SparseSample(Img,k):
            LR = Img[::k[0],::k[1]]
            return(LR)

        # Define the forward difference operator for evaluating TV
        def Forw_Diff(img):
            return np.stack((np.roll(img,1,axis = 0)-img, np.roll(img,1,axis = 1)-img),axis =2)

        # Define the adjoint of the TV evaluating operator
        def Back_Diff(U_s):
            U_x = U_s[...,0]
            U_y = U_s[...,1]
            Z = (np.roll(U_x,-1,axis = 0)-U_x)+(np.roll(U_y,-1,axis=1)-U_y)
            return Z
        
        # Initial guess
        g0 = np.repeat(np.repeat(self.lr_tau,k[0],axis = 0),k[1],axis = 1)
        y = np.zeros((row,col,2))
        z = np.zeros((row,col,2))
        rho = 2
        beta = 0.001
        GD_iter = 90
        gamma = 0.5
        delta = 0.1
        u  = y/rho

        run_matrix = np.zeros((512,512,20))

        # Main ADMM loop
        for iterations in tqdm(range (ADMM_iter),'ADMM'):
            run_matrix[:,:,iterations] = g0
            u = y/rho
            stp_sz = 1
            p = np.zeros(GD_iter,)

            # Gradient descent iterations
            for iters2 in range(GD_iter):

                v = z-u
                C1 = SparseSample(g0,k)-self.lr_tau 
                C1 = 0.5*((np.linalg.norm(C1.flatten()))**2)
                
                U_s = Forw_Diff(g0)
                C2 = U_s-v             
                C2 = rho*0.5*((np.linalg.norm(C2.flatten()))**2)         
            
                C3 = mask*(g0-global_prior)         
                C3 = gamma*0.5*((np.linalg.norm(C3.flatten()))**2)
                
                C4 = (g0-local_prior)
                C4 = delta*0.5*((np.linalg.norm(C4.flatten()))**2)
                
                Cost_fun = C1 + C2 + C3 + C4

                dC1 = SparseSample(g0,k)-self.lr_tau         
                Z = np.zeros((row,col))
                Z[::k[0],::k[0]] = dC1  
                dC1 = Z
                
                dC2 = rho*Back_Diff(U_s-v)
        
                dC3 = gamma*(mask*(g0-global_prior))
                
                dC4 = delta*(g0-local_prior)
                
                
                dC = dC1 + dC2 + dC3 + dC4

                donet = 0
                while donet == 0:
                    g0new = g0 - stp_sz*dC
                    g0new = np.maximum(g0new,0)
                    
                    C1new = (SparseSample(g0new,k)-self.lr_tau)
                    C1new = 0.5*((np.linalg.norm(C1new.flatten()))**2)
                    
                    C2new = Forw_Diff(g0new)-v
                    C2new = rho*0.5*((np.linalg.norm(C2new.flatten()))**2) 
                    
                    C3new = mask*(g0new-global_prior)
                    C3new = gamma*0.5*((np.linalg.norm(C3new.flatten()))**2)
                    
                    C4new = (g0new-local_prior)
                    C4new = delta*0.5*((np.linalg.norm(C4new.flatten()))**2) 
                    
                    Cost_fun_new  = C1new + C2new +C3new + C4new
                    
                    if Cost_fun_new < Cost_fun or stp_sz<1e-20:
                        donet = 1
                    else:
                        stp_sz = stp_sz/10
                        
                g0 = g0new
                p[iters2] = Cost_fun_new
                Cost_fun = Cost_fun_new

            s = Forw_Diff(g0)+(y/rho)
            z = np.sign(s)*np.maximum(0,(abs(s)-(beta/rho)))
            y = y+rho*(Forw_Diff(g0)-z)

        self.hr_tau_mean_estimate = g0

        np.save(r'E:\OneDrive - University of Glasgow\mega-FLIM\FLIPPER\admm_frames.npy',run_matrix)

        return g0
    
    def local_pipeline_segment(self,window_size=(5,5),func_type='linear_interp'):
        """End-to-end pipeline to create local prior, call directly after instantiating SiSIFUS on data"""
        self.data_clean()
        lr_tau_winds, lr_int_winds = self.local_data_prep(window_size=window_size)
        local_prior = self.local_prior(lr_tau_winds, lr_int_winds,func_type=func_type)
        return local_prior

    def global_pipeline_segment(self,pad=None,epochs=150,patch_size=(13, 13)):
        """End-to-end pipeline to create global prior, call directly after instantiating SiSIFUS on data"""
        self.data_clean()
        lr_int_patches, lr_tau_centres, hr_int_patches = self.global_data_prep(patch_size=patch_size,pad=pad)
        global_prior, history = self.global_prior(lr_int_patches, lr_tau_centres, hr_int_patches, epochs=epochs)
        return global_prior, history

#%% Other functions
try: loss_fn
except: loss_fn = LPIPS(net='alex').double()

def perceptual_loss(img0, img1):
    """https://github.com/richzhang/PerceptualSimilarity"""
    if len(img0.shape)==3 and len(img1.shape)==3:
        img0 = img0.transpose(2,0,1)
        img1 = img1.transpose(2,0,1)
        img0 = img0.reshape(1,3,img0.shape[1],img0.shape[2])
        img1 = img1.reshape(1,3,img1.shape[1],img1.shape[2])
    else: 
        img0 = img0.reshape(1,img0.shape[0],img0.shape[1])
        img1 = img1.reshape(1,img1.shape[0],img1.shape[1])
    img0 = img0 - np.min(img0)
    img0 = 2*(img0/np.max(img0))-1
    img1 = img1 - np.min(img1)
    img1 = 2*(img1/np.max(img1))-1

    img0 = torch.from_numpy(img0).double()
    img1 = torch.from_numpy(img1).double()

    return loss_fn(img0, img1).item()

def psnr(img_pred, img_ref, tau_limits):
    return 20*np.log10((tau_limits[1]-tau_limits[0])**2/(np.sqrt(np.mean(((img_pred - img_ref)[img_ref>0])**2))))


def hist_mean_std(n, centres):
    """Returns mean and std of distribution captured in a histogram"""
    mean = np.dot(n,centres)
    std = np.sqrt(np.sum(n*(centres-mean)**2))
    return mean, std

def windowing_func(input_data,window_shape):
    """
    Preprocesses input data for local prior generation.

    Parameters:
    - input_data (numpy.ndarray): The input data to be preprocessed.
    - window_shape (tuple): Shape of the window to be applied for processing.
    - step (int): Step size for the windowed data.

    Returns:
    - lr_input_data (numpy.ndarray): Preprocessed input data with appropriate padding.

    This function preprocesses the input data by applying a windowed view and padding
    it appropriately to handle edge cases. It pads the input data with reflected values
    to ensure that the windowed input data covers the entire input space.
    """
    windowed_input_data = view_as_windows(input_data,window_shape,step=1)
    
    lr_input_data = np.zeros((input_data.shape[0],input_data.shape[1],windowed_input_data.shape[2],windowed_input_data.shape[3]))
    
    lr_input_data[int(np.floor((window_shape[0]-1)/2)):int(np.floor((window_shape[0]-1)/2))+windowed_input_data.shape[0],
         int(np.floor((window_shape[1]-1)/2)):int(np.floor((window_shape[1]-1)/2))+windowed_input_data.shape[1],:] = windowed_input_data
    for i in range(int(np.floor((window_shape[0]-1)/2))): #left
        lr_input_data[i,:,:] = lr_input_data[int(np.floor((window_shape[0]-1)/2)),:,:] 
    for i in range(int(np.floor((window_shape[0]-1)/2))+windowed_input_data.shape[0],lr_input_data.shape[0]): #right
        lr_input_data[i,:,:] = lr_input_data[int(np.floor((window_shape[0]-1)/2))+windowed_input_data.shape[0]-1,:,:] 
    for i in range(int(np.floor((window_shape[1]-1)/2))): #top
        lr_input_data[:,i,:] = lr_input_data[:,int(np.floor((window_shape[1]-1)/2)),:] 
    for i in range(int(np.floor((window_shape[1]-1)/2))+windowed_input_data.shape[1],lr_input_data.shape[1]): #bottom
        lr_input_data[:,i,:] = lr_input_data[:,int(np.floor((window_shape[1]-1)/2))+windowed_input_data.shape[1]-1,:] 
    return lr_input_data


def local_contrast_enhancement(data,clipLimit=2,tileGridSize=(8,8),times=2):
    """Applies cv2 local contrast enhancement to an image"""
    data = data.astype('float32')
    data = 255*data/np.max(data)
    data = data.astype('uint8')        
    clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize=tileGridSize)
    clahedata = clahe.apply(data)#.astype('float32')
    for i in range(times-1):
        clahedata = clahe.apply(clahedata)#.astype('float32')
    clahedata = clahedata.astype('float32')
    return clahedata   
    
    
def data_fill(data):
    """Fills in the 0s in an image with the nearest non-zero pixel"""
    mask = np.where(~((data == 0)*np.isnan(data)*np.isinf(data)))
    interp = interpolate.NearestNDInterpolator(np.transpose(mask), data[mask])
    image_result = interp(*np.indices(data.shape))
    return image_result

def nan_fill(data):
    """Fills in the NaNs in an image with the nearest non-zero pixel"""
    mask = np.where(~(np.isnan(data)))
    interp = interpolate.NearestNDInterpolator(np.transpose(mask), data[mask])
    image_result = interp(*np.indices(data.shape))
    return image_result                

def inf_fill(data):
    """Fills in the NaNs in an image with the nearest non-zero pixel"""
    mask = np.where(~(np.isinf(data)))
    interp = interpolate.NearestNDInterpolator(np.transpose(mask), data[mask])
    image_result = interp(*np.indices(data.shape))
    return image_result  

def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1 #0.01 seems pretty stable
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)



def pad_signals(s1, s2):
    """For deconvolving phasor signal"""
    size = s1.size +s2.size - 1
    size = int(2 ** np.ceil(np.log2(size)))
    s1 = np.pad(s1, ((size-s1.size)//2, int(np.ceil((size-s1.size)/2))), 'constant', constant_values=(0, 0))
    s2 = np.pad(s2, ((size-s2.size)//2, int(np.ceil((size-s2.size)/2))), 'constant', constant_values=(0, 0))
    return s1, s2

def decon_fourier_ratio(signal, removed_signal):
    """For deconvolving phasor signal"""
    signal, removed_signal = pad_signals(signal, removed_signal)
    recovered = np.fft.fftshift(np.fft.ifft(np.fft.fft(signal)/np.fft.fft(removed_signal)))
    return np.real(recovered)

def decon_fourier_signal(datacube,irf):
    """Deconvolves an IRF from a datacube, returns the deconvolved signal (full size, so the centre of the signal is at the middle)"""
    tr = np.zeros((256,datacube.shape[1],datacube.shape[2]))
    global_sum = np.mean(np.mean(datacube,axis=0),axis=0)
    assert(np.argmax(global_sum)>10),'The code does background removal first. This makes the reconstructions a lot less noisy'
    bgd = np.mean(datacube[:5,:,:]/np.max(global_sum)) 
    for i in range(datacube.shape[1]):
        for j in range(datacube.shape[2]):
            gt = datacube[:,i,j]/np.max(datacube[:,i,j])-bgd
            tr[:,i,j] = decon_fourier_ratio(gt, irf)#[92:92+72]
    return tr

def phasor_lifetime(datacube, fr, irf, deconvolve_irf=False):    
    """
    Calculate the ratio of predicted tau values based on the provided datacube.

    Parameters:
    - datacube (numpy.ndarray): 3D array representing the input datacube.
    - fr (float): Sampling frame interval.
    - irf (numpy.ndarray, optional): Impulse response function (IRF) for deconvolution.
    - deconvolve_irf (bool, optional): Whether to deconvolve the IRF from the datacube.

    Returns:
    - tau_pred_ratio (numpy.ndarray): Array containing the ratio of predicted tau values.

    This function calculates the ratio of predicted tau values based on the input datacube.
    If 'deconvolve_irf' is set to True, it deconvolves the IRF from the datacube.
    It then performs Fourier analysis on the datacube to extract real and imaginary parts of the
    Fourier transform at a specific angular frequency. Finally, it computes the ratio of predicted
    tau values and returns the result.
    """
    if deconvolve_irf==True:
        datacube = decon_fourier_signal(datacube,irf)
    global_sum = np.sum(np.sum(datacube,axis=1),axis=1)
    data = datacube[np.argmax(global_sum):,:,:].copy()
    t_data = np.arange(fr,fr*(data.shape[0]+1)-1e-12,fr) #time 
    freq = 1/(fr*len(t_data)) #frequency is the inverse of the decay period. This period is arbitrary, but it's best to keep it as long as the full decay 
    omega = 2*np.pi*freq #angular frequency = 2pi*rep rate of laser
    g = np.zeros((data.shape[1],data.shape[2]))
    s = np.zeros((data.shape[1],data.shape[2]))
    intens = np.sum(data,axis=0)
    #Find the real and imaginary parts of the FT, evaluated at omega.
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if intens[i,j]!=0:
                g[i,j] = np.sum(data[:,i,j]*np.cos(omega*t_data))/intens[i,j]
                s[i,j] = np.sum(data[:,i,j]*np.sin(omega*t_data))/intens[i,j]
    tau_pred_ratio = s/(g*omega)*10**9
    return tau_pred_ratio

    

    