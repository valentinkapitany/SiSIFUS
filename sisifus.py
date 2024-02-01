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
from mpld3 import plugins
import mpld3
#%% Load data
import importlib
importlib.reload(data_utils)
# Set constants
cmap = plt.cm.jet
epochs = 150

# Load data using data_utils.load function
data_str = 'Flimera_Rac_Raichu' # TRIMSCOPE_FLIPPER_2, Flimera_Convallaria_Acridine_Orange, TRIMSCOPE_Rac_Raichu, Flimera_Rac_Raichu, TRIMSCOPE_FLIPPER <- used only for validation
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
importlib.reload(ir_utils)

df = 8
# For validation, we downsample the high-resolution FLIM ground truth image.
# For testing, load the low-resolution FLIM image directly

lr_tau = hr_tau[::df, ::df]

# Initialize SiSIFUS object with low-resolution tau and high-resolution intensity data
sisifus = ir_utils.SiSIFUS(lr_tau, hr_int, tau_limits=tau_limits)

# # # Generate priors and use them to perform ADMM
# local_prior = sisifus.local_pipeline_segment(window_size=(5,5))
# global_prior, history = sisifus.global_pipeline_segment(epochs=epochs,patch_size=(13,13))
# hr_tau_estimate = sisifus.admm_loop(local_prior,global_prior,ADMM_iter = 20)

# savemat('data/intermediate/TRIMSCOPE_FLIPPER/local_prior_upsampling_{}x{}.mat'.format(df,df),{'local_prior':local_prior})
# savemat('data/intermediate/TRIMSCOPE_FLIPPER/global_prior_upsampling_{}x{}.mat'.format(df,df),{'tau_mean':global_prior,'history':history})
# savemat('data/processed/TRIMSCOPE_FLIPPER_upsampling_{}x{}.mat'.format(df,df),{'hr_tau_estimate':hr_tau_estimate})

#%%
# Data can also be loaded in from repositories
local_prior = loadmat(os.path.join('data/intermediate', data_str, 'local_prior_upsampling_{}x{}.mat'.format(df,df)))['local_prior']
global_prior = loadmat(os.path.join('data/intermediate', data_str, 'global_prior_upsampling_{}x{}.mat'.format(df,df)))['tau_mean']
hr_tau_estimate = loadmat(os.path.join('data/processed','{}_upsampling_{}x{}.mat'.format(data_str,df,df)))['hr_tau_estimate']
#%% Visualise results
import importlib
importlib.reload(data_vis_utils)
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


#%%








# plugins.connect(fig, SliderView(line, callback_func="updateSlider"))
# plugins.connect(fig, plugins.MousePosition(fontsize=14))
# mpld3.save_html(fig,'test_fig.html')
# plt.show()
# #%%
# import matplotlib.pyplot as plt, mpld3
# from mpld3 import plugins
# fig = plt.figure()
# ax = fig.add_axes([0.1,0.1,0.9,0.9])
# ax.plot([3,1,4,1,5], 'ks-', mec='w', mew=5, ms=20)
# plugins.connect(fig, plugins.MousePosition(fontsize=14))
# mpld3.save_html(fig,'test_fig.html')
# mpld3.plugins.connect(fig, SliderView(line, callback_func="updateSlider"))
# #%%
# class SliderView(mpld3.plugins.PluginBase):
#     """ Add slider and JavaScript / Python interaction. """

#     JAVASCRIPT = """
#     mpld3.register_plugin("sliderview", SliderViewPlugin);
#     SliderViewPlugin.prototype = Object.create(mpld3.Plugin.prototype);
#     SliderViewPlugin.prototype.constructor = SliderViewPlugin;
#     SliderViewPlugin.prototype.requiredProps = ["idline", "callback_func"];
#     SliderViewPlugin.prototype.defaultProps = {}

#     function SliderViewPlugin(fig, props){
#         mpld3.Plugin.call(this, fig, props);
#     };

#     SliderViewPlugin.prototype.draw = function(){
#       var line = mpld3.get_element(this.props.idline);
#       var callback_func = this.props.callback_func;

#       var div = d3.select("#" + this.fig.figid);

#       // Create slider
#       div.append("input").attr("type", "range").attr("min", 0).attr("max", 10).attr("step", 0.1).attr("value", 1)
#           .on("change", function() {
#               var command = callback_func + "(" + this.value + ")";
#               console.log("running "+command);
#               var callbacks = { 'iopub' : {'output' : handle_output}};
#               var kernel = IPython.notebook.kernel;
#               kernel.execute(command, callbacks, {silent:false});
#           });

#       function handle_output(out){
#         //console.log(out);
#         var res = null;
#         // if output is a print statement
#         if (out.msg_type == "stream"){
#           res = out.content.data;
#         }
#         // if output is a python object
#         else if(out.msg_type === "pyout"){
#           res = out.content.data["text/plain"];
#         }
#         // if output is a python error
#         else if(out.msg_type == "pyerr"){
#           res = out.content.ename + ": " + out.content.evalue;
#           alert(res);
#         }
#         // if output is something we haven't thought of
#         else{
#           res = "[out type not implemented]";  
#         }

#         // Update line data
#         line.data = JSON.parse(res);
#         line.elements()
#           .attr("d", line.datafunc(line.data))
#           .style("stroke", "black");

#        }

#     };
#     """

#     def __init__(self, line, callback_func):
#         self.dict_ = {"type": "sliderview",
#                       "idline": mpld3.utils.get_id(line),
#                       "callback_func": callback_func}
        
    
# import numpy as np

# def updateSlider(val1):
#     t = np.linspace(0, 10, 500)
#     y = np.sin(val1*t)
#     return map(list, list(zip(list(t), list(y))))

# fig, ax = plt.subplots(figsize=(8, 4))

# t = np.linspace(0, 10, 500)
# y = np.sin(t)
# ax.set_xlabel('Time')
# ax.set_ylabel('Amplitude')

# # create the line object
# line, = ax.plot(t, y, '-k', lw=3, alpha=0.5)
# ax.set_ylim(-1.2, 1.2)
# ax.set_title("Slider demo")

# mpld3.plugins.connect(fig, SliderView(line, callback_func="updateSlider"))
# mpld3.save_html(fig,'test_fig.html')
#%%