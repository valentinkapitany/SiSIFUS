from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.patches as patches

my_vis_dict = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "font.size": 20,
    "axes.labelsize": 20,
    "ytick.labelsize": 20,
    "xtick.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    'axes.grid': False,
    'axes.prop_cycle': plt.cycler('color', ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']),
    "figure.titlesize":   20,
    'svg.fonttype' : 'none'
}

def vis_params(name):
    """
    Retrieve visualization windows and region of interest parameters based on the provided name.

    Parameters:
    - name (str): The name of the visualization window configuration.

    Returns:
    - vis_windows (list of list): List of window coordinates for visualization.
    - roi_co (list): Coordinates of the region of interest.
    - roi_s (int): Size of the region of interest.

    Note:
    - For TRIMSCOPE_Rac_Raichu:
        - Visualization windows are downsampled by 16x16.
        - ROI coordinates: [75, 120]
        - ROI size: 25
    - For Flimera_Convallaria_Acridine_Orange:
        - Visualization windows are downsampled by 8x8.
        - ROI coordinates: [80, 50]
        - ROI size: 30
    - For Flimera_Rac_Raichu:
        - Visualization windows are downsampled by 8x8.
        - ROI coordinates: [150, 30]
        - ROI size: 20
    - For TRIMSCOPE_FLIPPER_2:
        - Visualization windows are downsampled by 16x16.
        - ROI coordinates: [190, 260]
        - ROI size: 50
    """

    if name=='TRIMSCOPE_Rac_Raichu':
        vis_windows = [[2,7], [5,4]]  
        roi_co = [75, 120]
        roi_s = 25

    if name=='Flimera_Convallaria_Acridine_Orange':
        vis_windows = [[7, 4], [21, 1]] 
        roi_co = [80, 50]
        roi_s = 30

    if name=='Flimera_Rac_Raichu':
        vis_windows = [[7, 3], [4, 16]] 
        roi_co = [150, 30] 
        roi_s = 20

    if name=='TRIMSCOPE_FLIPPER_2':
        """Top left = (16,16). Downsampling factor (16x16). Assumes pad = 1"""
        vis_windows = [[21, 6], [20, 4]]
        roi_co = [190, 260]
        roi_s = 50      
        

    return vis_windows, roi_co, roi_s

def intensity_weighting(colors, weighting, limits, cmap, intensfactor):
    colors = Normalize(limits[0],limits[1], clip=True)(colors)
    colors = cmap(colors)[:,:,:3]
    weighting = intensfactor*weighting/np.max(weighting.flatten())
    weighting[weighting>1] = 1
    for i in range(3):
            colors[:,:,i]=colors[:,:,i]*(weighting)
    return colors, weighting    

def two_image_slider(image_options,image1_key,image2_key):
    image1 = image_options[image1_key]
    image2 = image_options[image2_key]

    # Resize images to have the same height
    rows, cols = image1.shape[:2]

    # Create figure and axes
    ratio = cols / rows

    fig = plt.figure(figsize=(ratio * 8*1.25, 8),layout='constrained')
    ax = fig.add_axes([0.1,0.1,0.65,0.85])
    plt.title('Drag red slider')
    
    combined_image = np.hstack((image1[:, :int(cols / 2)], image2[:, int(cols / 2):]))
    img_obj = ax.imshow(combined_image, cmap='gray')
    # plt.axis('off')

    l, b, w, h = ax._position.bounds
    ax_slider = fig.add_axes([l, b, w, h])
    ax_slider.patch.set_alpha(0)
    ax_radio1 = fig.add_axes([0.8, 0.8, 0.15, 0.025*len(image_options.keys())])
    ax_radio2 = fig.add_axes([0.8, 0.4, 0.15, 0.025*len(image_options.keys())])

    # Create slider
    slider = Slider(ax_slider, '', 0, 1, valinit=0.5, facecolor ='#00000000', track_color = '#00000000', initcolor = '#00000000',  handle_style = {'facecolor':'#FF0000FF'})
    slider.valtext.set_visible(False)

    # Create radio buttons
    radio1 = RadioButtons(ax_radio1, list(image_options.keys()), active=list(image_options.keys()).index(image1_key))
    radio2 = RadioButtons(ax_radio2, list(image_options.keys()), active=list(image_options.keys()).index(image2_key))

    # Initialize vertical line for slider position
    line = ax.axvline(combined_image.shape[1] / 2, color='r')

    # Update function for the slider
    def update(val):
        slider_val = slider.val
        width = combined_image.shape[1]
        split_point = int(slider_val * width)
        # Update position of vertical line
        line.set_xdata([split_point, split_point])
        # Split the combined image based on the slider value
        display_image = np.hstack((image1[:, :split_point], image2[:, split_point:]))
        img_obj.set_data(display_image)
        fig.canvas.draw_idle()

    def on_radio1_clicked(label):
        nonlocal image1_key, image1
        image1_key = label
        image1 = image_options[label]
        update(None)

    def on_radio2_clicked(label):
        nonlocal image2_key, image2
        image2_key = label
        image2 = image_options[label]
        update(None)

    slider.on_changed(update)
    radio1.on_clicked(on_radio1_clicked)
    radio2.on_clicked(on_radio2_clicked)
    
    return locals()

def show_local_patches(sisifus,data_str,tau_limits):
    sisifus.data_clean()
    lr_tau_winds, lr_int_winds = sisifus.local_data_prep()
    vis_windows,  _, _ = vis_params(data_str)

    wsv = 6
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(1,2),dpi=100)

    fig.subplots_adjust(0,0,1,1,0,0.05)

    y = lr_tau_winds[vis_windows[0][1]+1,vis_windows[0][0]+1]
    lr_tau_winds[lr_tau_winds<tau_limits[0]] = tau_limits[0]
    lr_tau_winds[lr_tau_winds>tau_limits[1]] = tau_limits[1]
    ylim = [np.min((np.min(lr_tau_winds[vis_windows[0][1]+1,vis_windows[0][0]+1]),np.min(lr_tau_winds[vis_windows[1][1]+1,vis_windows[1][0]+1])))-2e-2,
            np.max((np.max(lr_tau_winds[vis_windows[0][1]+1,vis_windows[0][0]+1]),np.max(lr_tau_winds[vis_windows[1][1]+1,vis_windows[1][0]+1])))+2e-2]
    ax[0].scatter(lr_int_winds[vis_windows[0][1]+1,vis_windows[0][0]+1], y)
    ax[0].set_ylim(ylim)
    ax[0].grid(axis='y')
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(2))

    y = lr_tau_winds[vis_windows[1][1]+1,vis_windows[1][0]+1]
    y[y>3.5] =3.5
    ax[1].scatter(lr_int_winds[vis_windows[1][1]+1,vis_windows[1][0]+1], y)
    ax[1].set_xlabel('counts',labelpad=-5)
    ax[1].set_ylim(ylim)
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(2))
    ax[1].grid(axis='y')
    ax[1].set_ylabel('lifetime (ns)')

    plt.setp(ax[1].get_xticklabels(), rotation=20)
    plt.show(block=False)
    plt.pause(0.01)

def show_global_patches(sisifus,data_str):
    sisifus.data_clean()
    lr_int_patches, lr_tau_centres, _ = sisifus.global_data_prep(pad=1)

    vis_windows,_,_ = vis_params(data_str)
    vis_windows = vis_windows

    wsv = 6
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(1,2),dpi=100)
    ax[0].imshow(lr_int_patches[vis_windows[0][1],vis_windows[0][0]],cmap='gray')
    ax[0].axis('off')
    ax[0].scatter(wsv,wsv,c=lr_tau_centres[vis_windows[0][1],vis_windows[0][0]],vmin=sisifus.tau_limits[0],vmax=sisifus.tau_limits[1],cmap='jet')
    ax[1].imshow(lr_int_patches[vis_windows[1][1],vis_windows[1][0]],cmap='gray')
    ax[1].axis('off')
    ax[1].scatter(wsv,wsv,c=lr_tau_centres[vis_windows[1][1],vis_windows[1][0]],vmin=sisifus.tau_limits[0],vmax=sisifus.tau_limits[1],cmap='jet')
    plt.show(block=False)
    plt.pause(0.01)
