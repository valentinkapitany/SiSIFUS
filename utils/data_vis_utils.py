from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons


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

def vis_windows(name):

    if name=='TRIMSCOPE_Rac_Raichu':
        #downsampling 16x16
        vis_windows = [[2,7], [5,12], [5,4]]  

    if name=='Flimera_Convallaria_Acridine_Orange':
        #downsampling 8x8
        vis_windows =  None
        vis_windows = [[6,0], [7, 4], [21, 1]] 

    if name=='Flimera_Rac_Raichu':
        #downsampling 8x8
        vis_windows = [[7, 3], [4, 16], [10, 1]] 

    if name=='TRIMSCOPE_FLIPPER_2':
        #downsampling 16x16
        """Top left = (16,16). Downsampling factor (16x16). Window centres in original coords (x,y): (384, 208) - 1.5ns, (160, 112) - 2.3ns, (336,80) - 3ns"""
        vis_windows = [[21, 6], [14, 6], [20, 4]]

    return vis_windows

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
