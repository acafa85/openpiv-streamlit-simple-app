import streamlit as st
from openpiv import tools, pyprocess, validation, filters, scaling 
import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.title('OpenPIV Streamlit App')

# Upload the images
image1 = st.file_uploader("Choose an Image 1...")
image2 = st.file_uploader("Choose an Image 2...")

if image1 is not None and image2 is not None:
    # Load images into numpy arrays
    frame_a = tools.imread(image1)
    frame_b = tools.imread(image2)

    # Perform PIV analysis
    # winsize = 32 # pixels, interrogation window size in frame A
    winsize = st.slider("Window size", min_value=8, max_value=64, value=32, step=8)
    st.write(f"Window size: {winsize}")
    searchsize = winsize + 8  # pixels, search in image B
    overlap = winsize//2 # pixels, 50% overlap
    dt = 0.1 # sec, time interval between pulses
    # st.write(f"Window size: {winsize} {searchsize},{overlap}")


    u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), 
                                                        frame_b.astype(np.int32), 
                                                        window_size=winsize, 
                                                        overlap=overlap, 
                                                        dt=dt, 
                                                        search_area_size=searchsize, 
                                                        sig2noise_method='peak2peak')
    
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, 
                                 search_area_size=searchsize, 
                                 overlap=overlap )
    
    flags = validation.sig2noise_val( sig2noise, 
                                 threshold = 1.1 )
    
    u2, v2 = filters.replace_outliers( u0, v0, 
                                   flags,
                                   method='localmean', 
                                   max_iter=3, 
                                   kernel_size=3)
    x, y, u3, v3 = scaling.uniform(x, y, u2, v2, 
                               scaling_factor = 1 ) # 96.52 microns/pixel
    
    x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)
    
    
    
    # Create a figure and plot vector field
    fig, ax = plt.subplots(1, figsize=(10,10))
    # image_rescaled = rescale(frame_a, 1/96.52, anti_aliasing=True)
    xmax = np.amax(x) + winsize / 2
    ymax = np.amax(y) + winsize / 2
    ax.imshow(frame_a, cmap="Greys_r", extent=[0.0, xmax, 0.0, ymax])     
    # ax[0].imshow(frame_a, cmap=plt.cm.gray)
    # img = np.stack([frame_a, frame_b, 0*frame_a],axis=2)
    # ax[0].imshow(, cmap=plt.cm.jet)
    ax.quiver(x, y, u3, v3, color='r', lw=3)
    # ax[1].set_aspect('equal')

    # Display the figure with streamlit
    st.pyplot(fig)
else:
    st.text("Please upload two images for PIV analysis.")
