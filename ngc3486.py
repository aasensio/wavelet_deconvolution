import numpy as np
from astropy.io import fits
from deconvolution import Deconvolution
import matplotlib.pyplot as pl
from astropy.visualization import (PercentileInterval, LogStretch,
                                   ImageNormalize)
from optimalSVHT import optimalSVHT
from matplotlib.patches import Rectangle


if (__name__ == '__main__'):

    which = 'LBT'

    if which == 'LBT':
        
        # Read the RAW image and crop the central region  
        size = 3702
        f1 = fits.open('ngc3486/NGC3486_Sloan-g-LBT.fits')
        nx, ny = f1[1].data.shape
        centerx = nx//2
        centery = ny//2
        image = f1[1].data[None, None, centerx-size//2:centerx+size//2, centery-size//2:centery+size//2].astype(np.float32)
        
        # Patch the very negative values. They seem to be artifacts
        image[image < -0.1] = 0.0
        
        # Read the PSF and crop the central region
        f2 = fits.open('ngc3486/model-g-band-new.fits')
        nx, ny = f2[1].data.shape
        centerx = nx//2
        centery = ny//2
        psf = f2[1].data[centerx-size//2:centerx+size//2, centery-size//2:centery+size//2].astype(np.float32)

        # Regularization parameters
        lambda_grad = 0.000
        lambda_obj = 0.000
        lambda_wavelet = 0.0
        lambda_l1 = 0.00
        lambda_iuwt = 0.0005#1
        wavelet = None

    if which == 'WHT':
        
        # Read the RAW image and crop the central region  
        size = 2702
        f1 = fits.open('ngc3486/NGC3486_Sloan-g-WHT.fits')
        nx, ny = f1[1].data.shape
        centerx = nx//2
        centery = ny//2
        image = f1[1].data[None, None, centerx-size//2:centerx+size//2, centery-size//2:centery+size//2].astype(np.float32)

        # Patch the very negative values. They seem to be artifacts
        image[image < -0.1] = 0.0
        
        # Read the PSF and crop the central region
        f2 = fits.open('ngc3486/model-g-band-new.fits')
        nx, ny = f2[1].data.shape
        centerx = nx//2
        centery = ny//2
        psf = f2[1].data[centerx-size//2:centerx+size//2, centery-size//2:centery+size//2].astype(np.float32)

        # Regularization parameters    
        lambda_grad = 0.000
        lambda_obj = 0.000
        lambda_wavelet = 0.0
        lambda_l1 = 0.00
        lambda_iuwt = 0.0005#1
        wavelet = None
        
    # Get dimensions (the first two are the batch and channel dimensions, not used in this case)
    # The algorithm can handle multiple images at once
    nb, nc, nx, ny = image.shape
    pct = 99.5

    # Pad width for apodization
    pad_width = 24
    
    # General configuration
    config = {
        'gpu': 0,
        'npix_apodization': 24,
        'n_pixel': None,
        'n_iter' : 100,
        'n_iter_regularization': None,                
        'pad_width': pad_width,
        'precision': 'float32',
        'checkpointing': False,
        'iuwt_nbands': 5
    }           

    
    config['n_pixel'] = nx + pad_width

    # Instantiate the model
    deconvolver = Deconvolution(config)
        
    # Deconvolve the data
    # It returns the deconvolved image, the Fourier filtered image and the loss
    rec, rec_H, loss = deconvolver.deconvolve(image,                                                 
                                            psf,
                                                regularize_fourier=None,#'mask', 
                                                diffraction_limit=0.65,
                                                lambda_grad=lambda_grad, 
                                                lambda_obj=lambda_obj,
                                                lambda_wavelet=lambda_wavelet,
                                                lambda_l1=lambda_l1,
                                                lambda_iuwt=lambda_iuwt,
                                                wavelet=wavelet,
                                                limit_zero=False,
                                                avoid_saturated=True)

    # Some plots
    if which == 'LBT':
        noise_win = [100, 500, 100, 500]
        zoom_win = [1500, 2000, 2800, 3300]
        
        std_rec_sky = np.std(rec[0, 0, noise_win[0]:noise_win[1], noise_win[2]:noise_win[3]])
        std_image_sky = np.std(image[0, 0, noise_win[0]:noise_win[1], noise_win[2]:noise_win[3]])
        
        _, _, std_rec, _ = optimalSVHT(rec[0, 0, noise_win[0]:noise_win[1], noise_win[2]:noise_win[3]])
        print(f'Reconstruction : noise_SVD={std_rec:.6f} - std_sky={std_rec_sky:.6f}')

        _, _, std_image, _ = optimalSVHT(image[0, 0, noise_win[0]:noise_win[1], noise_win[2]:noise_win[3]])
        print(f'Image : noise_SVD={std_image:.6f} - std_sky={std_image_sky:.6f}')
        

        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(15, 15))
        ax[0, 0].imshow(image[0, 0, :, :], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))    
        ax[0, 0].set_title(fr'Original image - $\sigma$={std_image_sky:.5f} - $n$={std_image:.5f}')

        ax[0, 1].imshow(rec[0, 0, :, :], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))
        ax[0, 1].set_title(f'Deconvolved image - $\sigma$={std_rec_sky:.5f} - $n$={std_rec:.5f}')
        ax[0, 1].add_patch(Rectangle((noise_win[0], noise_win[2]), noise_win[1]-noise_win[0], noise_win[3]-noise_win[2], edgecolor='red', facecolor='none'))

        ax[1, 0].imshow(image[0, 0, zoom_win[0]:zoom_win[1], zoom_win[2]:zoom_win[3]], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))            
        ax[1, 1].imshow(rec[0, 0, zoom_win[0]:zoom_win[1], zoom_win[2]:zoom_win[3]], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))
        
        hdu = fits.PrimaryHDU(data=rec[0, 0, :, :])
        hdu.writeto('reconstructed_LBT.fits', overwrite=True)

    if which == 'WHT':
        noise_win = [100, 500, 100, 500]
        zoom_win = [1500, 2000, 1800, 2300]
        
        std_rec_sky = np.std(rec[0, 0, noise_win[0]:noise_win[1], noise_win[2]:noise_win[3]])
        std_image_sky = np.std(image[0, 0, noise_win[0]:noise_win[1], noise_win[2]:noise_win[3]])
        
        _, _, std_rec, _ = optimalSVHT(rec[0, 0, noise_win[0]:noise_win[1], noise_win[2]:noise_win[3]])
        print(f'Reconstruction : noise_SVD={std_rec:.6f} - std_sky={std_rec_sky:.6f}')

        _, _, std_image, _ = optimalSVHT(image[0, 0, noise_win[0]:noise_win[1], noise_win[2]:noise_win[3]])
        print(f'Image : noise_SVD={std_image:.6f} - std_sky={std_image_sky:.6f}')
        

        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(15, 15))
        ax[0, 0].imshow(image[0, 0, :, :], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))    
        ax[0, 0].set_title(fr'Original image - $\sigma$={std_image_sky:.5f} - $n$={std_image:.5f}')
        ax[0, 0].add_patch(Rectangle((2800, 1950), 3000-2800, 2200-1950, edgecolor='red', facecolor='none'))

        ax[0, 1].imshow(rec[0, 0, :, :], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))
        ax[0, 1].set_title(f'Deconvolved image - $\sigma$={std_rec_sky:.5f} - $n$={std_rec:.5f}')
        ax[0, 1].add_patch(Rectangle((noise_win[0], noise_win[2]), noise_win[1]-noise_win[0], noise_win[3]-noise_win[2], edgecolor='red', facecolor='none'))

        ax[1, 0].imshow(image[0, 0, zoom_win[0]:zoom_win[1], zoom_win[2]:zoom_win[3]], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))    
        ax[1, 1].imshow(rec[0, 0, zoom_win[0]:zoom_win[1], zoom_win[2]:zoom_win[3]], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))
        
        hdu = fits.PrimaryHDU(data=rec[0, 0, :, :])
        hdu.writeto('reconstructed_WHT.fits', overwrite=True)

