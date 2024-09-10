import numpy as np
from astropy.io import fits
from deconvolution import Deconvolution
import matplotlib.pyplot as pl
from astropy.visualization import (PercentileInterval, LogStretch,
                                   ImageNormalize)
from optimalSVHT import optimalSVHT
from matplotlib.patches import Rectangle


if (__name__ == '__main__'):  

    # Read the RAW
    f1 = fits.open('NGC0521_g_raw.fits')
    image = f1[1].data[None, None, :, :].astype(np.float32)
    image[image < -0.1] = 0.0

    f1 = fits.open('NGC0521_g_crop.fits')
    image_sub = f1[1].data[None, None, :, :].astype(np.float32)
    image_sub[image_sub < -0.1] = 0.0

    f2 = fits.open('PSF_crop.fits')
    psf = f2[1].data.astype(np.float32)

    f3 = fits.open('NGC0521_g_deconvolved_wiener.fits')
    wiener = f3[1].data.astype(np.float32)
    
    nb, ns, nx, ny = image.shape

    # Pad width
    pad_width = 24
    
    # General configuration
    config = {
        'gpu': 0,
        'npix_apodization': 24,
        'n_pixel': None,
        'n_iter' : 150,
        'n_iter_regularization': None,                
        'pad_width': pad_width,
        'precision': 'float32',
        'checkpointing': False,
        'iuwt_nbands': 5
    }           

    # Regularization parameters    
    lambda_grad = 0.000
    lambda_obj = 0.000
    lambda_wavelet = 0.0
    lambda_l1 = 0.00
    lambda_iuwt = 0.001

    # WORKING
    # lambda_grad = 0.0
    # lambda_obj = 0.05
    # lambda_wavelet = 0.000001
    # lambda_l1 = 0.00
    # lambda_iuwt = 0.01

    wavelet = 'db5'
    
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
                                                limit_zero=False)
    
    pct = 99.5

    std_rec_sky = np.std(rec[0, 0, 1240:1280, 420:490])
    std_image_sky = np.std(image[0, 0, 1240:1280, 420:490])
    std_image_sub_sky = np.std(image_sub[0, 0, 1240:1280, 420:490])
    std_wiener_sky = np.std(wiener[1240:1280, 420:490])

    _, _, std_rec, _ = optimalSVHT(rec[0, 0, ...])
    print(f'Reconstruction : noise_SVD={std_rec:.6f} - std_sky={std_rec_sky:.6f}')

    _, _, std_image, _ = optimalSVHT(image[0, 0, ...])
    print(f'Image : noise_SVD={std_image:.6f} - std_sky={std_image_sky:.6f}')

    _, _, std_image_sub, _ = optimalSVHT(image_sub[0, 0, ...])
    print(f'Image : noise_SVD={std_image_sub:.6f} - std_sky={std_image_sub_sky:.6f}')

    _, _, std_wiener, _ = optimalSVHT(wiener)
    print(f'Wiener : noise_SVD={std_wiener:.6f} - std_sky={std_wiener_sky:.6f}')

    fig, ax = pl.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].imshow(image[0, 0, :, :], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))    
    ax[0, 0].set_title(f'Original image - sky_std={std_image_sky:.6f}')
    ax[0, 0].add_patch(Rectangle((420, 1240), 50, 40, edgecolor='red', facecolor='none'))
    # ax[0, 0].axis('off')
    ax[0, 1].imshow(wiener[:, :], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))
    ax[0, 1].set_title(f'Wiener - sky_std={std_wiener_sky:.6f}')
    ax[0, 1].add_patch(Rectangle((420, 1240), 50, 40, edgecolor='red', facecolor='none'))
    # ax[0, 1].axis('off')

    ax[1, 0].imshow(image_sub[0, 0, :, :], norm=ImageNormalize(image_sub, interval=PercentileInterval(pct), stretch=LogStretch()))
    ax[1, 0].set_title(f'Subtracted image - sky_std={std_image_sub_sky:.6f}')
    ax[1, 0].add_patch(Rectangle((420, 1240), 50, 40, edgecolor='red', facecolor='none'))
    # ax[1, 0].axis('off')
    ax[1, 1].imshow(rec[0, 0, :, :], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))
    ax[1, 1].set_title(f'Deconvolved image - sky_std={std_rec_sky:.6f}')
    ax[1, 1].add_patch(Rectangle((420, 1240), 50, 40, edgecolor='red', facecolor='none'))
    # ax[1, 1].axis('off')
    pl.savefig('deconvolution.png')


    fig, ax = pl.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].imshow(image[0, 0, 400:800, 700:1100], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))
    ax[0, 0].set_title('Original image')
    # ax[0, 0].axis('off')
    ax[0, 1].imshow(wiener[400:800, 700:1100], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))
    ax[0, 1].set_title('wiener')
    # ax[0, 1].axis('off')

    ax[1, 0].imshow(image_sub[0, 0, 400:800, 700:1100], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))
    ax[1, 0].set_title('Subtracted image')
    # ax[1, 0].axis('off')
    ax[1, 1].imshow(rec[0, 0, 400:800, 700:1100], norm=ImageNormalize(image, interval=PercentileInterval(pct), stretch=LogStretch()))
    ax[1, 1].set_title('Deconvolved image')
    # ax[1, 1].axis('off')
    
    pl.savefig('deconvolution_detail.png')

    hdu = fits.PrimaryHDU(data=rec)
    hdu.writeto('reconstructed.fits')