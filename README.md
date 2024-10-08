# wavelet_deconvolution

# Problem

Non-blind deconvolution of an image with a known PSF. Several types of regularization are available.

It optimizes the following loss:

        L = L_mse + R_grad + R_obj + R_wave + R_l1 + R_iuwt

        L_mse = 1/N sum (obs - convolved)^2 -> likelihood
        R_grad = lambda_grad * mean(grad^2) -> penalize large spatial gradients
        R_obj = lambda_obj * mean(image^2) -> penalize large pixel values
        R_wave = lambda_wavelet * wavelet_loss(image) -> penalize large wavelet coefficients
        R_l1 = lambda_l1 * mean(abs(image)) -> L1 amplitude regularization
        R_iuwt = lambda_iuwt * iuwt_loss(image) -> penalize large isotropic undecimated wavelet transform coefficients

It uses `PyTorch` and its automatic differentiation infrastructure to optimize the loss.

## Requirements

    numpy
    astropy
    ptwt
    torch
    matplotlib
    tqdm
    kornia

Install the packages following their instructions, but the following should do the trick:

    pip install numpy astropy ptwt matplotlib tqdm kornia
    pip install torch torchvision torchaudio
    
## References

- Starck, J.-L., and Murtagh, F., 1994, "Image restoration with noise suppression using the wavelet transform", A&A 288, 343–348.
- Starck, J.-L., and Murtagh, F., 2006, "Astronomical Image and Data Analysis", 2nd ed., Springer, New York
- Starck, J.-L., Murtagh, F., Fadili, J. M., 2006, "Sparse Image and Signal Processing: Wavelets, Curvelets, Morphological Diversity", Cambdridge
- https://academic.oup.com/mnras/article/426/2/1223/974193
