import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.utils.data
import matplotlib.pyplot as pl
from collections import OrderedDict
from tqdm import tqdm
from kornia.filters import median_blur, spatial_gradient
import ptwt
import pywt
import iuwt_torch
try:
    from nvitop import Device
    NVITOP = True
except:
    NVITOP = False

class Deconvolution(nn.Module):
    def __init__(self, config):
        """

        Parameters
        ----------
        npix_apodization : int
            Total number of pixel for apodization (divisible by 2)
        device : str
            Device where to carry out the computations
        batch_size : int
            Batch size
        """
        super().__init__()

        self.config = config          
        
        # Define the device (cpu or gpu)
        self.cuda = torch.cuda.is_available()
        if self.config['gpu'] == -1:
            self.cuda = False
        self.device = torch.device(f"cuda:{self.config['gpu']}" if self.cuda else "cpu")        

        # If nvitop is installed, print the GPU information
        if (NVITOP):
            self.handle = Device.all()[self.config['gpu']]            
            print(f"Computing in {self.device} : {self.handle.name()} - mem: {self.handle.memory_used_human()}/{self.handle.memory_total_human()}")
        else:
            print(f"Computing in {self.device}")
        
        # Generate Hamming window function for apodization
        self.npix_apod = self.config['npix_apodization']
        win = np.hanning(self.npix_apod)
        winOut = np.ones(self.config['n_pixel'])
        winOut[0:self.npix_apod//2] = win[0:self.npix_apod//2]
        winOut[-self.npix_apod//2:] = win[-self.npix_apod//2:]
        window = np.outer(winOut, winOut)
        self.window = torch.tensor(window.astype('float32')).to(self.device)

        self.pad_width = self.config['pad_width']
        self.n_iter_regularization = self.config['n_iter_regularization']
                
        # Define axis in Fourier space for the mask
        print("Computing mask...")        
        x = np.linspace(-1, 1, self.config['n_pixel'])
        xx, yy = np.meshgrid(x, x)
        self.rho = np.sqrt(xx ** 2 + yy ** 2)
        
        # Which precision are we using?
        if self.config['precision'] == 'float16':
            print("Working in float16...")
            self.use_amp = True
        else:
            print("Working in float32...")
            self.use_amp = False        

        # Define the scaler for the automatic mixed precision
        self.scaler = torch.GradScaler("cuda", enabled=self.use_amp)
    
                     
    def lofdahl_scharmer_filter(self, image_ft, psf_ft):
        """
        Löfdahl & Scharmer (1994) Fourier filtering for deconvolution
        """
                
        num = torch.abs(psf_ft)**2
        denom = torch.abs(image_ft.detach() * torch.conj(psf_ft))**2
        H = 1.0 - self.mask * self.config['n_pixel']**2 * self.sigma**2 * (num / denom)
                
        H[H > 1.0] = 1.0
        H[H < 0.2] = 0.0

        H = self.mask * median_blur(H[None, None, :, :], (3, 3)).squeeze()
        H = torch.nan_to_num(H)

        H[H < 0.2] = 0.0
        
        return H

    def forward(self, image):
        """
        Evaluate the forward model
        """

        # Apodize frames and compute FFT        
        mean_val = torch.mean(image, dim=(2, 3), keepdim=True)
        image_apod = image - mean_val
        image_apod *= self.window
        image_apod += mean_val        
        image_ft = torch.fft.fft2(image_apod)

        # Compute appropriate Fourier mask
        H = 1.0
        if (self.regularize_fourier == 'scharmer'):
            H = self.lofdahl_scharmer_filter(image_ft, self.psf_ft)
        if (self.regularize_fourier == 'mask'):            
            H = self.mask[None, None, :, :]

        # Convolve estimated image with PSF and apply mask
        image_H_ft = H * image_ft
        convolved = torch.fft.ifft2(image_H_ft * self.psf_ft).real

        # If we are using spatial gradient regularization, compute the spatial gradients
        if (self.lambda_grad > 0.0):
            grad = spatial_gradient(image, mode='sobel', order=1)
        else:
            grad = None

        return convolved, image_H_ft, grad
    
    def wavelet_loss(self, image, wavelet='db5'):
        """
        Compute the wavelet loss
        """

        # Compute the discrete wavelet decomposition
        coefs = ptwt.wavedec2(image, pywt.Wavelet(wavelet), level=4, mode="reflect")
        
        # Now compute the loss by addding L1 norms of all coefficients
        nlev = len(coefs)
        loss = 0.0
        for i in range(nlev-1):
            for j in range(3):
                loss += torch.mean(torch.abs(coefs[i+1][j]))

        return loss
    
    def iuwt_loss(self, image, scale):
        """
        Compute the IUWT loss
        """

        # Compute the isotropic undecimated wavelet transform
        coefs = iuwt_torch.starlet_transform(image, num_bands = self.config['iuwt_nbands'], gen2 = False)
        
        # Now compute the loss by addding L1 norms of all coefficients scaled by their standard deviation
        nlev = len(coefs)
        loss = 0.0
        for i in range(nlev-1):            
            loss += scale[i] * torch.mean(torch.abs(coefs[i]))

        return loss
    
    # def threshold_wavelet(self, image, thr, wavelet='db5'):
    #     coefs = ptwt.wavedec2(image, pywt.Wavelet(wavelet), level=4, mode="reflect")
        
    #     nlev = len(coefs)
        
    #     for i in range(nlev-1):
    #         for j in range(3):
    #             coefs[i+1][j][torch.abs(coefs[i+1][j]) < thr] = 0.0

    #     image_out = ptwt.waverec2(coefs, pywt.Wavelet(wavelet))

    #     return image_out
    
    # def threshold_iuwt(self, image, ):
    #     coefs = ptwt.wavedec2(image, pywt.Wavelet(wavelet), level=4, mode="reflect")
        
    #     nlev = len(coefs)
        
    #     for i in range(nlev-1):
    #         for j in range(3):
    #             coefs[i+1][j][torch.abs(coefs[i+1][j]) < thr] = 0.0

    #     image_out = ptwt.waverec2(coefs, pywt.Wavelet(wavelet))

    #     return image_out
    
    # def hard_threshold(self, image, image_H, threshold):
    # # Final wavelet hard thresholding
    #     image = torch.tensor(image).to(self.device)
    #     image_H = torch.tensor(image_H).to(self.device)

    #     if (threshold > 0.0):
    #         with torch.no_grad():
    #             image_H = self.threshold_wavelet(image_H, threshold, wavelet=self.wavelet)
    #             image = self.threshold_wavelet(image, threshold, wavelet=self.wavelet)

    #     return image.cpu().numpy(), image_H.cpu().numpy()

    
    def deconvolve(self, 
                   frames, 
                   psf,
                   regularize_fourier='mask', 
                   diffraction_limit=0.95, 
                   lambda_grad=0.1, 
                   lambda_obj=0.0,                    
                   lambda_wavelet=0.0, 
                   lambda_l1=0.0,
                   lambda_iuwt=0.0,
                   wavelet='haar',
                   limit_zero=False,
                   avoid_saturated=False):
        """Deconvolve a set of images using a common PSF

        It optimizes the following loss:

        L = L_mse + R_grad + R_obj + R_wave + R_l1 + R_iuwt

        L_mse = 1/N sum (obs - convolved)^2 -> likelihood
        R_grad = lambda_grad * mean(grad^2) -> penalize large spatial gradients
        R_obj = lambda_obj * mean(image^2) -> penalize large pixel values
        R_wave = lambda_wavelet * wavelet_loss(image) -> penalize large wavelet coefficients
        R_l1 = lambda_l1 * mean(abs(image)) -> L1 amplitude regularization
        R_iuwt = lambda_iuwt * iuwt_loss(image) -> penalize large isotropic undecimated wavelet transform coefficients

        Args:
            obs (tensor): [B,C,H,W] tensor to be convolved: H and W are the image dimensions, B and C are batch and channel dimensions
            psf (tensor): [B,C,H,W] tensor with the PSF (with its peak centered in the middle of the image), B and C are batch and channel dimensions
            regularize_fourier (str, optional): Type of Fourier masking for denoising. Defaults to 'mask'.
                'mask' applies a simple diffraction mask above a cutoff
                'scharmer' applies a more elaborate masking based on Löfhdal & Scharmer (1994)
            diffraction_limit (float, optional): diffraction cutoff used in the Fourier masking in units of the diffraction limit. Defaults to 0.95.
            lambda_grad (float, optional): weight parameter for image gradient regularization. Defaults to 0.1.
            lambda_obj (float, optional): weight parameter for pixel amplitude regularization. Defaults to 0.0.
            lambda_wavelet (float, optional): weight parameter for wavelet regularization. Defaults to 0.0.
            lambda_l1 (float, optional): weight parameter for L1 pixel regularization. Defaults to 0.0.
            lambda_iuwt (float, optional): weight parameter for IUWT regularization. Defaults to 0.0.
            wavelet (str, optional): wavelet family to use in case lambda_wavelet is not zero. Defaults to 'haar'.
            limit_zero (bool, optional): Force a clamping of the image to zero from below. Defaults to False.
            avoid_saturated (bool, optional): _description_. Defaults to False.
        
        Returns:

            image: deconvolved image
            image_H: Fourier filtered image
            losses: loss history
            
        """
                    
        # Pad the images and move arrays to tensors        
        obs = np.pad(frames, pad_width=((0, 0), (0, 0), (self.pad_width // 2, self.pad_width // 2), (self.pad_width // 2, self.pad_width // 2)), mode='symmetric')        
        
        # Compute pixels with NaN values. They are considered saturated and won't be used in the computation of the likelihood
        if avoid_saturated:
            self.good_pixels = np.isnan(obs) == 0
            self.saturated_pixels = np.isnan(obs)
        else:
            self.good_pixels = np.ones_like(obs).astype('bool')
            self.saturated_pixels = np.zeros_like(obs).astype('bool')

        # Set saturated pixels to 1.0 for the initial guess
        if avoid_saturated:
            obs[self.saturated_pixels] = 1.0

        # Transform to PyTorch tensor and move to GPU
        obs = torch.tensor(obs.astype('float32')).to(self.device)            
        
        # Define the PSF. Pad it, shift it and normalize it, and compute its Fourier transform
        self.psf = psf 
        self.psf = np.pad(self.psf, pad_width=((self.pad_width // 2, self.pad_width // 2), (self.pad_width // 2, self.pad_width // 2)), mode='symmetric')
        self.psf = np.fft.fftshift(self.psf)
        self.psf = torch.tensor(self.psf.astype('float32')).to(self.device)
        self.psf = self.psf / torch.sum(self.psf)
        self.psf_ft = torch.fft.fft2(self.psf)
        
        # Fourier mask
        mask = self.rho <= diffraction_limit
        mask = np.fft.fftshift(mask)

        # Define specific wavelet family
        self.wavelet = wavelet
        self.mask = torch.tensor(mask.astype('float32')).to(self.device)

        self.lambda_grad = torch.tensor(lambda_grad).to(self.device)
        self.lambda_obj = torch.tensor(lambda_obj).to(self.device)        
        self.lambda_wavelet = torch.tensor(lambda_wavelet).to(self.device)
        self.lambda_l1 = torch.tensor(lambda_l1).to(self.device)
        self.lambda_iuwt = torch.tensor(lambda_iuwt).to(self.device)
        
        self.regularize_fourier = regularize_fourier

        # Compute noise characteristics if using IUWT
        if (self.lambda_iuwt > 0.0):
            noise = torch.randn_like(obs)
            out = iuwt_torch.starlet_transform(noise, num_bands = self.config['iuwt_nbands'], gen2 = False)

            std_iuwt = torch.zeros(len(out))
            for i in range(len(out)):
                std_iuwt[i] = torch.std(out[i])
            print(f'Noise std IUWT: {std_iuwt}')
        
        # Initial guess for the deconvolution
        image = obs.clone().detach().requires_grad_(True).to(self.device)

        # Define the optimizer
        optimizer = torch.optim.AdamW([image], lr=0.1)

        losses = []

        t = tqdm(range(self.config['n_iter']))        

        # Optimization loop        
        for loop in t:

            optimizer.zero_grad(set_to_none=True)

            # Forward pass using automatic mixed precision if float16 is used
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):

                # Clamp the image to zero if needed
                if limit_zero:
                    image_clamp = torch.clamp(image, min=0.0)
                else:
                    image_clamp = image
                
                # Get the convolved image using the current estimation and the PSF, either using checkpointing or not
                if self.config['checkpointing']:
                    convolved, image_H_ft, grad = torch.utils.checkpoint.checkpoint(self.forward, image_clamp, use_reentrant=False)
                else:                    
                    convolved, image_H_ft, grad = self.forward(image_clamp)

                # Regularization for the spatial gradient and the object
                if (self.lambda_grad > 0.0):
                    regul_grad = self.lambda_grad * torch.mean(grad**2)
                else:
                    regul_grad = torch.tensor(0.0).to(self.device)

                # Regularization for the object to force it to be zero
                if (self.lambda_obj > 0.0):
                    regul_obj = self.lambda_obj * torch.mean(image_clamp**2)
                else:
                    regul_obj = torch.tensor(0.0).to(self.device)

                # Regularization for the object to force it to be sparse
                if (self.lambda_l1 > 0.0):
                    regul_l1 = self.lambda_l1 * torch.mean(torch.abs(image_clamp))
                else:
                    regul_l1 = torch.tensor(0.0).to(self.device)

                # L1 Regularization for the discrete wavelet decomposition
                if (self.lambda_wavelet > 0.0):
                    regul_wavelet = self.lambda_wavelet * self.wavelet_loss(image_clamp, wavelet=self.wavelet)
                else:
                    regul_wavelet = torch.tensor(0.0).to(self.device)

                # L1 Regularization for the isotropic undecimated wavelet transform
                if (self.lambda_iuwt > 0.0):
                    regul_iuwt = self.lambda_iuwt * self.iuwt_loss(image_clamp, scale=std_iuwt)
                else:
                    regul_iuwt = torch.tensor(0.0).to(self.device)
                
                # Compute the likelihood only in the unsaturated pixels
                loss_mse = torch.mean( (obs[self.good_pixels] - convolved[self.good_pixels])**2)

                # Add likelihood and regularization
                loss = loss_mse + regul_grad + regul_obj + regul_wavelet + regul_l1 + regul_iuwt

                # Do we want to turn off the regularization at some point?
                if self.n_iter_regularization is not None:
                    if loop > self.n_iter_regularization:
                        self.lambda_grad = 0.0
                        self.lambda_obj = 0.0
                        self.lambda_wavelet = 0.0
                        self.lambda_l1 = 0.0
                        self.lambda_iuwt = 0.0

            # Backward pass using automatic differentiation
            self.scaler.scale(loss).backward()

            # Update the parameters
            self.scaler.step(optimizer)
            self.scaler.update()            
                        
            # Update the progress bar
            tmp = OrderedDict()
            if (NVITOP):
                tmp['gpu'] = f'{self.handle.gpu_utilization()}'                
                tmp['mem'] = f' {self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
            tmp['L_mse'] = f'{loss_mse.item():.8f}'
            tmp['R_grad'] = f'{regul_grad.item():.8f}'
            tmp['R_obj'] = f'{regul_obj.item():.8f}'
            tmp['R_l1'] = f'{regul_l1.item():.8f}'
            tmp['R_iuwt'] = f'{regul_iuwt.item():.8f}'
            tmp['R_wave'] = f'{regul_wavelet.item():.8f}'
            tmp['L'] = f'{loss.item():.8f}'
            t.set_postfix(ordered_dict=tmp)

            losses.append(loss.item())

        losses = np.array(losses)

        # Now we have the final image. Do the final processing

        # Clamp the image to zero if needed
        if limit_zero:
            image_clamp = torch.clamp(image, min=0.0)
        else:
            image_clamp = image

        # Final result after the optimization takes place
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):                
            # Get the convolved image using the current estimation and the PSF
            if self.config['checkpointing']:
                convolved, image_H_ft, grad = torch.utils.checkpoint.checkpoint(self.forward, image_clamp, use_reentrant=False)
            else:                    
                convolved, image_H_ft, grad = self.forward(image_clamp)

        # Recover filter image
        image_H = torch.fft.ifft2(image_H_ft).real
        
        # Return the unfiltered image    
        image_clamp = image_clamp.detach().cpu().numpy()
        image_H = image_H.detach().cpu().numpy()

        # Crop the padded region
        image_clamp = image_clamp[:, :, self.pad_width // 2:-self.pad_width // 2, self.pad_width // 2:-self.pad_width // 2]
        image_H = image_H[:, :, self.pad_width // 2:-self.pad_width // 2, self.pad_width // 2:-self.pad_width // 2]

        return image_clamp, image_H, losses