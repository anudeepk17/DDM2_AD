
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import acoustics
import numpy as np
import cv2
import argparse
import nibabel as nib
def add_noise(signal, snr):
    ''' 
    signal: np.ndarray
    snr: float
    returns -> np.ndarray
    '''
    # Generate the noise as you did
    noise = acoustics.generator.white(signal.size).reshape(*signal.shape)
    # For the record I think np.random.random does exactly the same thing
    # work out the current SNR
    current_snr = np.mean(signal) / np.std(noise)
    # scale the noise by the snr ratios (smaller noise <=> larger snr)
    noise *= (current_snr / snr)
    # return the new signal with noise
    return signal + noise

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE. The lower the error, the more "similar" the two images are.
    return mse_error



def compare(imageA, imageB):
    # Calculate the MSE and SSIM
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB,data_range=(max(int(imageA.max()),int(imageB.max()))-min(int(imageB.min()),int(imageA.min()))))
    # Return the SSIM. The higher the value, the more "similar" the two images are.
    return s

fmri_file = '/srv/home/kumar256/.dipy/sherbrooke_3shell/final.nii'
f_img = nib.load(fmri_file)
print(f_img.shape)
print(f_img.header.get_zooms())
print(f_img.header.get_xyzt_units())
# Check for same size and ratio and report accordingly

f_img_data = f_img.get_fdata()
f_img_data_noise=f_img_data
dim=f_img_data.shape
ct=0
mse_value=0
ssim_value=0
for i in range(dim[-1]):
   for j in range(dim[-2]):
        ct+=1
        f_img_data_noise[:,:,j,i]=add_noise(f_img_data[:,:,j,i],3)
        dimnoise= f_img_data_noise.shape
        ratio_orig = dim[0]/dim[1]
        ratio_comp = dimnoise[0]/dimnoise[1]
        if round(ratio_orig, 2) == round(ratio_comp, 2):
            mse_value += mse(f_img_data[:,:,j,i], f_img_data_noise[:,:,j,i])
            ssim_value += compare(f_img_data[:,:,j,i], f_img_data_noise[:,:,j,i])

        if ct%500==0:
           print("Average MSE for ",ct," count is ",mse_value/ct)
           print("Average SSIM for ",ct," count is ",ssim_value/ct)

noise_img = nib.Nifti1Image(f_img_data_noise, f_img.affine, f_img.header)
nib.save(noise_img, 'noise_final3.nii')
