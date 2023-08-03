import numpy as np
import nibabel as nib
import core.metrics as Metrics
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def voxelIntensity(denoised,input,x,y,z):
    denoisedSlice=denoised[x,y,z,:]
    inputSlice=input[x,y,z,:]
    denoisedSlice=denoisedSlice.squeeze()
    inputSlice=inputSlice.squeeze()
    fig1, ax1 = plt.subplots()

    # Display the image
    ax1.imshow(input[:,:,z,-1])
    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax1.add_patch(rect)
    fig1.savefig('comparison.png')
    #Comparison signal v time
    time_point=np.arange(denoised_data.shape[3])
    fig = plt.figure(figsize=(10,5))
    plt.plot(time_point, denoisedSlice)
    plt.title('Denoised Time-Activity Curve for Voxel ({},{},{})'.format(x, y, z))
    plt.xlabel('Time')
    plt.ylabel('Activity')
    plt.grid()
    fig.savefig('DenoisedIntensity.png')
    fig = plt.figure(figsize=(10,5))
    plt.plot(time_point, inputSlice)
    plt.title('Real Time-Activity Curve for Voxel ({},{},{})'.format(x, y, z))
    plt.xlabel('Time')
    plt.ylabel('Activity')
    plt.grid()
    fig.savefig('RealIntensity.png')
    


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images

    mse_error = np.sum((imageA.astype("float32") - imageB.astype("float32")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE. The lower the error, the more "similar" the two images are.
    return mse_error

def compare_images(imageA, imageB,title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB,data_range=(max((imageA.max()),(imageB.max()))-min((imageB.min()),(imageA.min()))))
    psnr=peak_signal_noise_ratio(imageA,imageB)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.5f, SSIM: %.2f,PSNR: %.2f" % (m, s,psnr))
    # show first image
    ax = fig.add_subplot(2, 1, 1)
    ax.title.set_text("Denoised Image")
    plt.imshow(imageA)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(2, 1, 2)
    ax.title.set_text("Clean Image without noise")
    plt.imshow(imageB)
    plt.axis("off")
    fig.savefig(title+'comparison.png')
    # show the images
    plt.show()


real = nib.load('/srv/home/kumar256/.dipy/sherbrooke_3shell/final.nii')
denoised = nib.load('/srv/home/kumar256/DDM2/DDM2/experiments/s3sh_denoise_230726_190719/results/s3sh_denoised.nii.gz')
data=real.get_fdata()
denoised_data=denoised.get_fdata()
denoised_data = denoised_data.astype(np.float64) / np.max(denoised_data.astype(np.float64), axis=(0,1,2), keepdims=True)
data = data.astype(np.float64) / np.max(data.astype(np.float64), axis=(0,1,2), keepdims=True)
denoised_img = (denoised_data[:,:,40,2])  # uint8
input_img = (data[:,:,40,2])  # uint8
voxelIntensity(denoised_data,data,240,250,40)
compare_images(denoised_img,input_img,'InitMetrics')

