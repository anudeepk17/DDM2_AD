import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
# from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
import numpy as np
import acoustics
import cv2
import nibabel as nib
import copy
from skimage.metrics import structural_similarity as ssim

def add_noise(signal, snr):
    ''' 
    signal: np.ndarray
    snr: float
    returns -> np.ndarray
    '''
    # Generate the noise as you did
    noise = acoustics.generator.white(signal.numpy().size).reshape(*signal.shape)
    # For the record I think np.random.random does exactly the same thing
    # work out the current SNR
    current_snr = np.mean(signal.numpy()) / np.std(noise)
    # scale the noise by the snr ratios (smaller noise <=> larger snr)
    noise *= (current_snr / snr)
    # return the new signal with noise
    return signal + torch.from_numpy(noise)

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE. The lower the error, the more "similar" the two images are.
    return mse_error


def compare_images(imageA, imageB, imageC,title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB,0.9)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(2, 2, 1)
    ax.title.set_text("Denoised Image")
    plt.imshow(imageA, cmap = plt.cm.gray,)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(2, 2, 2)
    ax.title.set_text("Clean Image without noise")
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    # show the third image
    ax = fig.add_subplot(2, 1, 2)
    ax.title.set_text("Clean Image with noise")
    plt.imshow(imageC, cmap = plt.cm.gray)
    plt.axis("off")
    fig.savefig(title+'comparison.png')
    # show the images
    plt.show()
    

def savecompare(visuals,n):
    orig = visuals['X']
    den = visuals['denoised']
    orig=orig[0,0,:,:]
    den=den[0,0,:,:]
    # computes the residuals
    rms_diff = np.sqrt((orig - den) ** 2)

    fig1, ax = plt.subplots(1, 3, figsize=(12, 6),
                            subplot_kw={'xticks': [], 'yticks': []})

    fig1.subplots_adjust(hspace=0.3, wspace=0.05)

    ax.flat[0].imshow(orig.T, cmap='gray', interpolation='none',
                    origin='lower')
    ax.flat[0].set_title('Original')
    ax.flat[1].imshow(den.T, cmap='gray', interpolation='none',
                    origin='lower')
    ax.flat[1].set_title('Denoised Output')
    ax.flat[2].imshow(rms_diff.T, cmap='gray', interpolation='none',
                    origin='lower')
    ax.flat[2].set_title('Residuals')

    fig1.savefig('{}compdenoised.png'.format(n))

    print("The result saved in noisy.png")

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='/srv/home/kumar256/DDM2/DDM2/config/stage1_test.json',
                    help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                    help='Run either train(training) or val(generation)', default='train')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')

# parse configs
args = parser.parse_args()
print(args)
opt = Logger.parse(args, stage=1)
# Convert to NoneDict, which return None for missing key.
opt = Logger.dict_to_nonedict(opt)
Logger.setup_logger(None, opt['path']['log'],
                    'train', level=logging.INFO, screen=True)
Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
logger = logging.getLogger('base')
logger.info('[Phase 1] Training noise model!')

for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
logger.info('Initial Dataset Finished')


trainer = Model.create_noise_model(opt)
logger.info('Initial Model Finished')
snr=10
i=0
for _,  val_data in enumerate(val_loader):
    i=i+1
    val_data_copy=copy.deepcopy(val_data)
    val_data['X'][0,0,:,:]=add_noise(val_data['X'][0,0,:,:],snr)
    val_data['condition'][0,0,:,:]=add_noise(val_data['condition'][0,0,:,:],snr)
    val_data['condition'][0,1,:,:]=add_noise(val_data['condition'][0,1,:,:],snr)
    trainer.feed_data(val_data)
    trainer.test(continous=True)

    vis = trainer.get_current_visuals()
    trainer.feed_data(val_data_copy)
    trainer.test(continous=True)
    vis2=trainer.get_current_visuals()
    denoised_img = Metrics.tensor2img(vis['denoised'])  # uint8
    input_img = Metrics.tensor2img(vis2['X'])  # uint8
    noisy_image=Metrics.tensor2img(vis['X'])
    t="SNR="+str(snr)
    compare_images(denoised_img[:,:,0],input_img[:,:,0],noisy_image,t)

    savecompare(vis,str(i)+"noise"+str(snr))
    savecompare(vis2,str(i)+"clean"+str(snr))

