from dipy.io.image import save_nifti, load_nifti
import acoustics
dataroot="/srv/home/kumar256/.dipy/sherbrooke_3shell/final.nii"
raw_data, _ = load_nifti(dataroot) # width, height, slices, gradients
print('Loaded data of size:', raw_data.shape)

signal=raw_data[:,:,0,0]
# Generate the noise as you did
noise = acoustics.generator.white(signal.size).reshape(*signal.shape)
for i in range
# work out the current SNR
current_snr = np.mean(signal) / np.std(noise)

# scale the noise by the snr ratios (smaller noise <=> larger snr)
noise *= (current_snr / snr)

# return the new signal with noise
signal + noise