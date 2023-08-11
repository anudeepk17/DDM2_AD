import nibabel as nib
import os
npath='/Users/anudeep/Desktop/Desktop/DCEMRI/manifest-data1407430404196/QIN Breast DCE-MRI/QIN-Breast-DCE-MRI-BC01/04-24-1996-NA-Breast CE-59716/nifti'

nifts=[]
for dirpath, dirnames, filenames in os.walk(npath):
    for file in filenames:
        nifts.append(os.path.join(dirpath,file))
final=nib.concat_images(nifts)
nib.save(final,"final.nii")
# import nibabel as nib
# import numpy as np

# # List of file paths to individual 3D .nii files
# file_paths = ["file1.nii", "file2.nii", "file3.nii"]

# # Create an empty list to store the loaded image objects
# image_list = []

# # Load each individual 3D .nii file and append the image object to the list
# for file_path in file_paths:
#     image = nib.load(file_path)
#     image_list.append(image)

# # Convert the list of image objects to a stack
# stacked_images = np.stack([image.get_fdata() for image in image_list], axis=-1)

# # Create a new nibabel image object with the stacked data
# stacked_image_obj = nib.Nifti1Image(stacked_images, affine=image_list[0].affine)

# # Save the stacked image object to a new .nii file
# nib.save(stacked_image_obj, "stacked_image.nii")
