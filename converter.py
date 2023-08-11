import dicom2nifti
import nibabel as nib
import os
mainpath='/Users/anudeep/Desktop/Desktop/DCEMRI/manifest-data1407430404196/QIN Breast DCE-MRI/QIN-Breast-DCE-MRI-BC01/04-24-1996-NA-Breast CE-59716'
npath='/Users/anudeep/Desktop/Desktop/DCEMRI/manifest-data1407430404196/QIN Breast DCE-MRI/QIN-Breast-DCE-MRI-BC01/04-24-1996-NA-Breast CE-59716/nifti'
tstamp=os.listdir(mainpath)
for tsf in tstamp:
    dicom2nifti.convert_directory(os.path.join(mainpath,tsf),os.path.join(mainpath,npath))

nifts=[]
for dirpath, dirnames, filenames in os.walk(npath):
    for file in filenames:
        nifts.append(os.path.join(dirpath,file))
final=nib.concat_images(nifts)
nib.save(final,os.path.join(npath,"final_4d.nii"))