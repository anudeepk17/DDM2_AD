# DDM<sup>2</sup>: Self-Supervised Diffusion MRI Denoising with Generative Diffusion Models, ICLR 2023

Cited Paper: https://arxiv.org/pdf/2302.03018.pdf

![result](./assets/100000_1_denoised.png)

## Dependencies

Please clone our environment using the following command:

```
conda env create -f environment.yml  
conda activate ddm2
```

## Usage

### Data

We have used "Variations of dynamic contrast-enhanced magnetic resonance imaging in evaluation of breast cancer therapy response"
Link : https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=18514286
After the data is downloaded it needs to be converted into a .nii.gz compiled 4D file. This can be done by using 
```converter.py and mergernifti.py```
These just require path change to run.

### Configs

Different experiments are controlled by configuration files, which are in ```config/```. 

We have provided default training configurations for reproducing our experiments. Users are required to **change the path vairables** to their own directory/data before running any experiments. *More detailed guidances are provided as inline comments in the config files.*

### Noise Addition
Noise addition to every slice has been important and is done by 
```noiseadder.py```
The comments in the code are helpful as a guide to use the code

### Train

The training of DDM<sup>2</sup> contains three sequential stages. For each stage, a corresponding config file (or an update of the original config file) need to be passed as a coommand line arg.

1. To train our Stage I:  
```python3 train_noise_model.py -p train -c config/hardi_150.json```  
or alternatively, modify ```run_stage1.sh``` and run:  
```./run_stage1.sh```  

2. After Stage I training completed, the path to the checkpoint of the noise model need to be specific at 'resume_state' of the 'noise_model' section in corresponding config file. Additionally, a file path (.txt) needs to be specified at 'initial_stage_file' in the 'noise_model' section. This file will be recorded with the matched states in Stage II.  

3. To process our Stage II:  
```python3 match_state.py -p train -c config/hardi_150.json```  
or alternatively, modify ```run_stage2.sh``` and run:  
```./run_stage2.sh```  

4. After Stage II finished, the state file (a '.txt' file, generated in the previous step) needs to be specified at **'stage2_file'** variable in the last line of each config file. This step is neccesary for the following steps and inference.

5. To train our Stage III:  
```python3 train_diff_model.py -p train -c config/hardi_150.json```  
or alternatively, modify ```run_stage3.sh``` and run:  
```./run_stage3.sh```  

6. Validation results along with checkpoints will be saved in the ```/experiments``` folder.


### Inference (Denoise)

One can use the previously trained Stage III model to denoise a MRI dataset through:  
```python denoise.py -c config/hardi.json```  
or alternatively, modify ```denoise.sh``` and run:  
```./denoise.sh```   

The ```--save``` flag can be used to save the denoised reusults into a single '.nii.gz' file:  
```python denoise.py -c config/hardi.json --save```


### Quantitative Metrics Calulation

Stage 1 evaluations can be obtained for any input by 
```stage1Val.py```
We can provide any slice index in the laoded val array and send for denoising.


We have used Signal Intensity vs Time graph to study the temporal similarities. Along with that we have used MSE, SSIM and PSNR. The metrics can be obtained using denoisecomp.py. The code has comments to guide one to change voxel indices and slice index for various metrics calculation.
Functions in the code are :
```voxelIntensity(denoised,input,x,y,z)```
```compare_images(imageA, imageB,title)```
Run the code after necessary changes after loading the .nii.gz MRI data.

### Final files
The final model, logs and files can be obtained at this link:
https://drive.google.com/drive/folders/1bGrh2qb7Z641ohXhAbqNsN1NzYek3gGt?usp=drive_link
This also has the real and noise added dataset namely final.nii and noise_final.nii


## Citation  

We have used the following repo for our project
```
@inproceedings{xiangddm,
  title={DDM $\^{} 2$: Self-Supervised Diffusion MRI Denoising with Generative Diffusion Models},
  author={Xiang, Tiange and Yurt, Mahmut and Syed, Ali B and Setsompop, Kawin and Chaudhari, Akshay},
  booktitle={The Eleventh International Conference on Learning Representations}
}
```
