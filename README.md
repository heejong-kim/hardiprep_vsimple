# HARDIprep (v.simple)
HARDIprep is automatic pipeline for diffusion weighted MR image quality control and correction. Original version follows [1]. HARDIprep (v.simple) is a simple version of HARDIprep which includes 7 steps:
1. Check the quality of DWIs using DTIPrep(missing slices, intensity artifacts, Venetian blind)
2. Fast bulk motion check within each DWI
3. Baseline (b0) image brain masking 
4. Image correction using SHORE resampling [2]
5. Check image quality after the correction using DTIPrep and exclude not curable gradient volumes 
6. Motion correction  
7. Brain masking 

HARDIprep was created by [Shireen Elhabian](https://github.com/sheryjoe) and modified by [Heejong Kim](https://github.com/heejong-kim).
This code is written for [IBIS study](https://www.ibis-network.org/recruitment-1.html) data preprocessing but it can be easily modified for different datasets.

## Requirements
### Python packages
```
conda install -c anaconda numpy scipy
conda install -c conda-forge nibabel dipy nipype 
conda install -c kayarre pynrrd  
``` 

### Softwares
* DTIPrep: https://www.nitrc.org/projects/dtiprep/
* FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
* ANTs: https://github.com/ANTsX/ANTs/releases

For b-table check
* DSI Studio (>2018 version): http://dsi-studio.labsolver.org/dsi-studio-download

Add softwares to $PATH before running HARDIprep
```
DTIPrepDIR="path to DTIPrep"
FSLDIR="path to FSL"
ANTsDIR="path to ANTs" 
DSIStudioDIR="path to DSI Studio" 
export PATH=$PATH:$DTIPrepDIR:$FSLDIR:$ANTsDIR:$DSIStudioDIR
```


## References
[1] Elhabian, Shireen, et al. "Subject–motion correction in HARDI acquisitions: choices and consequences." Frontiers in neurology 5 (2014): 240.

[2] Elhabian, Shireen, et al. "Compressive sensing based Q-space resampling for handling fast bulk motion in hardi acquisitions." 2016 IEEE 13th International Symposium on Biomedical Imaging (ISBI). IEEE, 2016.

