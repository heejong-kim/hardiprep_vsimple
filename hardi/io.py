# -*- coding: utf-8 -*-
"""

Created on Sat Jan 10 10:38:34 2015
@author: Shireen Elhabian

Modified as hardi_vsimple Dec 21 2020
@author: Heejong Kim

objective:
    utitlity functions for io to be used in the processing pipeline

"""

import numpy as np
import csv
import os
import scipy.io as sio
import copy
import nrrd
import nibabel as nib
from dipy.core.gradients import gradient_table
from collections import OrderedDict
from dipy.io.gradients import read_bvals_bvecs

dtiprepbin = '/home/users/hk2451/Utils/DTIPrep-1.2.11/bin/' # server
dtiprepbin = '/home/heejong/HDD2T/utils/DTIPrep-1.2.11/bin/' # local

"""
Given the nrrdfilename, this function reads nrrd data plus 
the gradient directions, indicating which direction correspond to b-value =0, 
i.e. baseline, if any...
"""
def readHARDI(nrrdfilename):
    
    nrrdData, options = nrrd.read(nrrdfilename)
    
    if options['kinds'][0] == 'list': # as saved by slicer
        nrrdData = np.transpose(nrrdData, (1,2,3,0))
        options['kinds'] = ['space','space','space','list']
        
    if options['kinds'][0] == 'vector': # as saved by dtiprep, first dimension is directions
        nrrdData = np.transpose(nrrdData, (1,2,3,0))
        options['kinds'] = ['space','space','space','vector']

    """
    Getting the gradient directions
    """
    bvalue        = float(options['DWMRI_b-value'])
    
    # now lets go over each direction and get its value
    gradientDirections = list() #np.zeros(shape=(nDirs,3), dtype = 'float')
    baselineIndex = -1
    for d in range(0,max(nrrdData.shape)): # last dimension is directions
        keyname = 'DWMRI_gradient_' + str(d).zfill(4)
        #print keyname
        if keyname in options.keys():
            keyval  = options[keyname]
            keyval  = keyval.split()    
            gradDir = np.zeros((1,3))
            gradDir[0,0] = np.float(keyval[0])
            gradDir[0,1] = np.float(keyval[1])
            gradDir[0,2] = np.float(keyval[2])    
            if (np.sum(gradDir) == 0) or (np.linalg.norm(gradDir) == 0):
                baselineIndex = d
            else:
                gradDir = gradDir/np.linalg.norm(gradDir) # normalize the gradient direction
            gradientDirections.append(gradDir)
        else:
            break
        
    gradientDirections = np.array(gradientDirections)
    
    return nrrdData, bvalue, gradientDirections, baselineIndex, options

def readDataset(niifilename, niiBrainMaskFilename, btablefilename,  parcellationfilename = None):  
     
    # load the masked diffusion dataset
    diffusionData = nib.load(niifilename).get_data()
    affine        = nib.load(niifilename).get_affine()
    
    # load the brain mask
    mask    = nib.load(niiBrainMaskFilename).get_data()
    
    rows, cols, nSlices, nDirections = diffusionData.shape
    
    bvals, bvecs = readbtable(btablefilename)
    gtable       = gradient_table(bvals, bvecs)
    
    if parcellationfilename != None:
        #parcellation = nib.load(parcellationfilename).get_data()
        parcellation,_ = nrrd.read(parcellationfilename)
    
        if parcellation.shape[2] != nSlices:  # for the second phantom (unc_res)
            parcellation = parcellation[:,:,parcellation.shape[2]-nSlices:]        
        parcellation = np.squeeze(parcellation)
    else:
        parcellation = None
    
    return diffusionData, mask, affine, gtable, parcellation
    

def convertToNIFTI(nrrdfilename, niifilename, bvecsfilename, bvalsfilename):
    cmdStr = dtiprepbin + 'DWIConvert --inputVolume %s --conversionMode NrrdToFSL --outputVolume %s --outputBVectors %s --outputBValues %s' % (nrrdfilename, niifilename, bvecsfilename, bvalsfilename)
    os.system(cmdStr)
    
def save2nii(niifilename, data, affine=None):

    # Lines below (from the if part 110-115 is modified)
    if len(affine) == 0 :
        niiData = nib.Nifti1Image(data, np.eye(4))
        # print(np.eye(4))
    else:
        niiData = nib.Nifti1Image(data, affine)
        # print(affine)

    # ####Original lines
    # if affine == None:
    #     niiData = nib.Nifti1Image(data, np.eye(4))
    # else:
    #     niiData = nib.Nifti1Image(data, affine)
    
    # print(niiData.get_header()['dim'])
    # print(niiData.get_data_dtype())
    
    nib.save(niiData, niifilename)



def save_bval_bvec(bval, bvec, savename):
    savebval = savename+'.bvals'
    savebvec = savename + '.bvecs'
    with open(savebval, 'w') as file:
        for b in bval:
            file.write(str(int(b)) + '\n')
    #
    with open(savebvec, 'w') as file:
        for l in range(bvec.shape[0]):
            for b in bvec[l, :]:
                file.write(str((b)) + ' ')
            file.write('\n')

    # with open(savebvec, 'w') as file:
    #     for b in bvec[:, 0]:
    #         file.write(str((b)) + ' ')
    #     #
    #     file.write('\n')
    #     for b in bvec[:, 1]:
    #         file.write(str((b)) + ' ')
    #     #
    #     file.write('\n')
    #     for b in bvec[:, 2]:
    #         file.write(str((b)) + ' ')
    #     #
    #     file.write('\n')

def readbvalsbvecs(bvalsfilename, bvecsfilename):
    
    bvals = list()
    fid = open(bvalsfilename, 'r')
    for line in fid:
        bvals.append(float(line.strip()))
    fid.close()
    
    nDirections = len(bvals)
    bvals       = np.array(bvals)
    
    bvecs = np.zeros((nDirections, 3))
    fid = open(bvecsfilename, 'r')
    ii = 0
    for line in fid:
        line = line.strip()
        line = line.split()
        bvecs[ii,0] = float(line[0])
        bvecs[ii,1] = float(line[1])
        bvecs[ii,2] = float(line[2])
        ii = ii + 1
    fid.close()
    
    return bvals, bvecs
    
def bvecsbvals2btable(bvalsfilename, bvecsfilename, btablefilename, isbvaluelast=False):
    bvals, bvecs = readbvalsbvecs(bvalsfilename, bvecsfilename)
    writebtable(btablefilename, bvals, bvecs,isbvaluelast)

def bvecsbvals2btable_dipy(bvalsfilename, bvecsfilename, btablefilename, isbvaluelast=False):
    bvals, bvecs = read_bvals_bvecs(bvalsfilename, bvecsfilename)
    writebtable(btablefilename, bvals, bvecs,isbvaluelast)


def readbtable(btablefilename):
    
    bvals = list()
    fid = open(btablefilename, 'r')
    for line in fid:
        line = line.strip()
        line = line.split()
        bvals.append(float(line[0]))
    fid.close()
    
    nDirections = len(bvals)
    bvals       = np.array(bvals)
    
    bvecs = np.zeros((nDirections, 3))
    fid = open(btablefilename, 'r')
    ii = 0
    for line in fid:
        line = line.strip()
        line = line.split()
        bvecs[ii,0] = float(line[1])
        bvecs[ii,1] = float(line[2])
        bvecs[ii,2] = float(line[3])
        ii = ii + 1
    fid.close()
    
    return bvals, bvecs
    
def writebtable(btablefilename, bvals, bvecs, isbvaluelast=False):
    
    nDirections = bvecs.shape[0]
    
    fid = open(btablefilename, 'w')
    for ii in range(nDirections):
        if isbvaluelast:
            fid.write('%f\t%f\t%f\t%f\n' % (bvecs[ii,0], bvecs[ii,1], bvecs[ii,2], bvals[ii]))
        else:
            fid.write('%f\t%f\t%f\t%f\n' % (bvals[ii], bvecs[ii,0], bvecs[ii,1], bvecs[ii,2]))
    fid.close()
    
def nifti2src(niifilename, btablefilename, srcfilename): 
    cmdStr = 'dsi_studio --action=src --source=%s --b_table=%s --output=%s' % (niifilename, btablefilename, srcfilename)
    os.system(cmdStr)


def fixNRRDfile(nrrdfilename, encoding='gzip'):
    # data not saved by slicer ITK (from Clement) would have NAN in the thickness
    # note: slicer save number of directions as the first dimension
    
    # save to nrrd
    nrrdData, options = nrrd.read(nrrdfilename)
        
    if options['kinds'][0] == 'list' or options['kinds'][0] == 'vector': # as saved by slicer
        nrrdData = np.transpose(nrrdData, (1,2,3,0))
        options['kinds'] = ['space','space','space','list']
        options['centerings'] = ['cell','cell','cell','???']
        #
    if type(options['space directions'][0]) is str or np.isnan(options['space directions'][0][0]):
        options['space directions'] = [options['space directions'][1], options['space directions'][2], options['space directions'][3], None]
    else:
        options['space directions'] = [options['space directions'][0], options['space directions'][1], options['space directions'][2], None]

    options['thicknesses'] = [abs(options['space directions'][0][0]), abs(options['space directions'][1][1]), abs(options['space directions'][2][2]), np.nan]
    options['sizes']       = list(nrrdData.shape)
    options['encoding'] = encoding
    options['type'] = 'int16'
    nrrd.write( nrrdfilename, nrrdData.astype('int16'), options)
    return options
    
def updateNrrdOptions(options, new_bvecs):

    newoptions = options
    newoptions['sizes'][-1] = new_bvecs.shape[0]

    nDirections = new_bvecs.shape[0]    
    for d in range(nDirections):
        keyname = 'DWMRI_gradient_' + str(d).zfill(4)
        
        gradDir = np.zeros((1,3))
        gradDir[0,0] = new_bvecs[d,0]
        gradDir[0,1] = new_bvecs[d,1]
        gradDir[0,2] = new_bvecs[d,2]
        if np.linalg.norm(gradDir) > 0:
            gradDir = gradDir/np.linalg.norm(gradDir) # normalize the gradient direction
                
        newoptions[keyname] = '%f %f %f' % (gradDir[0,0], gradDir[0,1], gradDir[0,2])

    for d in range(nDirections, options['sizes'][-1]):
        keyname = 'DWMRI_gradient_' + str(d).zfill(4)
        print(keyname)
        del newoptions[keyname]
    
    return newoptions
    
def removeBaselineFromOptions(options, baselineIndex):
    new_options = copy.deepcopy(options)
    new_options['sizes'][-1] = new_options['sizes'][-1] -1
    new_options.pop('keyvaluepairs')
    new_options['keyvaluepairs'] = dict()

    ind = -1;
    for key in sorted(options['keyvaluepairs'].keys()):
        if key.find('DWMRI_gradient') >= 0:
            dd = int(key.split('_')[-1])
            if dd == baselineIndex:
                continue
            ind = ind + 1
            new_options['keyvaluepairs']['DWMRI_gradient_%s' % (str(ind).zfill(4))] = options['keyvaluepairs'][key]
        else:
            new_options['keyvaluepairs'][key] = options['keyvaluepairs'][key]
    
    return new_options
    
    
def extractAndSaveBaselineToNRRD(nrrdfilename, baselinenrrdfilename):
    
    nrrdData, bvalue, gradientDirections, baselineIndex,options = readHARDI(nrrdfilename)
                
    baseline = nrrdData[:,:,:,baselineIndex]

    # save to nrrd
    newoptions = OrderedDict()
    newoptions['type'] = options['type']
    newoptions['dimension'] = 3
    newoptions['space'] = options['space']
    newoptions['sizes'] = list(baseline.shape)
    newoptions['space directions'] = [options['space directions'][0], options['space directions'][1], options['space directions'][2]]
    newoptions['kinds'] = ['space', 'space', 'space']
    newoptions['endian'] = options['endian']
    newoptions['encoding'] = options['encoding']
    newoptions['thicknesses'] = [abs(options['space directions'][0][0]), abs(options['space directions'][1][1]),
                                 abs(options['space directions'][2][2])]
    newoptions['centerings'] = [options['centerings'][0], options['centerings'][1], options['centerings'][2]]
    newoptions['space origin'] = options['space origin']
    newoptions['measurement frame'] = options['measurement frame']
    nrrd.write( baselinenrrdfilename, baseline, newoptions)
    
def saveMatrixToCSV(A,filename, delim=', '):
    fid = open(filename, 'w')
    for row in A:
        fid.write(delim.join(map(str,row)) + '\n')
    fid.close()
    
def saveToFIB(fibfilename, dimension, voxel_size, csd_peaks, sphere, baseline):
    
    rows, cols, nSlices = dimension
    maxFibers           = csd_peaks.peak_indices.shape[-1]
    
    outData = dict()
    outData['dimension']  = np.array([rows, cols, nSlices]).reshape((1,3))
    outData['voxel_size'] = voxel_size.reshape((1,3))
    outData['gfa']        = np.ravel(csd_peaks.gfa.T)
    outData['baseline']   = np.ravel(baseline.T)
    
    for f in range(maxFibers):
        fa    = csd_peaks.qa[:,:,:,f]
        index = csd_peaks.peak_indices[:,:,:,f]
        
        outData['fa%d'%(f)]    = np.ravel(fa.T)
        outData['index%d'%(f)] = np.ravel(index.T)
        
    outData['odf_vertices'] = sphere.vertices.T
    outData['odf_faces']    = sphere.faces.T
    
    N          = 0
    for kk in range(nSlices):
        for jj in range(cols):
            for ii in range(rows):
                if csd_peaks.qa[ii,jj,kk,0] > 0.0:
                    N = N + 1
                    
    nSamples   = sphere.vertices.shape[0]
    odf        = np.zeros((nSamples/2, N ))
    ind        = -1                        
    for kk in range(nSlices):
        for jj in range(cols):
            for ii in range(rows):
                if csd_peaks.qa[ii,jj,kk,0] > 0.0:  
                    ind = ind+ 1
                    curODF = csd_peaks.odf[ii,jj,kk,0:nSamples/2] 
                    odf[:,ind] = curODF
    outData['odfs']    = odf 
    
    sio.savemat(fibfilename, mdict=outData, format='4')
    os.system(('mv %s.mat %s' %(fibfilename,fibfilename)))

def get_vox_dims(volume):
    import nibabel as nb
    if isinstance(volume, list):
        volume = volume[0]
    nii = nb.load(volume)
    hdr = nii.get_header()
    voxdims = hdr.get_zooms()
    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]

def get_data_dims(volume):
    import nibabel as nb
    if isinstance(volume, list):
        volume = volume[0]
    nii = nb.load(volume)
    hdr = nii.get_header()
    datadims = hdr.get_data_shape()
    return [int(datadims[0]), int(datadims[1]), int(datadims[2])]

def get_affine(volume):
    import nibabel as nb
    nii = nb.load(volume)
    return nii.get_affine()

    
def read_atlas_labels(labelsfilename):
    atlas_labels = dict()            
    fid = open(labelsfilename, 'r')
    for line in fid:
        line = line.strip()
        line = line.split()
        atlas_labels[line[1]] = int(line[0])
    fid.close()
    
    return atlas_labels
    
def read_atlas_labels_abbrevs (abbrevfilename):
    
    #----- get the abbreviations
    abbrevs = dict()
    fid = open(abbrevfilename, 'r')
    for line in fid:
        line = line.strip().split()
        key = line[0]
        label = line[1].lower()
        abbrevs[key] = label
    fid.close()
    
    return abbrevs 
    
def write_dict(mydict, csvfilename):
    writer = csv.writer(open(csvfilename, 'wb'))
    for key in sorted(mydict.keys()):
        value = mydict[key]
        writer.writerow([key, value])
    #for key, value in mydict.items():
    #   writer.writerow([key, value])

def read_dict(csvfilename):
    reader = csv.reader(open(csvfilename, 'rb'))
    mydict = dict(x for x in reader)
    return mydict    
    
def write_list(filename, vals):
    fid = open(filename, 'w')
    for val in vals:
        fid.write('%f\n'%(val) )
    fid.close()
    
def read_list(filename):
    vals = list()
    fid  = open(filename, 'r')
    for line in fid:
        vals.append(float(line.strip()))
    fid.close()
    return vals;
    
