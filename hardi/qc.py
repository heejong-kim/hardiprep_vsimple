# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 10:00:22 2015

@author: Shireen Elhabian
"""
from __future__ import division
import os
import glob
import numpy as np
import ntpath
import time
import shutil
import copy
import csv
import math
import sys

import nibabel as nib
import hardi.io as hardiIO
import nrrd
nrrd.reader.ALLOW_DUPLICATE_FIELD = True
import hardi.qc_utils as hardiQCUtils

import dipy.reconst.shm as drecon
from dipy.core.gradients import gradient_table
from dipy.reconst.shore import ShoreModel, shore_matrix
from dipy.reconst.shm import sh_to_sf
import dipy.reconst.dti as dti
from dipy.io.gradients import read_bvals_bvecs
# To deal with os.system does not wait for the process
import subprocess



dtiprepbin = '/home/users/hk2451/Utils/DTIPrep-1.2.11/bin/' # server
dtiprepbin = '/home/heejong/HDD2T/utils/DTIPrep-1.2.11/bin/' # local


def ParseFilename(nrrdfilename):
    _, filename = ntpath.split(nrrdfilename)
    # basename     = filename[:-len(ext)]
    basename = filename.split('.')[0]
    index = basename.find('_DWI_65dir')
    if index >= 0:
        phan_name = basename[:index]
    else:
        index = basename.find('_DSI')
        if index >= 0:
            phan_name = basename[:index]
        else:
            phan_name = basename

    return basename, phan_name


def GetPrepDir(nrrdfilename, outDir, prepDir_suffix=None):
    basename, phan_name = ParseFilename(nrrdfilename)

    if prepDir_suffix is None:
        prepDir = os.path.join(outDir, 'HARDIprep_QC')
    else:
        prepDir = os.path.join(outDir, 'HARDIprep_QC_%s' % (prepDir_suffix))

    return prepDir


def PrepareQCsession(origNrrdfilename, outDir, phan_name, prepDir_suffix=None, nDirections=65,
                     check_btable=True):
    start_time = time.time()

    # basename, phan_name = ParseFilename(origNrrdfilename)


    if prepDir_suffix is None:
        prepDir = os.path.join(outDir, 'HARDIprep_QC')
    else:
        prepDir = os.path.join(outDir, 'HARDIprep_QC_%s' % (prepDir_suffix))

    # if os.path.exists(prepDir): # don't delete in case processing multiple scans for the same subject
    #    shutil.rmtree(prepDir)
    if os.path.exists(prepDir) is False:
        os.mkdir(prepDir)
        os.mkdir(os.path.join(prepDir, 'DWI_%ddir' % (nDirections)))

    # prepare the files for subsequent processing
    nrrdfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.nrrd' % (
        nDirections, phan_name, nDirections))
    niifilename = os.path.join(prepDir,
                               'DWI_%ddir/%s_DWI_%ddir.nii' % (nDirections, phan_name, nDirections))
    bvecsfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.bvecs' % (
        nDirections, phan_name, nDirections))
    bvalsfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.bvals' % (
        nDirections, phan_name, nDirections))
    btablefilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_btable.txt' % (
        nDirections, phan_name, nDirections))
    srcfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.src.gz' % (
        nDirections, phan_name, nDirections))

    orignrrdreportfname = os.path.join(prepDir, 'DWI_%ddir/origNrrdReport.txt' % (
        nDirections))

    if not os.path.exists(nrrdfilename):
        # Copy orig nrrd
        cmdStr = 'cp -Rv %s %s' % (origNrrdfilename, nrrdfilename)
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().
        # fix the nrrd file (thickness and directions are the last dimension)
        options = hardiIO.fixNRRDfile(nrrdfilename)
        fid = open(orignrrdreportfname, 'w')
        fid.write('----------------------------------\n')
        fid.write('Original nrrd file info\n')
        fid.write('----------------------------------\n')
        fid.write('Filename: {}\n'.format(origNrrdfilename))
        fid.write(options['study'] + '\n')
        fid.close()

    # convert to nifti + save bvecs and bvals
    if not os.path.exists(niifilename) or not os.path.exists(bvalsfilename) or not os.path.exists(bvecsfilename):
        hardiIO.convertToNIFTI(nrrdfilename, niifilename, bvecsfilename, bvalsfilename)

    # write the btable file
    if not os.path.exists(btablefilename):
        hardiIO.bvecsbvals2btable(bvalsfilename, bvecsfilename, btablefilename)

    # save src file
    if not os.path.exists(srcfilename):
        hardiIO.nifti2src(niifilename, btablefilename, srcfilename)

    if check_btable:
        recfilename = glob.glob(srcfilename+'.*')
        if len(recfilename) == 0:
            dsistudiopath = '/media/HDD2T/utils/dsistudio/dsi-studio-2018/dsi_studio_64/dsi_studio'
            # method 0:DSI, 1:DTI, 2:Funk-Randon QBI, 3:Spherical Harmonic QBI, 4:GQI 6: Convert to HARDI 7:QSDR.
            cmdcheckbtable ='{} --action=rec --source={} --method=1 --check_btable={}'.format(dsistudiopath,
                                                                                      srcfilename, check_btable)
            subprocess.Popen(cmdcheckbtable,
                             shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

    # Check before QC
    # bval bvec check
    bvalcount = len(open(bvalsfilename).readlines())
    bvalcountflag = False
    bvalbveczeroflag = False
    if bvalcount < 7:
        bvalcountflag = True
        flagfname = os.path.join(prepDir, 'DWI_%ddir/lessgradients' % (nDirections))
        with open(flagfname, 'w') as fp:
            pass
    else:
        bvals, bvecs = read_bvals_bvecs(bvalsfilename, bvecsfilename)
        if np.sum(bvals) == 0 or np.sum(bvecs) == 0:
            bvalbveczeroflag = True
            flagfname = os.path.join(prepDir, 'DWI_%ddir/bvalbveczero' % (nDirections))
            with open(flagfname, 'w') as fp:
                pass

    end_time = time.time()

    print('PrepareQCsession: time elapsed = %f seconds ...' % (end_time - start_time))

    return prepDir, nrrdfilename, bvalcountflag + bvalbveczeroflag


"""
---------------------------------------------------------------------------------
RunDTIPrepStage
---------------------------------------------------------------------------------

OBJECTIVE:
    dtiprep without motion correction
    check the quality of the individual directions, e.g. missing slices, intensity artifacts, Venetian blind
"""


def RunDTIPrepStage(prepDir, phan_name, xmlfilename, nDirections=65):
    start_time = time.time()

    nrrdfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.nrrd' % (nDirections, phan_name, nDirections))

    qcnrrdfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_QCed.nrrd' % (
        nDirections, phan_name, nDirections))

    QCReportfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_DTIPrepQCReport.txt' % (
    nDirections, phan_name, nDirections))

    cmdStr = dtiprepbin + 'DTIPrep --DWINrrdFile %s --xmlProtocol %s --check --outputFolder %s' % (
        nrrdfilename, xmlfilename, os.path.join(prepDir, 'DWI_%ddir' % (nDirections)))
    # os.system(cmdStr)
    if not os.path.exists(qcnrrdfilename) or not os.path.exists(QCReportfilename):
        subprocess.Popen(cmdStr, shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

    _, _, _, baselineIndex, _ = hardiIO.readHARDI(qcnrrdfilename)

    if baselineIndex < 0:
        baselineExcluded = True
    else:
        baselineExcluded = False

    if baselineExcluded:
        print('BASELINE EXCLUDED: STOP HARDIprep-simple')
        flagfname = os.path.join(prepDir, 'DWI_%ddir/nobaseline' % (nDirections))
        with open(flagfname, 'w') as fp:
            pass
        return baselineExcluded

    end_time = time.time()

    print('RunDTIPrepStage: time elapsed = %f seconds ...' % (end_time - start_time))




"""
---------------------------------------------------------------------------------
PerformWithinGradientMotionQC
---------------------------------------------------------------------------------

OBJECTIVE:

    quantify fast bulk motion within each gradient to exclude those having intra-scan
    motion (see Benner et al (2011) - Diffusion imaging with prospective motion correction and reacquisition)

    here we have zero tolerance, any gradient having at least one slice with signal drop out will be excluded

TODO:
    (0) fix the dimension and thickness of the given nrrd files
    (1) read the nrrd files (provided directly by Clement) listed in the nrrdfilenames.csv (only process data with > 30 gradients)
    (2) compute the signal drop-out per slice and a score for the whole gradient volume
    (3) report a QC report that includes scores and excluded scans and a nrrd file with the resulting QCed acquisition

Assumptions:
    (1) DTIprep stage has been performed

"""


def PerformWithinGradientMotionQC(prepDir,phan_name, nDirections=65):
    start_time = time.time()


    nrrdfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_QCed.nrrd' % (
        nDirections, phan_name, nDirections))

    reportfilename = os.path.join(prepDir,
                                  'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCreport.txt' % (
                                      nDirections, phan_name, nDirections))
    nrrdfilename_new = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nrrd' % (
                                        nDirections, phan_name, nDirections))

    niifilename = os.path.join(prepDir,
                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nii' % (
                                   nDirections, phan_name, nDirections))
    bvecsfilename = os.path.join(prepDir,
                                 'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.bvecs' % (
                                     nDirections, phan_name, nDirections))
    bvalsfilename = os.path.join(prepDir,
                                 'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.bvals' % (
                                     nDirections, phan_name, nDirections))
    btablefilename = os.path.join(prepDir,
                                  'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_btable.txt' % (
                                      nDirections, phan_name, nDirections))
    srcfilename = os.path.join(prepDir,
                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.src.gz' % (
                                   nDirections, phan_name, nDirections))

    # fix the nrrd file (thickness and directions are the last dimension)

    if not os.path.exists(nrrdfilename_new):
        hardiIO.fixNRRDfile(nrrdfilename)
        # read the nrrd data along with hardi protocol information
        nrrdData, bvalue, gradientDirections, baselineIndex, options = hardiIO.readHARDI(nrrdfilename)

        # detect within gradient motion artifact
        nMotionCorrupted, slice_numbers = hardiQCUtils.DetectWithinGradientMotion(nrrdData,
                                                                                  baselineIndex, bvalue)

        # write the QC report
        nExcluded = hardiQCUtils.WriteWithinGradientMotionQCReport(reportfilename, nMotionCorrupted,
                                                                   slice_numbers, baselineIndex)

        # construct the corrected sequence
        correctedData, gradientDirections_new = hardiQCUtils.ConstructWithinGradientMotionCorrectedData(
            nrrdData, gradientDirections, nExcluded, nMotionCorrupted)

        options = hardiIO.updateNrrdOptions(options, gradientDirections_new)

        # save as nrrd with the save options as the original nrrd file
        nrrd.write(nrrdfilename_new, correctedData.astype('int16'), options)

        # fix the nrrd file (thickness and directions are the last dimension)
        hardiIO.fixNRRDfile(nrrdfilename_new)

    if not os.path.exists(niifilename) or not os.path.exists(bvecsfilename) or not os.path.exists(bvalsfilename):
        # convert to nifiti + save bvecs and bvals
        hardiIO.convertToNIFTI(nrrdfilename_new, niifilename, bvecsfilename, bvalsfilename)

    if not os.path.exists(btablefilename):
        # write the btable file
        hardiIO.bvecsbvals2btable(bvalsfilename, bvecsfilename, btablefilename)

    # save src file
    # hardiIO.nifti2src(niifilename, btablefilename, srcfilename)
    end_time = time.time()

    print('PerformWithinGradientMotionQC: time elapsed = %f seconds ...' % (end_time - start_time))



"""
---------------------------------------------------------------------------------
ExtractBaselineAndBrainMask
---------------------------------------------------------------------------------
Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed


TODO:
    (0) fix the dimension and thickness of the given nrrd files
    (1) extract the baseline
    (2) convert them to nifti
    (3) save the corresponding bvecs, bvals and btable
    (4) save the src files for visualization in dsi_studio
    (5) use the baseline to extract brain mask from baseline (to be used in motion correction) but don't do the actual masking
"""


def ExtractBaselineAndBrainMask(prepDir, phan_name, nDirections=65):
    start_time = time.time()

    nrrdfilename = os.path.join(prepDir,
                                'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nrrd' % (
                                    nDirections, phan_name, nDirections))

    baselineDir = os.path.join(prepDir, 'baseline')
    baselinenrrdfilename = os.path.join(prepDir, 'baseline/%s_baseline.nrrd' % (phan_name))

    baselinenrrdfilename_masked = os.path.join(prepDir,
                                               'baseline/%s_baseline_masked.nrrd' % (phan_name))
    baselineniifilename_masked = os.path.join(prepDir,
                                              'baseline/%s_baseline_masked.nii' % (phan_name))

    baselineNiifilename = os.path.join(prepDir, 'baseline/%s_baseline.nii' % (phan_name))
    tempbvecsfilename = os.path.join(prepDir, 'baseline/%s_baseline.bvecs' % (phan_name))
    tempbvalsfilename = os.path.join(prepDir, 'baseline/%s_baseline.bvals' % (phan_name))

    niiBrainFilename = os.path.join(prepDir, 'baseline/%s_baseline_brain.nii' % (phan_name))
    niiBrainMaskFilename = os.path.join(prepDir,
                                        'baseline/%s_baseline_brain_mask.nii' % (phan_name))

    if not os.path.exists(baselinenrrdfilename):
        if os.path.exists(baselineDir) == False:
            os.mkdir(baselineDir)

        # save the baseline to nrrd
        hardiIO.extractAndSaveBaselineToNRRD(nrrdfilename, baselinenrrdfilename)

    if not os.path.exists(baselineNiifilename):
        # then convert to nifti
        hardiIO.convertToNIFTI(baselinenrrdfilename, baselineNiifilename, tempbvecsfilename,
                               tempbvalsfilename)
        cmdStr = 'rm -f %s' % (tempbvecsfilename)
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().
        cmdStr = 'rm -f %s' % (tempbvalsfilename)
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

    print('Extract brain region ...')
    if not os.path.exists(niiBrainMaskFilename):
        if os.path.exists(niiBrainMaskFilename+'.gz'):
            niiBrainMaskFilename = niiBrainMaskFilename+'.gz'
        else:
            # get the mask based on FSL-BET tool - all nrrd files are converted to nifti in a bash script
            brainMask = hardiQCUtils.extractBrainRegion(os.path.abspath(baselineNiifilename),
                                                        os.path.abspath(niiBrainFilename),
                                                        os.path.abspath(niiBrainMaskFilename))

            print('Mask the baseline volume ...')
            nrrdData, options = nrrd.read(baselinenrrdfilename)
            nrrdDataMasked = hardiQCUtils.brainMaskingVolume(nrrdData, brainMask)
            nrrd.write(baselinenrrdfilename_masked, nrrdDataMasked.astype('short'), options)
            hardiIO.convertToNIFTI(baselinenrrdfilename_masked, baselineniifilename_masked,
                                   tempbvecsfilename, tempbvalsfilename)

            cmdStr = 'rm -f %s' % (tempbvecsfilename)
            # os.system(cmdStr)
            subprocess.Popen(cmdStr,
                             shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

            cmdStr = 'rm -f %s' % (tempbvalsfilename)
            # os.system(cmdStr)
            subprocess.Popen(cmdStr,
                             shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

    end_time = time.time()

    print('ExtractBaselineAndBrainMask: time elapsed = %f seconds ...' % (end_time - start_time))



"""
---------------------------------------------------------------------------------
PerformResampleCorruptedSlicesInQspace
---------------------------------------------------------------------------------

OBJECTIVE:
    resample the corrupted slices (detected via DTIPrep and within-gradient motion) Qc
    in the q-space

    This is the implementation of the correction strategy based on the below paper
    in order to replace DTIPrep (which excluded gradients suffering from intensity artifacts)
    this correction strategy is trying to correct for within-slice, within-volume and betwee-volumes
    motion artifacts ...

    Dubois, Jessica, Sofya Kulikova, Lucie Hertz-Pannier, Jean-François Mangin, Ghislaine Dehaene-Lambertz,
    and Cyril Poupon. "Correction strategy for diffusion-weighted images corrupted with motion:
    application to the DTI evaluation of infants' white matter." Magnetic resonance imaging 32, no. 8 (2014): 981-992.

Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) baseline and brain mask extraction

TODO:
    (1) get a list of gradients-slices that are to be resampled (identified as corrupted via DTIprep and within gradient motion QC)
    (2) for the corrupted slices, fit Qball on the non-corrupted data (regularized version based on Descoteaux, M., et. al. 2007. Regularized, fast, and robust analytical
        Q-ball imaging.)
    (3) resample corrupted slices based on the fitted dODF
    (4) save as nrrd, nii and other formats
"""


def PerformResampleCorruptedSlicesInQspace(prepDir, phan_name, resampling_method='shore', nDirections=65):
    start_time = time.time()

    nrrdfilename_orig = os.path.join(prepDir, 'DWI_%sdir/%s_DWI_%sdir.nrrd' % (
        str(nDirections), phan_name, str(nDirections)))
    niifilename_orig = os.path.join(prepDir, 'DWI_%sdir/%s_DWI_%sdir.nii' % (
        str(nDirections), phan_name, str(nDirections)))
    bvecsfilename_orig = os.path.join(prepDir, 'DWI_%sdir/%s_DWI_%sdir.bvecs' % (
        str(nDirections), phan_name, str(nDirections)))
    bvalsfilename_orig = os.path.join(prepDir, 'DWI_%sdir/%s_DWI_%sdir.bvals' % (
        str(nDirections), phan_name, str(nDirections)))
    btablefilename_orig = os.path.join(prepDir, 'DWI_%sdir/%s_DWI_%sdir_btable.txt' % (
        str(nDirections), phan_name, str(nDirections)))

    nrrdfilename = os.path.join(prepDir, 'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed.nrrd' % (
        str(nDirections), phan_name, str(nDirections)))
    niifilename = os.path.join(prepDir,
                               'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed.nii' % (
                                   str(nDirections), phan_name, str(nDirections)))
    bvecsfilename = os.path.join(prepDir,
                                 'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed.bvecs' % (
                                     str(nDirections), phan_name, str(nDirections)))
    bvalsfilename = os.path.join(prepDir,
                                 'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed.bvals' % (
                                     str(nDirections), phan_name, str(nDirections)))
    btablefilename = os.path.join(prepDir,
                                  'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed_btable.txt' % (
                                      str(nDirections), phan_name, str(nDirections)))


    nrrdfilename_rq = os.path.join(prepDir,
                                   'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed_Resample%s.nrrd' % (
                                       str(nDirections), phan_name, str(nDirections), resampling_method.upper()))

    niifilename_rq = os.path.join(prepDir,
                                  'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed_Resample%s.nii' % (
                                      str(nDirections), phan_name, str(nDirections), resampling_method.upper()))
    bvecsfilename_rq = os.path.join(prepDir,
                                    'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed_Resample%s.bvecs' % (
                                        str(nDirections), phan_name, str(nDirections), resampling_method.upper()))
    bvalsfilename_rq = os.path.join(prepDir,
                                    'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed_Resample%s.bvals' % (
                                        str(nDirections), phan_name, str(nDirections), resampling_method.upper()))
    btablefilename_rq = os.path.join(prepDir,
                                     'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed_Resample%s_btable.txt' % (
                                         str(nDirections), phan_name, str(nDirections),
                                         resampling_method.upper()))
    srcfilename_rq = os.path.join(prepDir,
                                  'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCed_Resample%s.src.gz' % (
                                      str(nDirections), phan_name, str(nDirections), resampling_method.upper()))
    resamplingDir = os.path.join(prepDir, 'resampling')
    bvalcountflag = False

    if not os.path.exists(resamplingDir):
        c = "mkdir %s" % (resamplingDir)
        subprocess.Popen(c, shell=True).wait()

    if not os.path.exists(nrrdfilename_rq):

        # (1) Check baseline
        nrrdData, bvalue, gradientDirections, baselineIndex, options = hardiIO.readHARDI(nrrdfilename)
        print("Baseline Index: ", baselineIndex)
        if baselineIndex < 0:  # baseline was excluded during dtiprep, no point to process this dataset
            return

        # (2) get which slices in which gradients are needed to be resampled
        diffusionData_orig, bvalue_orig, gradientDirections_orig, baselineIndex_orig, options_orig = hardiIO.readHARDI(
            nrrdfilename_orig)

        bvals_orig, bvecs_orig = hardiIO.readbtable(btablefilename_orig)
        gtable_orig = gradient_table(bvals_orig, bvecs_orig)

        gradientDirections_orig = np.squeeze(gradientDirections_orig)
        rows, cols, nSlices, nDirections_ = diffusionData_orig.shape

        # get the gradients and slices which suffer from slice-wise artifacts
        toBeResampled = np.zeros((nDirections, nSlices))

        # from DTIPrep - slice-wise intensity artifacts
        excluded = list()
        qcreportfilename = os.path.join(prepDir, 'DWI_%sdir/%s_DWI_%sdir_DTIPrepQCReport.txt' % (
            str(nDirections), phan_name, str(nDirections)))
        # qcreportfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_QCReport.txt' % (
        # nDirections, phan_name, nDirections))

        # print
        # "============================="
        # print "qcreportfilename:", qcreportfilename
        # print
        # "============================="

        fid = open(qcreportfilename, 'r')
        for line in fid:
            line = line.strip()
            if line.find('Slice-wise Check Artifacts:') >= 0:
                break

        for line in fid:
            line = line.strip().split('\t')
            if line[0].find('whole') < 0:
                continue
            d = int(line[1])
            s = int(line[2])
            toBeResampled[d, s] = 1
            excluded.append(d)

        fid.close()


        # from DTIPrep - interlace-wise intensity artifacts - aparently not excluded via dtiprep
        qcreportfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_DTIPrepQCReport.txt' % (nDirections, phan_name, nDirections))
        fid = open(qcreportfilename, 'r')
        for line in fid:
           line = line.strip()
           if line.find('Interlace-wise Check Artifacts:') >= 0:
               break


        for line in fid:
           if line.find('Gradient#') >= 0:
               continue
           if line[0] == '=':
               print("No interlace-wise artifacts")
               break
           elif line.find('Gradient direction #is less than 6!'):
               print("Gradient # less than 6!")
               bvalcountflag = True
               flagfname = os.path.join(prepDir, 'DWI_%ddir/lessgradients' % (nDirections))
               with open(flagfname, 'w') as fp:
                   pass
               return bvalcountflag
           line = line.strip().split('\t')
           d = int(line[0])
           for s in range(0,nSlices):
               toBeResampled[d,s] = 1
           #
           excluded.append(d)
        #
        fid.close()

        excluded = np.unique(np.array(excluded))
        # from DTIprep - interlace artifacts
        # see if  there are any other directions that were excluded due to interslice artifacts
        qcreportfilename = os.path.join(prepDir, 'DWI_%sdir/%s_DWI_%sdir_QCReport.txt' % (
            nDirections, phan_name, nDirections))
        dirIndex = dict()  # mapping that maps the direction index in dtiprep output to the original index
        fid = open(qcreportfilename, 'r')
        included = np.zeros((nDirections,))

        for line in fid:
            if line.find('QCIndex') >= 0 and line.find('Included Gradients:') >= 0:
                line = line.strip().split(' ')
                d = int(line[2])
                included[d] = 1
                dirIndex[int(line[4])] = int(line[2])

        fid.close()
        included[baselineIndex_orig] = 1

        for d in range(nDirections):
            if included[d] == 0:
                if len(np.where(excluded == d)[0]) == 0:  # was not excluded due to slice artifacts
                    for s in range(nSlices):
                        toBeResampled[d, s] = 1

        # from within-gradient motion QC
        qcreportfilename = os.path.join(prepDir,
                                        'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotionQCreport.txt' % (
                                            str(nDirections), phan_name, str(nDirections)))
        fid = open(qcreportfilename, 'r')
        for line in fid:
            if line.find('Excluded') >= 0:
                line = line.strip().split()
                dirind = int(line[0])
                nslices = int(line[2])
                for i in range(nslices):
                    s = int(line[3 + i])
                    d = dirIndex[dirind]
                    toBeResampled[d, s] = 1
        fid.close()

        qcreportfilename = os.path.join(prepDir,
                                        'DWI_%sdir/%s_DWI_%sdir_QCed_WithinGradientMotion_ResamplingReport.txt' % (
                                            str(nDirections), phan_name, str(nDirections)))
        fid = open(qcreportfilename, 'w')
        fid.write(' ,')
        for slice_number in range(nSlices):
            fid.write('slice # %d ,' % (slice_number))
        fid.write('\n')
        direction_number = -1
        for row in toBeResampled:
            direction_number = direction_number + 1
            fid.write('gradient direction # %d, ' % (direction_number))
            fid.write(','.join(map(str, row)) + '\n')
        fid.close()

        # (3)-0 Before resampling, we linearly register directional volumes to volume with corrupted slices.

        dataset_bad_vol_regist = list()
        ind_bad_vol = list()
        ## loop on bad volumes
        for ind, gradDir in enumerate(gradientDirections_orig):
            if np.sum(toBeResampled[ind, :]) != 0:  # corrupted volume
                niifilename_temp_corr = os.path.join(resamplingDir, 'MCto%d.nii.gz' % (ind))
                cmdStr = 'mcflirt -in ' + niifilename_orig[0:-4] + ' -out ' + niifilename_temp_corr.split('.nii.gz')[0] + ' -cost ' + 'normmi' + ' -refvol ' + str(
                    ind) + ' -dof 6 -stages 4 -verbose 1 -stats -mats -plots -report'
                subprocess.Popen(cmdStr, shell=True).wait()
                dataset_bad_vol_regist.append(niifilename_temp_corr)
                ind_bad_vol.append(ind)

        # (3) start resampling
        diffusionData_resampled = copy.deepcopy(diffusionData_orig)
        for vol in range(len(dataset_bad_vol_regist)):  # loop on bad volumes
            img1 = nib.load(dataset_bad_vol_regist[vol])
            diffusionData_to_be_use = np.asanyarray(img1.dataobj)
            ind = ind_bad_vol[vol]
            mcParamFolder = os.path.join(resamplingDir, 'MCto%d.mat' % (ind))
            bvals_corrected, bvecs_corrected = hardiQCUtils.ReorientBmatrix_mcflirt(bvals_orig, bvecs_orig, mcParamFolder)
            for kk in range(nSlices):
                if toBeResampled[ind, kk] == 0:
                    continue
                # for the current slice, get all the valid directions to be used to estimate the slice-wise diffusion model
                cur_bvecs = list()
                cur_bvals = list()
                valid_indices = list()
                for indgrad, gradDir in enumerate(gradientDirections_orig):
                    if toBeResampled[
                        indgrad, kk] == 1:  # this direction is corrupted, don't include it in the estimation process
                        continue
                    cur_bvals.append(bvals_corrected[indgrad])
                    cur_bvecs.append(bvecs_corrected[indgrad, :])
                    valid_indices.append(indgrad)
                cur_bvals = np.array(cur_bvals)
                cur_bvecs = np.array(cur_bvecs)
                cur_gtable = gradient_table(cur_bvals, cur_bvecs)
                curDiffusionData = copy.deepcopy(diffusionData_to_be_use[:, :, kk, valid_indices])
                if resampling_method == 'qbi':
                    qball_model = drecon.QballModel(cur_gtable, sh_order=6, smooth=0.006)
                    sphHarmFit = qball_model.fit(curDiffusionData)
                    shm_coeff = sphHarmFit.shm_coeff
                else:  # SHORE
                    radial_order = 6
                    zeta = 700
                    lambdaN = 1e-8
                    lambdaL = 1e-8
                    asm = ShoreModel(cur_gtable, radial_order=radial_order,
                                     zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
                    asmfit = asm.fit(curDiffusionData)
                    shore_coeff = asmfit.shore_coeff
                    shoreMatrix = shore_matrix(radial_order, zeta, gtable_orig)
                    print('Resampling slice = %d, dir = %d ...' % (kk, ind))
                    for ii in range(rows):
                        for jj in range(cols):
                            S0 = diffusionData_to_be_use[ii, jj, kk, baselineIndex_orig]
                            if resampling_method == 'qbi':
                                shCoeffs = shm_coeff[ii, jj, :]
                                r, theta, phi = drecon.cart2sphere(gradDir[0], gradDir[1], gradDir[2])
                                shBasis, m, n = drecon.real_sym_sh_basis(6, phi, theta)
                                shBasis = shBasis.flatten()
                                diffusionData_resampled[ii, jj, kk, ind] = S0 * np.dot(shBasis, shCoeffs)
                            else:  # SHORE
                                shoreCoeff = shore_coeff[ii, jj, :]
                                if np.any(np.isnan(shoreCoeff)):
                                    diffusionData_resampled[ii, jj, kk, ind] = 0.0
                                else:
                                    diffusionData_resampled[ii, jj, kk, ind] = S0 * np.dot(
                                        shoreMatrix[ind, :], shoreCoeff)

        nrrd.write(nrrdfilename_rq, diffusionData_resampled, options_orig)

        # fix the nrrd file (thickness and directions are the last dimension)
        _ = hardiIO.fixNRRDfile(nrrdfilename_rq)

    if not os.path.exists(niifilename_rq) or not os.path.exists(bvecsfilename_rq) or not os.path.exists(bvalsfilename_rq):
        # convert to nifiti + save bvecs and bvals
        hardiIO.convertToNIFTI(nrrdfilename_rq, niifilename_rq, bvecsfilename_rq, bvalsfilename_rq)

    if not os.path.exists(btablefilename_rq):
        # write the btable file
        hardiIO.bvecsbvals2btable(bvalsfilename_rq, bvecsfilename_rq, btablefilename_rq)

    end_time = time.time()

    print(
        'PerformResampleCorruptedSlicesInQspace: time elapsed = %f seconds ...' % (
                end_time - start_time))

    return bvalcountflag


"""
---------------------------------------------------------------------------------
RerunDTIPrepStage
---------------------------------------------------------------------------------

OBJECTIVE:
    Exclusion of any direction that has artifacts not curable by the resampling stage
"""


def RerunDTIPrepStage(prepDir, phan_name, xmlfilename, resampling_method, nDirections ):
    start_time = time.time()

    nrrdfilename = os.path.join(prepDir,
                                'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s.nrrd' % (
                                    nDirections, phan_name, nDirections, resampling_method.upper()))
    nrrdfilename_qc = os.path.join(prepDir,
                                   'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.nrrd' % (
                                       nDirections, phan_name, nDirections, resampling_method.upper()))
    niifilename_qc = os.path.join(prepDir,
                                  'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.nii' % (
                                      nDirections, phan_name, nDirections, resampling_method.upper()))

    bvecsfilename_qc = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.bvecs' % (
                                        nDirections, phan_name, nDirections, resampling_method.upper()))
    bvalsfilename_qc = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.bvals' % (
                                        nDirections, phan_name, nDirections, resampling_method.upper()))
    btablefilename_qc = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_btable.txt' % (
                                         nDirections, phan_name, nDirections,
                                         resampling_method.upper()))
    srcfilename_qc = os.path.join(prepDir,
                                  'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.src.gz' % (
                                      nDirections, phan_name, nDirections, resampling_method.upper()))

    #cmdStr = 'DTIPrep --DWINrrdFile %s --xmlProtocol %s --check --outputFolder %s' % (
    #nrrdfilename, xmlfilename, os.path.join(prepDir, 'DWI_%ddir' % (nDirections)))
    # os.system(cmdStr)
    #subprocess.Popen(cmdStr,
    #                 shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

    QCReportfilename = os.path.join(prepDir,
                                   'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_DTIPrepQCReport.txt' % (
                                       nDirections, phan_name, nDirections, resampling_method.upper()))

    if not os.path.exists(nrrdfilename_qc) or not os.path.exists(QCReportfilename):
        cmdStr = dtiprepbin + 'DTIPrep --DWINrrdFile %s --xmlProtocol %s --check --outputFolder %s' % (
            nrrdfilename, xmlfilename, os.path.join(prepDir, 'DWI_%ddir' % (nDirections)))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

    if not os.path.exists(niifilename_qc) or not os.path.exists(bvecsfilename_qc) or not os.path.exists(bvalsfilename_qc):
        # fix the nrrd file (thickness and directions are the last dimension)
        hardiIO.fixNRRDfile(nrrdfilename_qc)

        # convert to nifiti + save bvecs and bvals
        hardiIO.convertToNIFTI(nrrdfilename_qc, niifilename_qc, bvecsfilename_qc, bvalsfilename_qc)

    if not os.path.exists(btablefilename_qc):
        # write the btable file
        hardiIO.bvecsbvals2btable(bvalsfilename_qc, bvecsfilename_qc, btablefilename_qc)

    # # save src file
    # hardiIO.nifti2src(niifilename_qc, btablefilename_qc, srcfilename_qc)

    end_time = time.time()

    print
    ('RerunDTIPrepStage, %s: time elapsed = %f seconds ...' % (
        resampling_method, end_time - start_time))




"""
---------------------------------------------------------------------------------
RerunDTIPrepStage
---------------------------------------------------------------------------------

OBJECTIVE:
    Exclusion of any direction that has artifacts not curable by the resampling stage
"""


def RerunDTIPrepStage(prepDir, phan_name, xmlfilename, resampling_method, nDirections=65):
    start_time = time.time()

    # basename, phan_name = ParseFilename(nrrdfilename)

    nrrdfilename = os.path.join(prepDir,
                                'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s.nrrd' % (
                                    nDirections, phan_name, nDirections, resampling_method.upper()))
    nrrdfilename_qc = os.path.join(prepDir,
                                   'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.nrrd' % (
                                       nDirections, phan_name, nDirections, resampling_method.upper()))
    niifilename_qc = os.path.join(prepDir,
                                  'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.nii' % (
                                      nDirections, phan_name, nDirections, resampling_method.upper()))

    bvecsfilename_qc = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.bvecs' % (
                                        nDirections, phan_name, nDirections, resampling_method.upper()))
    bvalsfilename_qc = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.bvals' % (
                                        nDirections, phan_name, nDirections, resampling_method.upper()))
    btablefilename_qc = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_btable.txt' % (
                                         nDirections, phan_name, nDirections,
                                         resampling_method.upper()))
    srcfilename_qc = os.path.join(prepDir,
                                  'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.src.gz' % (
                                      nDirections, phan_name, nDirections, resampling_method.upper()))

    cmdStr = dtiprepbin + 'DTIPrep --DWINrrdFile %s --xmlProtocol %s --check --outputFolder %s' % (
        nrrdfilename, xmlfilename, os.path.join(prepDir, 'DWI_%ddir' % (nDirections)))
    # os.system(cmdStr)
    if not os.path.exists(nrrdfilename_qc):
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        # fix the nrrd file (thickness and directions are the last dimension)
        _ = hardiIO.fixNRRDfile(nrrdfilename_qc)

    if not os.path.exists(niifilename_qc) or not os.path.exists(bvalsfilename_qc) or not os.path.exists(bvecsfilename_qc):
        # convert to nifiti + save bvecs and bvals
        hardiIO.convertToNIFTI(nrrdfilename_qc, niifilename_qc, bvecsfilename_qc, bvalsfilename_qc)

    if not os.path.exists(btablefilename_qc):
        # write the btable file
        hardiIO.bvecsbvals2btable(bvalsfilename_qc, bvecsfilename_qc, btablefilename_qc)

    # # save src file
    # hardiIO.nifti2src(niifilename_qc, btablefilename_qc, srcfilename_qc)

    end_time = time.time()

    print(
        'RerunDTIPrepStage, %s: time elapsed = %f seconds ...' % (
            resampling_method, end_time - start_time))




def GradientWiseDenoise(nrrdfilename, prepDir, phan_name, resampling_method, nDirections=65,
                        EstimationRadius=11, FilteringRadius=11, minVoxelsEstimation=3,
                        minVoxelsFiltering=3):
    start_time = time.time()

    EstimationRadius_x = EstimationRadius
    EstimationRadius_y = EstimationRadius
    EstimationRadius_z = 0
    FilteringRadius_x = FilteringRadius
    FilteringRadius_y = FilteringRadius
    FilteringRadius_z = 0
    NoOfIterations = 1
    # minVoxelsFiltering        = 7
    # minVoxelsEstimation       = 7
    histRes = 2
    minStd = 0
    maxStd = 100
    absVal = 0

    # basename, phan_name = ParseFilename(nrrdfilename)

    if resampling_method == None:
        nrrdfilename = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nrrd' % (
                                        nDirections, phan_name, nDirections))
        nrrdfilename_qc = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE.nrrd' % (
                                           nDirections, phan_name, nDirections))
        niifilename_qc = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE.nii' % (
                                          nDirections, phan_name, nDirections))

        bvecsfilename_qc = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE.bvecs' % (
                                            nDirections, phan_name, nDirections))
        bvalsfilename_qc = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE.bvals' % (
                                            nDirections, phan_name, nDirections))
        btablefilename_qc = os.path.join(prepDir,
                                         'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE_btable.txt' % (
                                             nDirections, phan_name, nDirections))
        srcfilename_qc = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE.src.gz' % (
                                          nDirections, phan_name, nDirections))
    else:
        nrrdfilename = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.nrrd' % (
                                        nDirections, phan_name, nDirections, resampling_method.upper()))
        nrrdfilename_qc = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE.nrrd' % (
                                           nDirections, phan_name, nDirections,
                                           resampling_method.upper()))
        niifilename_qc = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE.nii' % (
                                          nDirections, phan_name, nDirections,
                                          resampling_method.upper()))

        bvecsfilename_qc = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE.bvecs' % (
                                            nDirections, phan_name, nDirections,
                                            resampling_method.upper()))
        bvalsfilename_qc = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE.bvals' % (
                                            nDirections, phan_name, nDirections,
                                            resampling_method.upper()))
        btablefilename_qc = os.path.join(prepDir,
                                         'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE_btable.txt' % (
                                             nDirections, phan_name, nDirections,
                                             resampling_method.upper()))
        srcfilename_qc = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE.src.gz' % (
                                          nDirections, phan_name, nDirections,
                                          resampling_method.upper()))

    if EstimationRadius >= 1 and FilteringRadius >= 1:
        # print('==============================================')
        # print 'DWIRicianLMMSE %s %s %f %f %f %f %f %f %f %f %f %f %f %f %f' % (
        # nrrdfilename, nrrdfilename_qc, FilteringRadius_x, FilteringRadius_y, FilteringRadius_z,
        # EstimationRadius_x, EstimationRadius_y, EstimationRadius_z, NoOfIterations,
        # minVoxelsFiltering, minVoxelsEstimation, histRes, minStd, maxStd, absVal)
        # print('==============================================')

        cmdStr = 'DWIRicianLMMSE %s %s %f %f %f %f %f %f %f %f %f %f %f %f %f' % (
            nrrdfilename, nrrdfilename_qc, FilteringRadius_x, FilteringRadius_y, FilteringRadius_z,
            EstimationRadius_x, EstimationRadius_y, EstimationRadius_z, NoOfIterations,
            minVoxelsFiltering, minVoxelsEstimation, histRes, minStd, maxStd, absVal)

    else:
        cmdStr = 'cp -Rvp %s %s' % (nrrdfilename, nrrdfilename_qc)
    # os.system(cmdStr)
    subprocess.Popen(cmdStr,
                     shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

    # fix the nrrd file (thickness and directions are the last dimension)

    hardiIO.fixNRRDfile(nrrdfilename_qc)

    # convert to nifiti + save bvecs and bvals
    hardiIO.convertToNIFTI(nrrdfilename_qc, niifilename_qc, bvecsfilename_qc, bvalsfilename_qc)

    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename_qc, bvecsfilename_qc, btablefilename_qc)

    # save src file
    hardiIO.nifti2src(niifilename_qc, btablefilename_qc, srcfilename_qc)

    end_time = time.time()

    print
    'GradientWiseDenoise: time elapsed = %f seconds ...' % (end_time - start_time)




"""
PerformDWIBaselineReferenceMotionCorrection

OBJECTIVE:
    correct for motion the DTIPrep QCed sequences where slice-wise/gradient-wise denoising already
    been carried out, as such we can use standard interpolation

Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) baseline and brain mask extraction
    (4) q-space resampling of corrupted slices has been performed along with gradient-wise denoising

TODO:
    (0) fix the dimension and thickness of the given nrrd files (if any)
    (1) ?
    (2) ?
    (3) ?
"""

def PerformDWIBaselineReferenceMotionCorrection(prepDir,phan_name,
                                       resampling_method, nDirections=65,
                                       ):

    start_time = time.time()
    interpMethod = 'trilinear'

    # basename, phan_name = ParseFilename(nrrdfilename)

    # useBrainMask  = True # false if we have severe motion (> 5deg), otherwise try to avoid background noise when doing motion correction
    if resampling_method == None:
        nrrdfilename = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nrrd' % (
                                        nDirections, phan_name, nDirections))
        niifilename = os.path.join(prepDir,
                                   'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nii' % (
                                       nDirections, phan_name, nDirections))

        bvalsfilename = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.bvals' % (
                                         nDirections, phan_name, nDirections))
        bvecsfilename = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.bvecs' % (
                                         nDirections, phan_name, nDirections))

        nrrdfilename_MC = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID.nrrd' % (
                                           nDirections, phan_name, nDirections))
        niifilename_MC = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID.nii' % (
                                          nDirections, phan_name, nDirections))

        bvecsfilename_MC = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID.bvecs' % (
                                            nDirections, phan_name, nDirections))
        bvalsfilename_MC = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID.bvals' % (
                                            nDirections, phan_name, nDirections))
        btablefilename_MC = os.path.join(prepDir,
                                         'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID_btable.txt' % (
                                             nDirections, phan_name, nDirections))
        srcfilename_MC = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID.src.gz' % (
                                          nDirections, phan_name, nDirections))

        reportfilename = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID_QCreport.txt' % (
                                          nDirections, phan_name, nDirections))

    else:
        nrrdfilename = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.nrrd' % (
                                        nDirections, phan_name, nDirections, resampling_method.upper()))
        niifilename = os.path.join(prepDir,
                                   'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.nii' % (
                                       nDirections, phan_name, nDirections, resampling_method.upper()))

        bvalsfilename = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.bvals' % (
                                         nDirections, phan_name, nDirections,
                                         resampling_method.upper()))
        bvecsfilename = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.bvecs' % (
                                         nDirections, phan_name, nDirections,
                                         resampling_method.upper()))

        nrrdfilename_MC = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID.nrrd' % (
                                           nDirections, phan_name, nDirections,
                                           resampling_method.upper()))
        niifilename_MC = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID.nii' % (
                                          nDirections, phan_name, nDirections,
                                          resampling_method.upper()))

        bvecsfilename_MC = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID.bvecs' % (
                                            nDirections, phan_name, nDirections,
                                            resampling_method.upper()))
        bvalsfilename_MC = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID.bvals' % (
                                            nDirections, phan_name, nDirections,
                                            resampling_method.upper()))
        btablefilename_MC = os.path.join(prepDir,
                                         'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID_btable.txt' % (
                                             nDirections, phan_name, nDirections,
                                             resampling_method.upper()))
        srcfilename_MC = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID.src.gz' % (
                                          nDirections, phan_name, nDirections,
                                          resampling_method.upper()))
        reportfilename = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID_QCreport.txt' % (
                                          nDirections, phan_name, nDirections,
                                          resampling_method.upper()))


    # baseline was excluded during dtiprep, no point to process this dataset
    if os.path.exists(nrrdfilename) == False:
        return

    mcOutDir = os.path.join(prepDir, 'DWI_%ddir/BASELINE_ANTsRIGID_OUT' % (nDirections))
    if not os.path.exists(mcOutDir):
        os.mkdir(mcOutDir)

    # Align DWIs to baseline (First volume for IBIS)
    niifilename_for_ants_out = os.path.join(mcOutDir,
                                               'ants_dwi_out.nii.gz')
    # Always with brain mask
    niimaskedfilename = hardiQCUtils.extractBrainRegionDWI(niifilename)
    # niifilename_for_mcflirt_in = niifilename_for_mcflirt_in + '.gz'
    # shutil.copyfile(niimaskedfilename, niifilename_for_mcflirt_in)

    fbase = niifilename.split('.')[0]
    bvalsfilename = fbase + '.bvals'
    bvalcount = len(open(bvalsfilename).readlines())
    mcParamFolder = os.path.join(mcOutDir, 'antsmat2fsl_out.mat')
    if not os.path.exists(mcParamFolder):
        os.mkdir(mcParamFolder)

    if not os.path.exists(niifilename_for_ants_out):
        dwisplitname = os.path.join(fbase, 'dwi')

        fcollect = ''
        for idx in range(bvalcount):
            print(idx, end=' ')
            dwi_moving = dwisplitname + str(idx).zfill(4) + '_brain.nii.gz'
            dwibaseline_fixed =  dwisplitname + str(0).zfill(4) + '_brain.nii.gz'
            outname = os.path.join(mcOutDir,'dwi'+str(idx).zfill(4))
            fcollect = fcollect + outname + '_warped.nii.gz' + ' '

            if os.path.exists(outname+'0GenericAffine.mat'):
                continue

            antsCommand = 'antsRegistration -d 3 -o [{}, {}_warped.nii.gz] -n BSpline --transform Rigid[0.1] ' \
                          '--metric MI[{}, {}, 1, 32, Regular, 0.25] ' \
                          '--convergence [1000x500x250x100,1e-6,10]     --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox'.format(outname, outname,dwibaseline_fixed,dwi_moving)
            subprocess.Popen(antsCommand,
                             shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

            matants2fslCommand = '/home/heejong/HDD2T/utils/c3d-1.1.0-Linux-x86_64/bin/c3d_affine_tool -itk {} -ref {} -src {} -ras2fsl -o {} '.format(outname+'0GenericAffine.mat',dwibaseline_fixed,dwi_moving,mcParamFolder + '/MAT_'+str(idx).zfill(4))
            subprocess.Popen(matants2fslCommand,
                             shell=True).wait()  # subprocess.Popen() is strict superset of os.system().


        outname_ = os.path.join(mcOutDir,'dwi*'+'_warped.nii.gz')
        cmdStr = 'rm -rf %s' % (outname_)
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        fslmergecommand = 'fslmerge -t {} {}'.format(niifilename_for_ants_out, fcollect)
        subprocess.Popen(fslmergecommand,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

    # quantify the current motion
    if not os.path.exists(reportfilename):
        motionQuantification = hardiQCUtils.QuantifyMotionAnts(mcOutDir, baselineIndex=0, gradvols=bvalcount)
        hardiQCUtils.WriteMotionCorrectionQCReport(motionQuantification, reportfilename)

    nointegerflag = False
    if not os.path.exists(nrrdfilename_MC):

        correctedData = np.asanyarray(nib.load(niifilename_for_ants_out).dataobj)
        affine = nib.load(niifilename_for_ants_out).affine
        # if you want to make images consistent with other steps (with fsl, smoother result)
        # correctedData, affine = hardiQCUtils.ApplyAntsFSL(niimaskedfilename, mcParamFolder, interpMethod, fbase)

        # # Check if corrected data has significant digit below tenths (WILL RAISE CONVERSION PROBLEM)
        # if np.sum((correctedData*10) %10) > 0.0:
        #     nointegerflag = True
        #     return nointegerflag

        # update the btable
        bvals, bvecs = hardiIO.readbvalsbvecs(bvalsfilename, bvecsfilename)

        baselineIndex = np.where(bvals == 0)[0][0]
        bvals_corrected, bvecs_corrected = hardiQCUtils.ReorientBmatrix_ants(bvals, bvecs, mcOutDir)

        # load the corrected data
        _, options = nrrd.read(nrrdfilename)

        # update the btable in the nrrd options then start the saving cycle
        options = hardiIO.updateNrrdOptions(options, bvecs_corrected)

        # save as nrrd with the save options as the original nrrd file
        nrrd.write(nrrdfilename_MC, correctedData, options) # No longer short
        hardiIO.save2nii(niifilename_MC, correctedData, affine=affine) # No longer short
        # add Fix header (qform / sform) ?
        #!!!qform format index change when saved nifti https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/qsform.html
        hardiIO.save_bval_bvec(bvals_corrected, bvecs_corrected, bvalsfilename_MC.split('.')[0])

    # if not os.path.exists(niifilename_MC) or not os.path.exists(bvecsfilename_MC) or not os.path.exists(bvalsfilename_MC):
    #     # convert to nifiti + save bvecs and bvals
    #     hardiIO.convertToNIFTI(nrrdfilename_MC, niifilename_MC, bvecsfilename_MC, bvalsfilename_MC)

    # write the btable file
    if not os.path.exists(btablefilename_MC):
        hardiIO.bvecsbvals2btable(bvalsfilename_MC, bvecsfilename_MC, btablefilename_MC)

    # save src file
    # hardiIO.nifti2src(niifilename_MC, btablefilename_MC, srcfilename_MC)

    end_time = time.time()

    print('PerformBaselineReferenceMotionCorrectionANTsRIGID: time elapsed = %f seconds ...' % (
            end_time - start_time))
    return




"""
---------------------------------------------------------------------------------
PerformDWIGeometricMeanReferenceMotionCorrectionMCFLIRT6DOFupdate
---------------------------------------------------------------------------------

OBJECTIVE:
    correct for motion the DTIPrep QCed sequences where slice-wise/gradient-wise denoising already
    been carried out, as such we can use standard interpolation

Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) baseline and brain mask extraction
    (4) q-space resampling of corrupted slices has been performed along with gradient-wise denoising

TODO:
    (0) fix the dimension and thickness of the given nrrd files (if any)
    (1) prepare nifti files for mcflirt
    (2) perform the always-interpolate option to correct for subject motion using mcflirt without using masked sequence
    (3) back to nrrd format, nii and src
"""


def PerformDWIGeometricMeanReferenceMotionCorrectionMCFLIRT6DOFupdate(prepDir,phan_name,
                                                                       resampling_method, nDirections=65,
                                                                       useBrainMask=True):  # false if we have severe motion (> 5deg), otherwise try to avoid background noise when doing motion correction

    start_time = time.time()
    interpMethod = 'trilinear'
    baselineIndex = 0

    # basename, phan_name = ParseFilename(nrrdfilename)

    # useBrainMask  = True # false if we have severe motion (> 5deg), otherwise try to avoid background noise when doing motion correction
    if resampling_method == None:
        nrrdfilename = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nrrd' % (
                                        nDirections, phan_name, nDirections))
        niifilename = os.path.join(prepDir,
                                   'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nii' % (
                                       nDirections, phan_name, nDirections))

        bvalsfilename = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.bvals' % (
                                         nDirections, phan_name, nDirections))
        bvecsfilename = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.bvecs' % (
                                         nDirections, phan_name, nDirections))

        nrrdfilename_MC = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF.nrrd' % (
                                           nDirections, phan_name, nDirections))
        niifilename_MC = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF.nii' % (
                                          nDirections, phan_name, nDirections))

        bvecsfilename_MC = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF.bvecs' % (
                                            nDirections, phan_name, nDirections))
        bvalsfilename_MC = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF.bvals' % (
                                            nDirections, phan_name, nDirections))
        btablefilename_MC = os.path.join(prepDir,
                                         'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF_btable.txt' % (
                                             nDirections, phan_name, nDirections))
        srcfilename_MC = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF.src.gz' % (
                                          nDirections, phan_name, nDirections))

        reportfilename = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF_QCreport.txt' % (
                                          nDirections, phan_name, nDirections))

    else:
        nrrdfilename = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.nrrd' % (
                                        nDirections, phan_name, nDirections, resampling_method.upper()))
        niifilename = os.path.join(prepDir,
                                   'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.nii' % (
                                       nDirections, phan_name, nDirections, resampling_method.upper()))

        bvalsfilename = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.bvals' % (
                                         nDirections, phan_name, nDirections,
                                         resampling_method.upper()))
        bvecsfilename = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.bvecs' % (
                                         nDirections, phan_name, nDirections,
                                         resampling_method.upper()))

        nrrdfilename_MC = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF.nrrd' % (
                                           nDirections, phan_name, nDirections,
                                           resampling_method.upper()))
        niifilename_MC = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF.nii' % (
                                          nDirections, phan_name, nDirections,
                                          resampling_method.upper()))

        bvecsfilename_MC = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF.bvecs' % (
                                            nDirections, phan_name, nDirections,
                                            resampling_method.upper()))
        bvalsfilename_MC = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF.bvals' % (
                                            nDirections, phan_name, nDirections,
                                            resampling_method.upper()))
        btablefilename_MC = os.path.join(prepDir,
                                         'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF_btable.txt' % (
                                             nDirections, phan_name, nDirections,
                                             resampling_method.upper()))
        srcfilename_MC = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF.src.gz' % (
                                          nDirections, phan_name, nDirections,
                                          resampling_method.upper()))
        reportfilename = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF_QCreport.txt' % (
                                          nDirections, phan_name, nDirections,
                                          resampling_method.upper()))

    # baseline was excluded during dtiprep, no point to process this dataset
    if os.path.exists(nrrdfilename) == False:
        return



    mcOutDir = os.path.join(prepDir, 'DWI_%ddir/DWIGEOMETRICMEAN_MCFLIRT6DOF_OUT' % (nDirections))
    if not os.path.exists(mcOutDir):
        os.mkdir(mcOutDir)

    # (1) align all DWI to the first one
    niifilename_for_mcflirt_in = os.path.join(mcOutDir, 'mcflirt_dwi_in.nii' )
    niifilename_for_mcflirt_out = os.path.join(mcOutDir, 'mcflirt_dwi_out.nii' )

    niifilename_for_mcflirt_gm_in = os.path.join(mcOutDir, 'mcflirt_gm_in.nii' )
    niifilename_for_mcflirt_gm_out = os.path.join(mcOutDir, 'mcflirt_gm_out.nii' )


    if useBrainMask: # Bet first for each DWIs for motion correction
        niimaskedfilename = hardiQCUtils.extractBrainRegionDWI(niifilename)
        niifilename = niimaskedfilename

    origData = np.asanyarray(nib.load(niifilename).dataobj)
    dwiData = origData[:, :, :, baselineIndex+1:]
    affine = nib.load(niifilename).affine

    if not os.path.exists(niifilename_for_mcflirt_gm_in):
        hardiIO.save2nii(niifilename_for_mcflirt_in, dwiData, affine)

        # now go ahead and align all dwi images to the first dwi image
        # for dwi's normalized cross correlation is better in recovering severe motion
        # note that we are not dowing any brain masking here in order not to lose brain regions
        hardiQCUtils.RunMCFLIRT(niifilename_for_mcflirt_in, niifilename_for_mcflirt_out, interpMethod,cost='normcorr')

        if not os.path.exists(niifilename_for_mcflirt_out):
            if os.path.exists(niifilename_for_mcflirt_out + '.gz'):
                niifilename_for_mcflirt_out = niifilename_for_mcflirt_out + '.gz'

        cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_dwi_out_meanvol.nii.gz' ))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_dwi_out_sigma.nii.gz' ))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_dwi_out_variance.nii.gz'  ))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_dwi_out.par'  ))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        dwiData_aligned = np.asanyarray(nib.load(niifilename_for_mcflirt_out).dataobj)
        # options_dwi     = hardiIO.removeBaselineFromOptions(options, baselineIndex)
        # nrrd.write( niifilename_for_mcflirt_out[:-4]+'.nrrd', dwiData_aligned, options_dwi)

        cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_dwi_in.nii'  ))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_dwi_out.nii.gz'  ))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_dwi_out.mat'  ))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        geomMean = np.ones(dwiData_aligned.shape[0:3])
        for ind in range(dwiData_aligned.shape[-1]):
            geomMean = geomMean * dwiData_aligned[:, :, :, ind]
        geomMean = geomMean ** (1 / dwiData_aligned.shape[-1])
        # nrrd.write( niifilename_for_mcflirt_out[:-4]+'_geomean.nrrd', geomMean)
        niiData = np.zeros((geomMean.shape[0], geomMean.shape[1], geomMean.shape[2], 2))
        niiData[:, :, :, 0] = copy.deepcopy(origData[:, :, :, baselineIndex])
        niiData[:, :, :, 1] = copy.deepcopy(geomMean)
        hardiIO.save2nii(niifilename_for_mcflirt_gm_in, niiData, affine)

    # (2) align the geometric mean to the baseline
    if not os.path.exists(niifilename_for_mcflirt_gm_out):
        if os.path.exists(niifilename_for_mcflirt_gm_out + '.gz'):
            niifilename_for_mcflirt_gm_out = niifilename_for_mcflirt_gm_out + '.gz'
        else:
            hardiQCUtils.RunMCFLIRT(niifilename_for_mcflirt_gm_in, niifilename_for_mcflirt_gm_out,interpMethod)
            cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_gm_out_meanvol.nii.gz' ))
            # os.system(cmdStr)
            subprocess.Popen(cmdStr,
                             shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

            cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_gm_out_sigma.nii.gz' ))
            # os.system(cmdStr)
            subprocess.Popen(cmdStr,
                             shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

            cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_gm_out_variance.nii.gz' ))
            # os.system(cmdStr)
            subprocess.Popen(cmdStr,
                             shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

            cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_gm_out.par'))
            # os.system(cmdStr)
            subprocess.Popen(cmdStr,
                             shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

            cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_gm_out.mat' ))
            # os.system(cmdStr)
            subprocess.Popen(cmdStr,
                             shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

    # (3) proceed as in baseline reference but using the geometric mean as the reference
    niifilename_for_mcflirt_in = os.path.join(mcOutDir,'mcflirt_in.nii' )
    niifilename_for_mcflirt_out = os.path.join(mcOutDir,'mcflirt_out.nii' )

    mcParamFolder = os.path.join(mcOutDir,'mcflirt_out.mat' )
    parfilename = os.path.join(mcOutDir,'mcflirt_out.par' )


    if not os.path.exists(niifilename_for_mcflirt_in):

        if not os.path.exists(niifilename_for_mcflirt_gm_out):
            if os.path.exists(niifilename_for_mcflirt_gm_out + '.gz'):
                niifilename_for_mcflirt_gm_out = niifilename_for_mcflirt_gm_out + '.gz'
        niiData_aligned = np.asanyarray(nib.load(niifilename_for_mcflirt_gm_out).dataobj)
        # nrrd.write( niifilename_for_mcflirt_out[:-4]+'.nrrd', niiData_aligned, options)
        geomMean_aligned = copy.deepcopy(niiData_aligned[:, :, :, 1])  # to the baseline

        niiData = np.asanyarray(nib.load(niifilename).dataobj)
        niiData[:, :, :, baselineIndex] = copy.deepcopy(geomMean_aligned)

        affine = nib.load(niifilename).affine
        # nDirections   = niiData.shape[-1]

        hardiIO.save2nii(niifilename_for_mcflirt_in, niiData, affine)

    # now go ahead and do motion correction
    dircount = len(open(bvalsfilename).readlines())
    if os.path.exists(mcParamFolder):
        if len(glob.glob(mcParamFolder + '/*')) != dircount:
            shutil.rmtree(mcParamFolder)

    if not os.path.exists(niifilename_for_mcflirt_out) and not os.path.exists(niifilename_for_mcflirt_out+'.gz'):
        # hardiQCUtils.RunMCFLIRT(niifilename_for_mcflirt_in, niifilename_for_mcflirt_out,
        #                               interpMethod, cost='normcorr')
        hardiQCUtils.RunMCFLIRT(niifilename_for_mcflirt_in, niifilename_for_mcflirt_out,
                                interpMethod)

        # nrrd.write( niifilename_for_mcflirt_out[:-4]+'.nrrd', nib.load(niifilename_for_mcflirt_out).get_data(), options)

        cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_out_meanvol.nii' ))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_out_sigma.nii' ))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

        cmdStr = 'rm -rf %s' % (os.path.join(mcOutDir,'mcflirt_out_variance.nii'))
        # os.system(cmdStr)
        subprocess.Popen(cmdStr,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().


    # quantify the current motion
    if not os.path.exists(reportfilename):
        motionQuantification = hardiQCUtils.QuantifyMotion(parfilename, mcParamFolder)
        hardiQCUtils.WriteMotionCorrectionQCReport(motionQuantification, reportfilename)

    # apply the transformation found by mcflirt on the original data to get an unmasked corrected sequence (for noise removal afterwards too)
    # this would help in maintaining the same processing pipeline of the reference motion-free sequence versus the motion corrected ones
    nointegerflag = False
    if not os.path.exists(nrrdfilename_MC):
        out_prefix = os.path.join(mcOutDir,'mcflirt' )
        correctedData = hardiQCUtils.ApplyMCFLIRT(niifilename, mcParamFolder, interpMethod, out_prefix)

        # Check if corrected data has significant digit below tenths (WILL RAISE CONVERSION PROBLEM)

        if np.sum((correctedData*10) %10) > 0.0:
            nointegerflag = True
            return nointegerflag

        # update the btable
        bvals, bvecs = hardiIO.readbvalsbvecs(bvalsfilename, bvecsfilename)

        baselineIndex = np.where(bvals == 0)[0][0]
        bvals_corrected, bvecs_corrected = hardiQCUtils.ReorientBmatrix_mcflirt(bvals, bvecs, mcParamFolder)

        # load the corrected data
        _, options = nrrd.read(nrrdfilename)

        # update the btable in the nrrd options then start the saving cycle
        options = hardiIO.updateNrrdOptions(options, bvecs_corrected)

        # save as nrrd with the save options as the original nrrd file
        nrrd.write(nrrdfilename_MC, correctedData.astype('short'), options)


    if not os.path.exists(niifilename_MC) or not os.path.exists(bvecsfilename_MC) or not os.path.exists(bvalsfilename_MC):
        # convert to nifiti + save bvecs and bvals
        hardiIO.convertToNIFTI(nrrdfilename_MC, niifilename_MC, bvecsfilename_MC, bvalsfilename_MC)

    # write the btable file
    if not os.path.exists(btablefilename_MC):
        hardiIO.bvecsbvals2btable(bvalsfilename_MC, bvecsfilename_MC, btablefilename_MC)

    # save src file
    # hardiIO.nifti2src(niifilename_MC, btablefilename_MC, srcfilename_MC)

    end_time = time.time()

    print('PerformDWIGeometricMeanReferenceMotionCorrectionMCFLIRT6DOF: time elapsed = %f seconds ...' % (
            end_time - start_time))
    return nointegerflag



def BrainMaskDWIBaselineReferenceMotionCorrectedDWIupdate(prepDir, phan_name, resampling_method,
                                                      nDirections=65, check_btable = True):
    start_time = time.time()

    # basename, phan_name = ParseFilename(nrrdfilename)

    niiBrainMaskFilename = os.path.join(prepDir,
                                        'baseline/%s_baseline_brain_mask.nii' % (phan_name))

    if resampling_method == None:
        nrrdfilename_MC = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID.nrrd' % (
                                           nDirections, phan_name, nDirections))

        nrrdfilename_MC_masked = os.path.join(prepDir,
                                              'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID_masked.nrrd' % (
                                                  nDirections, phan_name, nDirections))
        niifilename_MC_masked = os.path.join(prepDir,
                                             'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID_masked.nii' % (
                                                 nDirections, phan_name, nDirections))

        bvecsfilename_MC_masked = os.path.join(prepDir,
                                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID_masked.bvecs' % (
                                                   nDirections, phan_name, nDirections))
        bvalsfilename_MC_masked = os.path.join(prepDir,
                                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID_masked.bvals' % (
                                                   nDirections, phan_name, nDirections))
        btablefilename_MC_masked = os.path.join(prepDir,
                                                'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID_masked_btable.txt' % (
                                                    nDirections, phan_name, nDirections))
        srcfilename_MC_masked = os.path.join(prepDir,
                                             'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Baseline_ANTsRIGID_masked.src.gz' % (
                                                 nDirections, phan_name, nDirections))

    else:
        nrrdfilename_MC = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID.nrrd' % (
                                           nDirections, phan_name, nDirections,
                                           resampling_method.upper()))
        niifilename_MC = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID.nii' % (
                                           nDirections, phan_name, nDirections,
                                           resampling_method.upper()))

        bvecsfilename_MC = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID.bvecs' % (
                                            nDirections, phan_name, nDirections,
                                            resampling_method.upper()))
        bvalsfilename_MC = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID.bvals' % (
                                            nDirections, phan_name, nDirections,
                                            resampling_method.upper()))
        btablefilename_MC = os.path.join(prepDir,
                                         'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID_btable.txt' % (
                                             nDirections, phan_name, nDirections,
                                             resampling_method.upper()))

        nrrdfilename_MC_masked = os.path.join(prepDir,
                                              'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID_masked.nrrd' % (
                                                  nDirections, phan_name, nDirections,
                                                  resampling_method.upper()))
        niifilename_MC_masked = os.path.join(prepDir,
                                             'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID_masked.nii' % (
                                                 nDirections, phan_name, nDirections,
                                                 resampling_method.upper()))

        bvecsfilename_MC_masked = os.path.join(prepDir,
                                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID_masked.bvecs' % (
                                                   nDirections, phan_name, nDirections,
                                                   resampling_method.upper()))
        bvalsfilename_MC_masked = os.path.join(prepDir,
                                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID_masked.bvals' % (
                                                   nDirections, phan_name, nDirections,
                                                   resampling_method.upper()))
        btablefilename_MC_masked = os.path.join(prepDir,
                                                'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID_masked_btable.txt' % (
                                                    nDirections, phan_name, nDirections,
                                                    resampling_method.upper()))
        srcfilename_MC_masked = os.path.join(prepDir,
                                             'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_Baseline_ANTsRIGID_masked.src.gz' % (
                                                 nDirections, phan_name, nDirections,
                                                 resampling_method.upper()))

    if os.path.exists(nrrdfilename_MC) == False:
        return

    if not os.path.exists(niiBrainMaskFilename):
        if os.path.exists(niiBrainMaskFilename+'.gz'):
            niiBrainMaskFilename = niiBrainMaskFilename+'.gz'

    if not os.path.exists(nrrdfilename_MC_masked):
        brainMask = np.asanyarray(nib.load(niiBrainMaskFilename).dataobj)
        correctedData, options = nrrd.read(nrrdfilename_MC)
        correctedDataMasked = hardiQCUtils.brainMasking(correctedData, brainMask)
        # save as nrrd with the save options as the original nrrd file
        nrrd.write(nrrdfilename_MC_masked, correctedDataMasked, options) # No longer short
        affine = nib.load(niifilename_MC).affine
        hardiIO.save2nii(niifilename_MC_masked, correctedDataMasked, affine=affine)  # No longer short
        # add Fix header (qform / sform) ?
        # !!!qform format index change when saved nifti https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/qsform.html
        shutil.copyfile(bvecsfilename_MC, bvecsfilename_MC_masked)
        shutil.copyfile(bvalsfilename_MC, bvalsfilename_MC_masked)
        shutil.copyfile(btablefilename_MC, btablefilename_MC_masked)


    # if not os.path.exists(niifilename_MC_masked) or not os.path.exists(bvecsfilename_MC_masked) or not os.path.exists(bvalsfilename_MC_masked):
    #     # convert to nifiti + save bvecs and bvals
    #     hardiIO.convertToNIFTI(nrrdfilename_MC_masked, niifilename_MC_masked, bvecsfilename_MC_masked,
    #                            bvalsfilename_MC_masked)

    # write the btable file
    if not os.path.exists(btablefilename_MC_masked):
        hardiIO.bvecsbvals2btable(bvalsfilename_MC_masked, bvecsfilename_MC_masked,
                                  btablefilename_MC_masked)

    # save src file
    if not os.path.exists(srcfilename_MC_masked):
        hardiIO.nifti2src(niifilename_MC_masked, btablefilename_MC_masked, srcfilename_MC_masked)


    if check_btable:
        recfilename = glob.glob(srcfilename_MC_masked+'.*')
        # if len(recfilename) == 0:
        dsistudiopath = '/media/HDD2T/utils/dsistudio/dsi-studio-2018/dsi_studio_64/dsi_studio'
        # method 0:DSI, 1:DTI, 2:Funk-Randon QBI, 3:Spherical Harmonic QBI, 4:GQI 6: Convert to HARDI 7:QSDR.
        cmdcheckbtable ='{} --action=rec --source={} --method=1 --check_btable={}'.format(dsistudiopath,
                                                                                  srcfilename_MC_masked, check_btable)
        subprocess.Popen(cmdcheckbtable,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().


    end_time = time.time()

    print('BrainMaskBaselineMotionCorrectedDWI: time elapsed = %f seconds ...' % (
            end_time - start_time))



"""
---------------------------------------------------------------------------------
BrainMaskDWIGeometricMeanMotionCorrectedDWIupdate
---------------------------------------------------------------------------------

OBJECTIVE:
    brain mask the motion corrected datasets

Assumptions:
    (1) DTIPrep (without motion) has been performed
    (2) WithinGradientMotionQC has been performed
    (3) brain mask has been extracted
    (4) q-space resampling of corrupted slices has been performed along with gradient-wise denoising
    (5) motion correction has been performed

TODO:
    (1) brain mask the motion corrected data
"""



def BrainMaskDWIGeometricMeanMotionCorrectedDWIupdate(prepDir, phan_name, resampling_method,
                                                      nDirections=65, check_btable = True):
    start_time = time.time()

    # basename, phan_name = ParseFilename(nrrdfilename)

    niiBrainMaskFilename = os.path.join(prepDir,
                                        'baseline/%s_baseline_brain_mask.nii' % (phan_name))

    if resampling_method == None:
        nrrdfilename_MC = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF.nrrd' % (
                                           nDirections, phan_name, nDirections))

        nrrdfilename_MC_masked = os.path.join(prepDir,
                                              'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF_masked.nrrd' % (
                                                  nDirections, phan_name, nDirections))
        niifilename_MC_masked = os.path.join(prepDir,
                                             'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF_masked.nii' % (
                                                 nDirections, phan_name, nDirections))

        bvecsfilename_MC_masked = os.path.join(prepDir,
                                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF_masked.bvecs' % (
                                                   nDirections, phan_name, nDirections))
        bvalsfilename_MC_masked = os.path.join(prepDir,
                                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF_masked.bvals' % (
                                                   nDirections, phan_name, nDirections))
        btablefilename_MC_masked = os.path.join(prepDir,
                                                'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF_masked_btable.txt' % (
                                                    nDirections, phan_name, nDirections))
        srcfilename_MC_masked = os.path.join(prepDir,
                                             'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_DWIGeometricMean_MCFLIRT6DOF_masked.src.gz' % (
                                                 nDirections, phan_name, nDirections))

    else:
        nrrdfilename_MC = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF.nrrd' % (
                                           nDirections, phan_name, nDirections,
                                           resampling_method.upper()))

        nrrdfilename_MC_masked = os.path.join(prepDir,
                                              'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF_masked.nrrd' % (
                                                  nDirections, phan_name, nDirections,
                                                  resampling_method.upper()))
        niifilename_MC_masked = os.path.join(prepDir,
                                             'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF_masked.nii' % (
                                                 nDirections, phan_name, nDirections,
                                                 resampling_method.upper()))

        bvecsfilename_MC_masked = os.path.join(prepDir,
                                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF_masked.bvecs' % (
                                                   nDirections, phan_name, nDirections,
                                                   resampling_method.upper()))
        bvalsfilename_MC_masked = os.path.join(prepDir,
                                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF_masked.bvals' % (
                                                   nDirections, phan_name, nDirections,
                                                   resampling_method.upper()))
        btablefilename_MC_masked = os.path.join(prepDir,
                                                'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF_masked_btable.txt' % (
                                                    nDirections, phan_name, nDirections,
                                                    resampling_method.upper()))
        srcfilename_MC_masked = os.path.join(prepDir,
                                             'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_DWIGeometricMean_MCFLIRT6DOF_masked.src.gz' % (
                                                 nDirections, phan_name, nDirections,
                                                 resampling_method.upper()))

    if os.path.exists(nrrdfilename_MC) == False:
        return

    if not os.path.exists(niiBrainMaskFilename):
        if os.path.exists(niiBrainMaskFilename+'.gz'):
            niiBrainMaskFilename = niiBrainMaskFilename+'.gz'

    if not os.path.exists(nrrdfilename_MC_masked):
        brainMask = np.asanyarray(nib.load(niiBrainMaskFilename).dataobj)
        correctedData, options = nrrd.read(nrrdfilename_MC)
        correctedDataMasked = hardiQCUtils.brainMasking(correctedData, brainMask)
        # save as nrrd with the save options as the original nrrd file
        nrrd.write(nrrdfilename_MC_masked, correctedDataMasked.astype('short'), options)

    if not os.path.exists(niifilename_MC_masked) or not os.path.exists(bvecsfilename_MC_masked) or not os.path.exists(bvalsfilename_MC_masked):
        # convert to nifiti + save bvecs and bvals
        hardiIO.convertToNIFTI(nrrdfilename_MC_masked, niifilename_MC_masked, bvecsfilename_MC_masked,
                               bvalsfilename_MC_masked)

    # write the btable file
    if not os.path.exists(btablefilename_MC_masked):
        hardiIO.bvecsbvals2btable(bvalsfilename_MC_masked, bvecsfilename_MC_masked,
                                  btablefilename_MC_masked)

    # save src file
    if not os.path.exists(srcfilename_MC_masked):
        hardiIO.nifti2src(niifilename_MC_masked, btablefilename_MC_masked, srcfilename_MC_masked)


    if check_btable:
        recfilename = glob.glob(srcfilename_MC_masked+'.*')
        # if len(recfilename) == 0:
        dsistudiopath = '/media/HDD2T/utils/dsistudio/dsi-studio-2018/dsi_studio_64/dsi_studio'
        # method 0:DSI, 1:DTI, 2:Funk-Randon QBI, 3:Spherical Harmonic QBI, 4:GQI 6: Convert to HARDI 7:QSDR.
        cmdcheckbtable ='{} --action=rec --source={} --method=1 --check_btable={}'.format(dsistudiopath,
                                                                                  srcfilename_MC_masked, check_btable)
        subprocess.Popen(cmdcheckbtable,
                         shell=True).wait()  # subprocess.Popen() is strict superset of os.system().


    end_time = time.time()

    print('BrainMaskDWIGeometricMeanMotionCorrectedDWI: time elapsed = %f seconds ...' % (
            end_time - start_time))


"""
---------------------------------------------------------------------------------
GradientWiseDenoise 
---------------------------------------------------------------------------------

OBJECTIVE:
    Gradient-wise denoising
"""


def GradientWiseDenoise(prepDir, phan_name, resampling_method, nDirections ,
                        EstimationRadius=11, FilteringRadius=11, minVoxelsEstimation=3,
                        minVoxelsFiltering=3):
    start_time = time.time()

    EstimationRadius_x = EstimationRadius
    EstimationRadius_y = EstimationRadius
    EstimationRadius_z = 0
    FilteringRadius_x = FilteringRadius
    FilteringRadius_y = FilteringRadius
    FilteringRadius_z = 0
    NoOfIterations = 1
    # minVoxelsFiltering        = 7
    # minVoxelsEstimation       = 7
    histRes = 2
    minStd = 0
    maxStd = 100
    absVal = 0

    # basename, phan_name = ParseFilename(nrrdfilename,nDirections)

    if resampling_method == None:
        nrrdfilename = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nrrd' % (
                                        nDirections, phan_name, nDirections))
        nrrdfilename_qc = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE.nrrd' % (
                                           nDirections, phan_name, nDirections))
        niifilename_qc = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE.nii' % (
                                          nDirections, phan_name, nDirections))

        bvecsfilename_qc = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE.bvecs' % (
                                            nDirections, phan_name, nDirections))
        bvalsfilename_qc = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE.bvals' % (
                                            nDirections, phan_name, nDirections))
        btablefilename_qc = os.path.join(prepDir,
                                         'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE_btable.txt' % (
                                             nDirections, phan_name, nDirections))
        srcfilename_qc = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_LMMSE.src.gz' % (
                                          nDirections, phan_name, nDirections))
    else:
        nrrdfilename = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed.nrrd' % (
                                        nDirections, phan_name, nDirections, resampling_method.upper()))
        nrrdfilename_qc = os.path.join(prepDir,
                                       'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE.nrrd' % (
                                           nDirections, phan_name, nDirections,
                                           resampling_method.upper()))
        niifilename_qc = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE.nii' % (
                                          nDirections, phan_name, nDirections,
                                          resampling_method.upper()))

        bvecsfilename_qc = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE.bvecs' % (
                                            nDirections, phan_name, nDirections,
                                            resampling_method.upper()))
        bvalsfilename_qc = os.path.join(prepDir,
                                        'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE.bvals' % (
                                            nDirections, phan_name, nDirections,
                                            resampling_method.upper()))
        btablefilename_qc = os.path.join(prepDir,
                                         'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE_btable.txt' % (
                                             nDirections, phan_name, nDirections,
                                             resampling_method.upper()))
        srcfilename_qc = os.path.join(prepDir,
                                      'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_QCed_LMMSE.src.gz' % (
                                          nDirections, phan_name, nDirections,
                                          resampling_method.upper()))

    if EstimationRadius >= 1 and FilteringRadius >= 1:
        cmdStr = 'DWIRicianLMMSE %s %s %f %f %f %f %f %f %f %f %f %f %f %f %f' % (
            nrrdfilename, nrrdfilename_qc, FilteringRadius_x, FilteringRadius_y, FilteringRadius_z,
            EstimationRadius_x, EstimationRadius_y, EstimationRadius_z, NoOfIterations,
            minVoxelsFiltering, minVoxelsEstimation, histRes, minStd, maxStd, absVal)
    else:
        cmdStr = 'cp -Rvp %s %s' % (nrrdfilename, nrrdfilename_qc)
    # os.system(cmdStr)
    subprocess.Popen(cmdStr,
                     shell=True).wait()  # subprocess.Popen() is strict superset of os.system().

    # fix the nrrd file (thickness and directions are the last dimension)
    hardiIO.fixNRRDfile(nrrdfilename_qc)

    # convert to nifiti + save bvecs and bvals
    hardiIO.convertToNIFTI(nrrdfilename_qc, niifilename_qc, bvecsfilename_qc, bvalsfilename_qc)

    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename_qc, bvecsfilename_qc, btablefilename_qc)

    # save src file
    hardiIO.nifti2src(niifilename_qc, btablefilename_qc, srcfilename_qc)

    end_time = time.time()

    print
    ('GradientWiseDenoise: time elapsed = %f seconds ...' % (end_time - start_time))

def PerformResampleCorruptedSlicesInQspaceOLD(nrrdfilename, phan_name, prepDir, resampling_method='shore',
                                              nDirections=65):
    start_time = time.time()

    #    basename, phan_name = ParseFilename(nrrdfilename)
    basename, phan_name = ParseFilename(nrrdfilename)

    nrrdfilename_orig = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.nrrd' % (
        nDirections, phan_name, nDirections))
    niifilename_orig = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.nii' % (
        nDirections, phan_name, nDirections))
    bvecsfilename_orig = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.bvecs' % (
        nDirections, phan_name, nDirections))
    bvalsfilename_orig = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.bvals' % (
        nDirections, phan_name, nDirections))
    btablefilename_orig = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_btable.txt' % (
        nDirections, phan_name, nDirections))

    nrrdfilename = os.path.join(prepDir,
                                'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nrrd' % (
                                    nDirections, phan_name, nDirections))
    niifilename = os.path.join(prepDir,
                               'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.nii' % (
                                   nDirections, phan_name, nDirections))
    bvecsfilename = os.path.join(prepDir,
                                 'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.bvecs' % (
                                     nDirections, phan_name, nDirections))
    bvalsfilename = os.path.join(prepDir,
                                 'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.bvals' % (
                                     nDirections, phan_name, nDirections))
    btablefilename = os.path.join(prepDir,
                                  'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_btable.txt' % (
                                      nDirections, phan_name, nDirections))

    nrrdfilename_rq = os.path.join(prepDir,
                                   'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s.nrrd' % (
                                       nDirections, phan_name, nDirections, resampling_method.upper()))
    niifilename_rq = os.path.join(prepDir,
                                  'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s.nii' % (
                                      nDirections, phan_name, nDirections, resampling_method.upper()))
    bvecsfilename_rq = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s.bvecs' % (
                                        nDirections, phan_name, nDirections, resampling_method.upper()))
    bvalsfilename_rq = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s.bvals' % (
                                        nDirections, phan_name, nDirections, resampling_method.upper()))
    btablefilename_rq = os.path.join(prepDir,
                                     'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s_btable.txt' % (
                                         nDirections, phan_name, nDirections,
                                         resampling_method.upper()))
    srcfilename_rq = os.path.join(prepDir,
                                  'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_Resample%s.src.gz' % (
                                      nDirections, phan_name, nDirections, resampling_method.upper()))

    # (1) reconstruct the dODF based on regularized Qball based on the sequence from the previous QC steps

    diffusionData, bvalue, gradientDirections, baselineIndex, options = hardiIO.readHARDI(
        nrrdfilename)
    bvals, bvecs = hardiIO.readbtable(btablefilename)
    gtable = gradient_table(bvals, bvecs)

    # print
    # "============================="
    # print
    # "Baseline Index"
    # print
    # baselineIndex
    # print
    # "============================="

    if baselineIndex < 0:  # baseline was excluded during dtiprep, no point to process this dataset
        return

    # (2) get which slices in which gradients are needed to be resampled
    diffusionData_orig, bvalue_orig, gradientDirections_orig, baselineIndex_orig, options_orig = hardiIO.readHARDI(
        nrrdfilename_orig)

    bvals_orig, bvecs_orig = hardiIO.readbtable(btablefilename_orig)
    gtable_orig = gradient_table(bvals_orig, bvecs_orig)

    gradientDirections_orig = np.squeeze(gradientDirections_orig)
    rows, cols, nSlices, nDirections_ = diffusionData_orig.shape

    # get the gradients and slices which suffer from slice-wise artifacts
    toBeResampled = np.zeros((nDirections, nSlices))

    # from DTIPrep - slice-wise intensity artifacts
    excluded = list()
    qcreportfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_DTIPrepQCReport.txt' % (
        nDirections, phan_name, nDirections))

    # print
    # "============================="
    # print
    # "qcreportfilename:"
    # print
    # qcreportfilename
    # print
    # "============================="

    fid = open(qcreportfilename, 'r')
    for line in fid:
        line = line.strip()
        if line.find('Slice-wise Check Artifacts:') >= 0:
            break
    for line in fid:
        line = line.strip().split('\t')
        if line[0].find('whole') < 0:
            continue
        d = int(line[1])
        s = int(line[2])
        toBeResampled[d, s] = 1
        excluded.append(d)
    fid.close()

    print("=============================")
    print("excluded data shape")
    print(excluded)
    # excluded
    print("=============================")

    #    # from DTIPrep - interlace-wise intensity artifacts - aparently not excluded via dtiprep
    #    qcreportfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_DTIPrepQCReport.txt' % (nDirections, phan_name, nDirections))
    #    fid = open(qcreportfilename, 'r')
    #    for line in fid:
    #        line = line.strip()
    #        if line.find('Interlace-wise Check Artifacts:') >= 0:
    #            break
    #    for line in fid:
    #        if line.find('Gradient#') >= 0:
    #            continue
    #        line = line.strip().split('\t')
    #        d = int(line[0])
    #        for s in range(0,nSlices):
    #            toBeResampled[d,s] = 1
    #        excluded.append(d)
    #    fid.close()
    excluded = np.unique(np.array(excluded))

    # from DTIprep - interlace artifacts
    # see if  there are any other directions that were excluded due to interslice artifacts
    qcreportfilename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_QCReport.txt' % (
        nDirections, phan_name, nDirections))
    dirIndex = dict()  # mapping that maps the direction index in dtiprep output to the original index
    fid = open(qcreportfilename, 'r')
    included = np.zeros((nDirections,))
    for line in fid:
        if line.find('QCIndex') >= 0 and line.find('Included Gradients:') >= 0:
            line = line.strip().split(' ')
            d = int(line[2])
            included[d] = 1
            dirIndex[int(line[4])] = int(line[2])
    fid.close()
    included[baselineIndex_orig] = 1

    for d in range(nDirections):
        if included[d] == 0:
            if len(np.where(excluded == d)[0]) == 0:  # was not excluded due to slice artifacts
                for s in range(nSlices):
                    toBeResampled[d, s] = 1

    # from within-gradient motion QC
    qcreportfilename = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCreport.txt' % (
                                        nDirections, phan_name, nDirections))
    fid = open(qcreportfilename, 'r')
    for line in fid:
        if line.find('Excluded') >= 0:
            line = line.strip().split()
            dirind = int(line[0])
            nslices = int(line[2])
            for i in range(nslices):
                s = int(line[3 + i])
                d = dirIndex[dirind]
                toBeResampled[d, s] = 1
    fid.close()

    qcreportfilename = os.path.join(prepDir,
                                    'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotion_ResamplingReport.txt' % (
                                        nDirections, phan_name, nDirections))
    fid = open(qcreportfilename, 'w')
    fid.write(' ,')
    for slice_number in range(nSlices):
        fid.write('slice # %d ,' % (slice_number))
    fid.write('\n')
    direction_number = -1
    for row in toBeResampled:
        direction_number = direction_number + 1
        fid.write('gradient direction # %d, ' % (direction_number))
        fid.write(','.join(map(str, row)) + '\n')
    fid.close()

    # (3) start resampling
    diffusionData_resampled = copy.deepcopy(diffusionData_orig)
    for kk in range(nSlices):
        if np.sum(toBeResampled[:,
                  kk]) == 0:  # no need for resampling this slice along all directions
            continue

        # for the current slice, get all the valid directions to be used to estimate the slice-wise diffusion model
        cur_bvecs = list()
        cur_bvals = list()
        valid_indices = list()
        for ind, gradDir in enumerate(gradientDirections_orig):
            if toBeResampled[
                ind, kk] == 1:  # this direction is corrupted, don't include it in the estimation process
                continue
            cur_bvals.append(bvals_orig[ind])
            cur_bvecs.append(bvecs_orig[ind, :])
            valid_indices.append(ind)
        cur_bvals = np.array(cur_bvals)
        cur_bvecs = np.array(cur_bvecs)

        cur_gtable = gradient_table(cur_bvals, cur_bvecs)
        curDiffusionData = copy.deepcopy(diffusionData_orig[:, :, kk, valid_indices])

        if resampling_method == 'qbi':
            qball_model = drecon.QballModel(cur_gtable, sh_order=6, smooth=0.006)
            sphHarmFit = qball_model.fit(curDiffusionData)
            shm_coeff = sphHarmFit.shm_coeff
        else:  # SHORE
            radial_order = 6
            zeta = 700
            lambdaN = 1e-8
            lambdaL = 1e-8
            asm = ShoreModel(cur_gtable, radial_order=radial_order,
                             zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
            asmfit = asm.fit(curDiffusionData)
            shore_coeff = asmfit.shore_coeff
            shoreMatrix = shore_matrix(radial_order, zeta, gtable_orig)
            # print kk

        for ind, gradDir in enumerate(gradientDirections_orig):

            if ind == baselineIndex_orig:
                continue

            if toBeResampled[ind, kk] == 0:  # only resample the corrupted slices
                continue

            for ii in range(rows):
                for jj in range(cols):

                    print(
                        'Resampling ii = %d, jj = %d, slice = %d, dir = %d ...' % (ii, jj, kk, ind))
                    S0 = diffusionData_orig[ii, jj, kk, baselineIndex_orig]
                    if resampling_method == 'qbi':
                        shCoeffs = shm_coeff[ii, jj, :]

                        r, theta, phi = drecon.cart2sphere(gradDir[0], gradDir[1], gradDir[2])

                        shBasis, m, n = drecon.real_sym_sh_basis(6, phi, theta)
                        shBasis = shBasis.flatten()

                        diffusionData_resampled[ii, jj, kk, ind] = S0 * np.dot(shBasis, shCoeffs)
                    else:  # SHORE
                        shoreCoeff = shore_coeff[ii, jj, :]
                        if np.any(np.isnan(shoreCoeff)):
                            diffusionData_resampled[ii, jj, kk, ind] = 0.0
                        else:
                            diffusionData_resampled[ii, jj, kk, ind] = S0 * np.dot(
                                shoreMatrix[ind, :], shoreCoeff)

    nrrd.write(nrrdfilename_rq, diffusionData_resampled, options_orig)

    # (4) redo the denoising part (Gradient wise) of DTIPrep
    # cmdStr = 'DTIPrep --DWINrrdFile %s --xmlProtocol %s --check --outputFolder %s' % (nrrdfilename_rq, xmlfilename, os.path.join(curDir, 'DWI_65dir'))
    # cmdStr = 'DWIRicianLMMSE %s %s %f %f %f %f %f %f %f %f %f %f %f %f %f' % (nrrdfilename_rq, nrrdfilename_rq, FilteringRadius_x, FilteringRadius_y, FilteringRadius_z, EstimationRadius_x, EstimationRadius_y, EstimationRadius_z, NoOfIterations, minVoxelsFiltering, minVoxelsEstimation, histRes, minStd, maxStd, absVal)
    # os.system(cmdStr)

    # fix the nrrd file (thickness and directions are the last dimension)
    hardiIO.fixNRRDfile(nrrdfilename_rq)

    # convert to nifiti + save bvecs and bvals
    hardiIO.convertToNIFTI(nrrdfilename_rq, niifilename_rq, bvecsfilename_rq, bvalsfilename_rq)

    # write the btable file
    hardiIO.bvecsbvals2btable(bvalsfilename_rq, bvecsfilename_rq, btablefilename_rq)

    # save src file
    hardiIO.nifti2src(niifilename_rq, btablefilename_rq, srcfilename_rq)

    end_time = time.time()

    print(
        'PerformResampleCorruptedSlicesInQspace: time elapsed = %f seconds ...' % (
                end_time - start_time))