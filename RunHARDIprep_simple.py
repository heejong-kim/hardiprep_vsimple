# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:30:57 2015
@author: Shireen Elhabian 

Modified on Tue Feb 5 2019
@author: Heejong Kim & Edouard Mior

Modified as hardi_vsimple Dec 21 2020
@author: Heejong Kim

"""

from __future__ import division
import os
import glob
import numpy as np
import ntpath
import time
import sys

import subprocess
from dipy.io.gradients import read_bvals_bvecs

import nrrd
import hardi.io as hardiIO
import hardi.qc as hardiQC

import argparse
from joblib import Parallel, delayed

def hardiprep_vsimple(nrrdfilename, subjname, outDir, nDirections, prepDir_suffix, resampling_method, xmlfilename, useBrainMask):

    if useBrainMask > 0:
        useBrainMask = True
    else:
        useBrainMask = False

    if not os.path.exists(nrrdfilename):
        sys.exit("MISSING INPUT NRRD FILE")
    else:

        print("===================================")
        print('Working on %s' % (nrrdfilename))
        print("===================================")

        if not os.path.exists(outDir):
            os.makedirs(outDir)

        # mincs dcm2mnc -dname  -stdin -clobber /tmp/908.1.all.q/TarLoad-23-27-k7hdQD\\n')

        """
        ---------------------------------------------------------------------------------
        STEP Zero
        ---------------------------------------------------------------------------------

        (1) create hardi QC directories and copy the original nrrd
        (2) fix the dimension and thickness of the given nrrd files
        (3) convert them to nifti and generate the btable for dsi_studio visualization

        """
        prepDir, fixednrrdfilename, bvalbveczeroflag = hardiQC.PrepareQCsession(nrrdfilename, outDir, subjname,
                                                                                prepDir_suffix, nDirections)
        if bvalbveczeroflag:
            sys.exit("NO BVALUES BVECTORS or LESS THAN 6 GRADIENTS IN NRRD FILE")

        """
        ---------------------------------------------------------------------------------
        STEP ONE
        ---------------------------------------------------------------------------------

        OBJECTIVE:
            dtiprep without motion correction
            check the quality of the individual directions, e.g. missing slices, intensity artifacts, Venetian blind
        """

        baselineexcludeflag = hardiQC.RunDTIPrepStage(fixednrrdfilename, prepDir, subjname, xmlfilename, nDirections)
        if baselineexcludeflag:
            sys.exit("NO BASELINE")

        """
        # ---------------------------------------------------------------------------------
        # STEP TWO
        # ---------------------------------------------------------------------------------
        # 
        # OBJECTIVE:
        #     quantify fast bulk motion within each gradient to exclude those having intra-scan
        #     motion (see Benner et al (2011) - Diffusion imaging with prospective motion correction and reacquisition)
        # 
        #     here we have zero tolerance, any gradient having at least one slice with signal drop out will be excluded
        # """

        hardiQC.PerformWithinGradientMotionQC(fixednrrdfilename, prepDir, subjname, nDirections)

        """
        ---------------------------------------------------------------------------------
        STEP TRHEE
        ---------------------------------------------------------------------------------
        Assumptions:
            (1) WithinGradientMotionQC has been performed
            (2) DTIPrep (without motion) has been performed

        """
        hardiQC.ExtractBaselineAndBrainMask(fixednrrdfilename, prepDir, subjname, nDirections)
        """
        ---------------------------------------------------------------------------------
        STEP FOUR
        ---------------------------------------------------------------------------------

        OBJECTIVE:
            resample the corrupted slices (detected via DTIPrep and within-gradient motion) Qc
            in the q-space plus gradient-wise denoising
        """
        lessgradientflag = hardiQC.PerformResampleCorruptedSlicesInQspace(fixednrrdfilename, prepDir, subjname, resampling_method,
                                                       nDirections)

        if lessgradientflag:
            sys.exit("LESS THAN 6 GRADIENTS INPUT FOR RESAMPLING STEP")

        """
        # ---------------------------------------------------------------------------------
        # STEP FIVE
        # ---------------------------------------------------------------------------------
        #
        # OBJECTIVE:
        #     Exclusion of any direction that has artifacts not curable by the resampling stage
        # """
        hardiQC.RerunDTIPrepStage(fixednrrdfilename, prepDir, subjname, xmlfilename, resampling_method,
                                  nDirections)

        """
        ---------------------------------------------------------------------------------
        STEP SIX
        ---------------------------------------------------------------------------------

        OBJECTIVE:
            correct for motion the DTIPrep QCed sequences where slice-wise/gradient-wise denoising already
            been carried out, as such we can use standard interpolation

        """

        nointegerflag = hardiQC.PerformDWIGeometricMeanReferenceMotionCorrectionMCFLIRT6DOFupdate(fixednrrdfilename,
                                                                                                  prepDir,
                                                                                                  subjname,
                                                                                                  resampling_method,
                                                                                                  nDirections,
                                                                                                  False)
        if nointegerflag:
            sys.exit("CHECK MOTION CORRECTED IMAGE INTENSITY: HAS FLOAT")


        """
        ---------------------------------------------------------------------------------
        STEP SEVEN
        ---------------------------------------------------------------------------------

        OBJECTIVE:
            brain mask the motion corrected datasets
        """
        hardiQC.BrainMaskDWIGeometricMeanMotionCorrectedDWIupdate(fixednrrdfilename, prepDir, subjname,
                                                                  resampling_method, nDirections)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrrdfilename',
                        help='nrrd filename for the hardi dataset to be processed')
    parser.add_argument('--subjectname',
                        help='subject name for directory name')
    parser.add_argument('--outDir', help='where to put the processing output')
    parser.add_argument('--nDirections', help='number of directions in the sequence')
    parser.add_argument('--prepDir_suffix',
                        help='suffix for the prep dir HARDIPREP_QC_<suffix>, can be used to have different qc pipelines (different parameters) being applied to the same dataset or to identify the qc of individual datasets')
    parser.add_argument('--resampling_method',
                        help='resampling method to be used shore or qbi (shore is recommended)')
    parser.add_argument('--xmlfilename', help='dtiprep parameter file')
    parser.add_argument('--useBrainMask',
                        help='whether to use brain masking while doing motion correction, need to be disabled in case of severe motion as in infants datasets')

    args = parser.parse_args()

    nrrdfilename = args.nrrdfilename.strip()
    subjname = args.subjectname.strip()
    outDir = args.outDir.strip()
    nDirections = int(args.nDirections.strip())
    prepDir_suffix = args.prepDir_suffix.strip()
    resampling_method = args.resampling_method.strip()
    xmlfilename = args.xmlfilename.strip()
    useBrainMask = int(args.useBrainMask.strip())


    if useBrainMask > 0:
        useBrainMask = True
    else:
        useBrainMask = False

    if not os.path.exists(nrrdfilename):
        sys.exit("MISSING INPUT NRRD FILE")
    else:

        print("===================================")
        print('Working on %s' % (nrrdfilename))
        print("===================================")

        if not os.path.exists(outDir):
            os.makedirs(outDir)

        # mincs dcm2mnc -dname  -stdin -clobber /tmp/908.1.all.q/TarLoad-23-27-k7hdQD\\n')

        """
        ---------------------------------------------------------------------------------
        STEP Zero
        ---------------------------------------------------------------------------------

        (1) create hardi QC directories and copy the original nrrd
        (2) fix the dimension and thickness of the given nrrd files
        (3) convert them to nifti and generate the btable for dsi_studio visualization

        """
        prepDir, fixednrrdfilename, bvalbveczeroflag = hardiQC.PrepareQCsession(nrrdfilename, outDir, subjname,
                                                                                prepDir_suffix, nDirections)
        if bvalbveczeroflag:
            sys.exit("NO BVALUES BVECTORS or LESS THAN 6 GRADIENTS IN NRRD FILE")

        """
        ---------------------------------------------------------------------------------
        STEP ONE
        ---------------------------------------------------------------------------------

        OBJECTIVE:
            dtiprep without motion correction
            check the quality of the individual directions, e.g. missing slices, intensity artifacts, Venetian blind
        """

        baselineexcludeflag = hardiQC.RunDTIPrepStage(fixednrrdfilename, prepDir, subjname, xmlfilename, nDirections)
        if baselineexcludeflag:
            sys.exit("NO BASELINE")

        """
        # ---------------------------------------------------------------------------------
        # STEP TWO
        # ---------------------------------------------------------------------------------
        # 
        # OBJECTIVE:
        #     quantify fast bulk motion within each gradient to exclude those having intra-scan
        #     motion (see Benner et al (2011) - Diffusion imaging with prospective motion correction and reacquisition)
        # 
        #     here we have zero tolerance, any gradient having at least one slice with signal drop out will be excluded
        # """

        hardiQC.PerformWithinGradientMotionQC(fixednrrdfilename, prepDir, subjname, nDirections)

        """
        ---------------------------------------------------------------------------------
        STEP TRHEE
        ---------------------------------------------------------------------------------
        Assumptions:
            (1) WithinGradientMotionQC has been performed
            (2) DTIPrep (without motion) has been performed

        """
        hardiQC.ExtractBaselineAndBrainMask(fixednrrdfilename, prepDir, subjname, nDirections)
        """
        ---------------------------------------------------------------------------------
        STEP FOUR
        ---------------------------------------------------------------------------------

        OBJECTIVE:
            resample the corrupted slices (detected via DTIPrep and within-gradient motion) Qc
            in the q-space plus gradient-wise denoising
        """
        lessgradientflag = hardiQC.PerformResampleCorruptedSlicesInQspace(fixednrrdfilename, prepDir, subjname, resampling_method,
                                                       nDirections)

        if lessgradientflag:
            sys.exit("LESS THAN 6 GRADIENTS INPUT FOR RESAMPLING STEP")
        """
        # ---------------------------------------------------------------------------------
        # STEP FIVE
        # ---------------------------------------------------------------------------------
        #
        # OBJECTIVE:
        #     Exclusion of any direction that has artifacts not curable by the resampling stage
        # """
        hardiQC.RerunDTIPrepStage(fixednrrdfilename, prepDir, subjname, xmlfilename, resampling_method,
                                  nDirections)

        """
        ---------------------------------------------------------------------------------
        STEP SIX
        ---------------------------------------------------------------------------------

        OBJECTIVE:
            correct for motion the DTIPrep QCed sequences where slice-wise/gradient-wise denoising already
            been carried out, as such we can use standard interpolation

        """

        nointegerflag = hardiQC.PerformDWIGeometricMeanReferenceMotionCorrectionMCFLIRT6DOFupdate(fixednrrdfilename,
                                                                                                  prepDir,
                                                                                                  subjname,
                                                                                                  resampling_method,
                                                                                                  nDirections,
                                                                                                  useBrainMask)
        if nointegerflag:
            sys.exit("CHECK MOTION CORRECTED IMAGE INTENSITY: HAS FLOAT")


        """
        ---------------------------------------------------------------------------------
        STEP SEVEN
        ---------------------------------------------------------------------------------

        OBJECTIVE:
            brain mask the motion corrected datasets
        """
        hardiQC.BrainMaskDWIGeometricMeanMotionCorrectedDWIupdate(fixednrrdfilename, prepDir, subjname,
                                                                  resampling_method, nDirections)



if __name__ == "__main__":

    local = '/home/heejong/server/oak_research/'
    server = '/research/vidaimaging/projects/'
    localorserver = local

    # # IBIS_DTI65_Data_anonymized: Total image 494
    # dir_anon = '/research/vidaimaging/projects/Autism/IBIS_DTI65_Data_anonymized/'
    # anon_ids = glob.glob(dir_anon+'*')
    # anon_allnrrds = glob.glob(dir_anon+'*/*/*/*/*/*/ibis_*.nrrd', recursive=True)

    # IBIS_DTI65_Data_nrrd: Total image 494 / ID-agegroup: 465
    dir_nrrd = localorserver+'Autism/IBIS_DTI65_Data_nrrd/'
    # nrrd_ids = glob.glob(dir_nrrd + '*')
    nrrd_idgroups = glob.glob(dir_nrrd + '*/*')
    # nrrd_allnrrds = glob.glob(dir_nrrd + '*/*/*/*/*/*/ibis_*.nrrd', recursive=True)

    # # IBIS_DTI65_Data_minc: Total image 2617 / ID-agegroup 521
    # local = '/home/server/oak_research/'
    # server = '/research/vidaimaging/projects/'
    # dir_minc = local+'Autism/IBIS_DTI65_Data_minc/DTI65_Minc/'
    # # mnc2nii
    # minc_ids = glob.glob(dir_minc + '*')
    # minc_idgroups = glob.glob(dir_minc + '*/*')
    # minc_allmincs = glob.glob(dir_minc + '*/*/*/*/ibis_*.mnc', recursive=True)


    # ID = nrrd_idgroups[sidx].split('/')[-2]
    # time = nrrd_idgroups[sidx].split('/')[-1]
    suffnrrd = '/mri/native/DTI65/NrrdOrig/ibis_'
    # nrrdfilename = os.path.join(dir_nrrd, ID + '/' + time) + suffnrrd + ID + '_' + time + '_DTI65_001.nrrd'
    # outDir = os.path.join(local + 'Autism/output_hardiprep_simple/', ID + '/' + time)
    nDirections = int(65)
    prepDir_suffix = 'SHORE'
    resampling_method = 'shore'
    xmlfilename = './IBIS_DTIPrep_PROTOCOL_simple.xml'
    # phanname = ID + '_' + time
    useBrainMask = False


    # num_cores = 10
    # O = Parallel(n_jobs=num_cores)(
    #     delayed(hardiprep_vsimple)(os.path.join(dir_nrrd, nrrd_idgroups[sidx].split('/')[-2] + '/' + nrrd_idgroups[sidx].split('/')[-1]) + suffnrrd + nrrd_idgroups[sidx].split('/')[-2] + '_' +nrrd_idgroups[sidx].split('/')[-1] + '_DTI65_001.nrrd',
    #                                nrrd_idgroups[sidx].split('/')[-2] + '_' +nrrd_idgroups[sidx].split('/')[-1],
    #                                os.path.join(local + 'Autism/output_hardiprep_simple/', nrrd_idgroups[sidx].split('/')[-2] + '/' +nrrd_idgroups[sidx].split('/')[-1]),
    #                                nDirections, prepDir_suffix, resampling_method,
    #                                xmlfilename, useBrainMask) for sidx in range(len(nrrd_idgroups)))


    # reindices = [118,
    #  127,
    #  128,
    #  140,
    #  143,
    #  217,
    #  223,
    #  229,
    #  238,
    #  241,
    #  244,
    #  245,
    #  246,
    #  247,
    #  253,
    #  261,
    #  266]
    # for sidx in reindices:

    # import csv
    # with open('/home/heejong/HDD2T/hardiprep_vsimple/temp-id-agegroup.csv') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         ID, agegroup = row[0].split('\t')
    #         print(ID, agegroup)


    # half = int(np.floor(len(nrrd_idgroups)/2))
    for sidx in range(len(nrrd_idgroups)):
    # for sidx in (range(146)):

        print("=======================")
        print(" {} out of {} ".format(sidx, len(nrrd_idgroups)))
        print("=======================")

        # sidx = 0
        ID = nrrd_idgroups[sidx].split('/')[-2]
        agegroup = nrrd_idgroups[sidx].split('/')[-1]

        # main()
        suffnrrd = '/mri/native/DTI65/NrrdOrig/ibis_'
        origNrrdfilename = os.path.join(dir_nrrd, ID+'/'+agegroup) + suffnrrd + ID + '_'+agegroup+'_DTI65_001.nrrd'
        outDir = os.path.join(local+'Autism/output_hardiprep_simple/', ID+'/'+agegroup)
        nDirections = int(65)
        prepDir_suffix = 'SHORE'
        resampling_method = 'shore'
        xmlfilename = './IBIS_DTIPrep_PROTOCOL_simple.xml'
        phanname = ID + '_' + agegroup


        if os.path.exists(origNrrdfilename):

            print("===================================")
            print('Working on %s' % (origNrrdfilename))
            print("===================================")

            if not os.path.exists(outDir):
                os.makedirs(outDir)

            # mincs dcm2mnc -dname  -stdin -clobber /tmp/908.1.all.q/TarLoad-23-27-k7hdQD\\n')

            """
            ---------------------------------------------------------------------------------
            STEP Zero
            ---------------------------------------------------------------------------------
            
            (1) create hardi QC directories and copy the original nrrd
            (2) fix the dimension and thickness of the given nrrd files
            (3) convert them to nifti and generate the btable for dsi_studio visualization
            
            """
            prepDir, fixednrrdfilename, bvalbveczeroflag = hardiQC.PrepareQCsession(origNrrdfilename, outDir, phanname, prepDir_suffix,nDirections, check_btable=True)
            if bvalbveczeroflag:
                print("NO BVALUES BVECTORS or LESS THAN 6 GRADIENTS IN NRRD FILE")
                continue

            """
            ---------------------------------------------------------------------------------
            STEP ONE
            ---------------------------------------------------------------------------------

            OBJECTIVE:
                dtiprep without motion correction
                check the quality of the individual directions, e.g. missing slices, intensity artifacts, Venetian blind
            """

            baselineexcludeflag = hardiQC.RunDTIPrepStage( prepDir, phanname, xmlfilename, nDirections)
            if baselineexcludeflag:
                print("NO BASELINE")
                continue


            """
            # ---------------------------------------------------------------------------------
            # STEP TWO
            # ---------------------------------------------------------------------------------
            #
            # OBJECTIVE:
            #     quantify fast bulk motion within each gradient to exclude those having intra-scan
            #     motion (see Benner et al (2011) - Diffusion imaging with prospective motion correction and reacquisition)
            #
            #     here we have zero tolerance, any gradient having at least one slice with signal drop out will be excluded
            # """

            hardiQC.PerformWithinGradientMotionQC( prepDir, phanname, nDirections)

            """
            ---------------------------------------------------------------------------------
            STEP TRHEE
            ---------------------------------------------------------------------------------
            Assumptions:
                (1) WithinGradientMotionQC has been performed
                (2) DTIPrep (without motion) has been performed

            """
            hardiQC.ExtractBaselineAndBrainMask( prepDir, phanname, nDirections)
            """
            ---------------------------------------------------------------------------------
            STEP FOUR
            ---------------------------------------------------------------------------------

            OBJECTIVE:
                resample the corrupted slices (detected via DTIPrep and within-gradient motion) Qc
                in the q-space plus gradient-wise denoising
            """
            lessgradientflag = hardiQC.PerformResampleCorruptedSlicesInQspace( prepDir,phanname,
                                                                              resampling_method,
                                                                              nDirections)

            if lessgradientflag:
                print("LESS THAN 6 GRADIENTS INPUT FOR RESAMPLING STEP")
                continue
            #     sys.exit("LESS THAN 6 GRADIENTS INPUT FOR RESAMPLING STEP")

            """
            # ---------------------------------------------------------------------------------
            # STEP FIVE
            # ---------------------------------------------------------------------------------
            #
            # OBJECTIVE:
            #     Exclusion of any direction that has artifacts not curable by the resampling stage
            # """
            hardiQC.RerunDTIPrepStage( prepDir, phanname, xmlfilename, resampling_method,
                                      nDirections)


            """
            ---------------------------------------------------------------------------------
            STEP SIX
            ---------------------------------------------------------------------------------

            OBJECTIVE:
                correct for motion the DTIPrep QCed sequences where slice-wise/gradient-wise denoising already
                been carried out, as such we can use standard interpolation

            """

            # nointegerflag = hardiQC.PerformDWIGeometricMeanReferenceMotionCorrectionMCFLIRT6DOFupdate(fixednrrdfilename, prepDir,
            #                                                                            phanname,
            #                                                                            resampling_method,
            #                                                                            nDirections, useBrainMask)
            # if nointegerflag:
            #     print("CHECK MOTION CORRECTED IMAGE INTENSITY: HAS FLOAT")
            #     continue

            hardiQC.PerformDWIBaselineReferenceMotionCorrection( prepDir,phanname,
                                                                       resampling_method,
                                                                       nDirections)


            """
            ---------------------------------------------------------------------------------
            STEP SEVEN
            ---------------------------------------------------------------------------------

            OBJECTIVE:
                brain mask the motion corrected datasets
            """
            # hardiQC.BrainMaskDWIGeometricMeanMotionCorrectedDWIupdate( prepDir, phanname,
            #                                                           resampling_method, nDirections, check_btable=True)

            hardiQC.BrainMaskDWIBaselineReferenceMotionCorrectedDWIupdate(prepDir, phanname, resampling_method,
                                                      nDirections, check_btable = True)





