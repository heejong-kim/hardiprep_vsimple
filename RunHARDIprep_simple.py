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

import hardi.qc as hardiQC
import argparse
from joblib import Parallel, delayed

def hardiprep_vsimple(nrrdfilename, phanname, outDir, nDirections, prepDir_suffix, resampling_method, xmlfilename, check_btable=False):

    if not os.path.exists(nrrdfilename):
        print("MISSING INPUT NRRD FILE")
        # sys.exit("MISSING INPUT NRRD FILE")
        return
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
        prepDir, fixednrrdfilename, bvalbveczeroflag = hardiQC.PrepareQCsession(origNrrdfilename, outDir, phanname, prepDir_suffix,nDirections, check_btable=check_btable)
        if bvalbveczeroflag:
            # sys.exit("NO BVALUES BVECTORS or LESS THAN 6 GRADIENTS IN NRRD FILE")
            print("NO BVALUES BVECTORS or LESS THAN 6 GRADIENTS IN NRRD FILE")
            return

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
            # sys.exit("NO BASELINE")
            print("NO BASELINE")
            return


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
            # sys.exit("LESS THAN 6 GRADIENTS INPUT FOR RESAMPLING STEP")
            print("LESS THAN 6 GRADIENTS INPUT FOR RESAMPLING STEP")
            return

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
                                                  nDirections, check_btable = check_btable)




if __name__ == "__main__":

    local = '/home/heejong/server/oak_research/'
    server = '/research/vidaimaging/projects/'
    localorserver = server

    # IBIS_DTI65_Data_nrrd: Total image 494 / ID-agegroup: 465
    dir_nrrd = localorserver+'Autism/IBIS_DTI65_Data_nrrd/'
    nrrd_idgroups = glob.glob(dir_nrrd + '*/*')

    suffnrrd = '/mri/native/DTI65/NrrdOrig/ibis_'
    nDirections = int(65)
    prepDir_suffix = 'SHORE'
    resampling_method = 'shore'
    xmlfilename = './IBIS_DTIPrep_PROTOCOL_simple.xml'


    for sidx in range(len(nrrd_idgroups)):

        print("=======================")
        print(" {} out of {} ".format(sidx, len(nrrd_idgroups)))
        print("=======================")

        # sidx = 0
        ID = nrrd_idgroups[sidx].split('/')[-2]
        agegroup = nrrd_idgroups[sidx].split('/')[-1]

        # main()
        suffnrrd = '/mri/native/DTI65/NrrdOrig/ibis_'
        origNrrdfilename = os.path.join(dir_nrrd, ID+'/'+agegroup) + suffnrrd + ID + '_'+agegroup+'_DTI65_001.nrrd'
        outDir = os.path.join(localorserver+'Autism/output_hardiprep_simple/', ID+'/'+agegroup)
        nDirections = int(65)
        prepDir_suffix = 'SHORE'
        resampling_method = 'shore'
        xmlfilename = './IBIS_DTIPrep_PROTOCOL_simple.xml'
        phanname = ID + '_' + agegroup
        check_btable = True

        hardiprep_vsimple(origNrrdfilename, phanname, outDir, nDirections, prepDir_suffix, resampling_method, xmlfilename, check_btable=check_btable)
