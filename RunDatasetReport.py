

from __future__ import division
import os
import glob
import numpy as np
import ntpath
import time

import subprocess
from dipy.io.gradients import read_bvals_bvecs

import nrrd
import hardi.io as hardiIO
import hardi.qc as hardiQC

import argparse


import pandas as pd



if __name__ == "__main__":
    local = '/home/heejong/server/oak_research/'
    server = '/research/vidaimaging/projects/'
    localorserver = local

    ## input
    # IBIS_DTI65_Data_nrrd: Total image 494 / ID-agegroup: 465
    dir_nrrd = localorserver+'Autism/IBIS_DTI65_Data_nrrd/'
    # nrrd_ids = glob.glob(dir_nrrd + '*')
    nrrd_idgroups = glob.glob(dir_nrrd + '*/*')
    dir_output = localorserver + 'Autism/output_hardiprep_simple/'

    metric_file = dir_output+'dataset-report-nrrd-final2.tsv'



    writer_head = ['ID_agegroup', 'empty_dir', 'no_baseline', 'less_gradient', 'bvector_empty',
                   'n_of_gradients_orig','corrupted_gradients',  'corrected_gradients',
                   'image_size', 'image_origin', 'image_spacing', 'image_space','diffusion_fail',
                   'avg_rotation', 'avg_trasnlation','bvector_flip_first','bvector_flip_last',
                    'qced_filename', 'minc_version']


    if not os.path.exists(metric_file):
        df_head = pd.DataFrame(columns=writer_head)
        df_head.to_csv(metric_file, sep='\t', index=False, header=True, mode='w')
        print('Creating report file {}'.format(metric_file))

    
    for sidx in range(len(nrrd_idgroups)):

        ID = nrrd_idgroups[sidx].split('/')[-2]
        time = nrrd_idgroups[sidx].split('/')[-1]
        idagegroup = ID+'_'+time
        df = pd.read_csv(metric_file, delimiter='\t')
        if idagegroup in [str(si) for si in df.ID_agegroup]:
            print(idagegroup + ' stats already in the report, skip')
            continue

        suffnrrd = '/mri/native/DTI65/NrrdOrig/ibis_'
        origNrrdfilename = os.path.join(dir_nrrd, ID+'/'+time) + suffnrrd + ID + '_'+time+'_DTI65_001.nrrd'
        outDir = os.path.join(local+'Autism/output_hardiprep_simple/', ID+'/'+time)
        nDirections = int(65)
        prepDir_suffix = 'SHORE'
        resampling_method = 'shore'
        xmlfilename = './IBIS_DTIPrep_PROTOCOL_simple.xml'

        if not os.path.exists(origNrrdfilename):
            empty_dir = True
            print('EMPTY NRRD')
            df = pd.DataFrame([[idagegroup, empty_dir, np.nan, np.nan,np.nan,np.nan, np.nan, np.nan,
                                np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
                                ]], columns=writer_head)
            print(df)
            df.to_csv(metric_file, sep='\t', index=False, header=False, mode='a')
        else:
            empty_dir = False
            prepDir = os.path.join(outDir, 'HARDIprep_QC_%s' % (prepDir_suffix))
                    # prepare the files for subsequent processing
            origbvalf = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.bvals' % (
                nDirections, idagegroup, nDirections))
            orignrrdreportfname = os.path.join(prepDir, 'DWI_%ddir/origNrrdReport.txt' % (
                nDirections))
            qcedbvalf = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed.bvals' % (
                nDirections, idagegroup, nDirections))
            corredbvalf = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_ResampleSHORE_QCed.bvals' % (
                nDirections, idagegroup, nDirections))
            dtiprepreport = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_DTIPrepQCReport.txt' % (
                nDirections, idagegroup, nDirections))
            fibcheck = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir.*.fib.gz' % (
                nDirections, idagegroup, nDirections))
            fibfinalcheck = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_ResampleSHORE_QCed_DWIGeometricMean_MCFLIRT6DOF_masked.src.gz.dti.fib.gz' % (
                nDirections, idagegroup, nDirections))
            qced_filename = os.path.join(prepDir, 'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_ResampleSHORE_QCed_DWIGeometricMean_MCFLIRT6DOF_masked.nrrd' % (
                nDirections, idagegroup, nDirections))
            motionqcfname = os.path.join(prepDir,
                                         'DWI_%ddir/%s_DWI_%ddir_QCed_WithinGradientMotionQCed_ResampleSHORE_QCed_DWIGeometricMean_MCFLIRT6DOF_QCreport.txt' % (
                                             nDirections, idagegroup, nDirections))

            no_baseline = os.path.exists(os.path.join(prepDir, 'DWI_%ddir/nobaseline' % (nDirections)))
            less_gradient = os.path.exists(os.path.join(prepDir, 'DWI_%ddir/lessgradients' % (nDirections)))
            bvector_empty = os.path.exists(os.path.join(prepDir, 'DWI_%ddir/bvalbveczero' % (nDirections)))
            with open(orignrrdreportfname, 'r') as orep:
                filereport = orep.readlines()

            minc_version = filereport[-1].split()[-1]

            if no_baseline or less_gradient or bvector_empty:
                n_of_gradients_orig, corrupted_gradients, corrected_gradients, \
                image_size, image_origin, image_spacing, image_space, \
                bvector_flip, qced_filename = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
            else:

                with open(origbvalf,'r') as origb:
                    n_of_gradients_orig = len(origb.readlines())

                with open(qcedbvalf, 'r') as qcb:
                    corrupted_gradients = n_of_gradients_orig-len(qcb.readlines())

                with open(dtiprepreport, 'r') as dp:
                    dtiprepresult = dp.readlines()

                with open(motionqcfname, 'r') as mqc:
                    motionqcresult = mqc.readlines()

                avg_rotation = motionqcresult[6].split()[4]
                avg_translation = motionqcresult[5].split()[4]

                image_size = dtiprepresult[1].split()[-1] == 'OK'
                image_origin = dtiprepresult[2].split()[-1] == 'OK'
                image_spacing = dtiprepresult[3].split()[-1] == 'OK'
                image_space = dtiprepresult[4].split()[-1] == 'OK'

                for l in dtiprepresult:
                    if 'Diffusion gradient Check' in l:
                        diffusion_fail = l.split()[-1] == 'FAILED'

                fibglob =glob.glob(fibcheck)
                fibfinalglob = glob.glob(fibfinalcheck)
                if len(fibglob) > 1:
                    bvector_flip_first = glob.glob(fibcheck)[1].split('.')[-4] if glob.glob(fibcheck)[1].split('.')[-4] != 'gz' else False
                else:
                    bvector_flip_first = glob.glob(fibcheck)[0].split('.')[-4] if glob.glob(fibcheck)[0].split('.')[
                                                                                      -4] != 'gz' else False

                if len(fibfinalglob) > 1:
                    bvector_flip_last = glob.glob(fibfinalcheck)[1].split('.')[-4] if glob.glob(fibfinalcheck)[1].split('.')[
                                                                                  -4] != 'gz' else False
                else:
                    bvector_flip_first = glob.glob(fibcheck)[0].split('.')[-4] if glob.glob(fibcheck)[0].split('.')[
                                                                                      -4] != 'gz' else False


                if not os.path.exists(corredbvalf):
                    corrected_gradients = np.nan
                else:
                    with open(corredbvalf, 'r') as cb:
                        corrected_gradients = corrupted_gradients-(n_of_gradients_orig-len(cb.readlines()))

            df = pd.DataFrame( [[idagegroup, empty_dir, no_baseline, less_gradient, bvector_empty,
                                 n_of_gradients_orig,corrupted_gradients,corrected_gradients,
                                  image_size, image_origin, image_spacing, image_space,diffusion_fail,
                                 avg_rotation, avg_translation,bvector_flip_first,bvector_flip_last,
                                  qced_filename, minc_version]], columns=writer_head)
            print(df)
            df.to_csv(metric_file, sep='\t', index=False, header=False, mode='a')



    report = pd.read_csv(metric_file, delimiter='\t')
    print('Empty dir (wrong format nrrd filename): {}'.format(np.sum(report.empty_dir)))
    print('No baseline or less gradient or empty bvector: {}'.format(np.sum((report.no_baseline + report.less_gradient + report.bvector_empty)> 0)))

    report1 = report[(report.empty_dir == False) * (report.no_baseline + report.less_gradient + report.bvector_empty)==0]
    print("More than 30% corrupted: {}".format(np.sum(report1.corrupted_gradients >= 64*0.3)))

    report2 = report1[report1.corrupted_gradients <= 64*0.3]
    report3 = report2[(report2.n_of_gradients_orig - report2.corrupted_gradients + report2.corrected_gradients - 1) > 64 * 0.7].reset_index()
    print("Final # of gradients is less than 70% of 64 gradients: {}".format(np.sum((report2.n_of_gradients_orig - report2.corrupted_gradients + report2.corrected_gradients - 1) < 64 * 0.7)))

    # Visual check before bet image (all dwis one page)

