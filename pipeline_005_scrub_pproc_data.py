__author__ = 'kanaan'

import os
from network.centrality import *
from utilities.utils import mkdir_path
from denoise.bandpass import *
from variables.subject_list import *
import subprocess
import shutil


def scrub_data(population, workspace_dir):

    count = 0
    for subject in population:
        count +=1
        print '####################################'
        print '%s. Running SCRUBBING FOR SUBJECT %s'%(count,subject)

        #############################################################
        #################### Input and output folders
        subject_dir = os.path.join(workspace_dir, 'GluConnectivity', subject)

        mkdir_path( os.path.join(subject_dir , 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp'))
        pproc      = os.path.join(subject_dir , 'functional_native_brain_preproc_FWHM_AROMA_residual_bp/bandpassed_demeaned_filtered.nii.gz')
        pproc_2mm  = os.path.join(subject_dir , 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/rest_pproc.nii')
        anat2mni   = os.path.join(subject_dir , 'anatomical_MNI2mm_xfm/MP2RAGE_DESKULL_RPI_resample_ero_fieldwarp.nii.gz')
        func2anat  = os.path.join(subject_dir , 'functional_ANAT2mm_xfm/REST_calc_resample_corrected_volreg_maths_tstat_flirt.mat')


        #############################################################
        #################### WARPING PPROC TO MNI
        print '.... warping pproc to MNI SPACE'
        if not os.path.isfile(pproc_2mm):
            if os.path.isfile(pproc):
                print '... Warping to MNI'
                os.system('FSLOUTPUTTYPE=NIFTI')
                os.system(' '.join([ 'applywarp',
                                 '--in='     +  pproc,
                                 '--ref='    +  '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz',
                                 '--out='    +  pproc_2mm,
                                 '--warp='   +  anat2mni,
                                 '--premat=' +  func2anat]))

        #############################################################
        #################### MAKE SCRUB

        print '....scrubbing '

        if os.path.isfile(os.path.join(subject_dir, 'functional_motion_FDPower/FD.1D' )):

            FDs = np.loadtxt(os.path.join(subject_dir, 'functional_motion_FDPower/FD.1D' ))

            # GET LIST OF GOOD FRAMES
            in_frames = []
            for frame, fd in enumerate(FDs):
                if fd < 0.2:
                    in_frames.append(frame)
            #print subject,'-----> GOOD FRAMES =', in_frames


            if len(in_frames) > 130:
                print '..........Scrubbing frames above FD=0.2 for subject [ %s ]' %subject
                print '..........Subject has %s good frames' % len(in_frames)
                print '...........taking first 100 good frames'
                frames = str(in_frames[0:130]).replace(" ","")

                # SCRUB DATA
                pproc_2mm  = os.path.join(subject_dir , 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/rest_pproc.nii')
                scrubbed   = os.path.join(subject_dir , 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/rest_pproc_scrubbed.nii')

                os.system("3dcalc -a %s%s -expr 'a' -prefix %s" %(pproc_2mm, frames, scrubbed))

            else:
                print '**** Subject [ %s ]  has less than 100 frames with FD below the 0.2mm threshold'%subject

        scrub_subs = []
        for subject in population:
            if os.path.isfile( os.path.join(workspace_dir, 'GluConnectivity', subject, 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/rest_pproc_scrubbed.nii')):
                scrub_subs.append(subject)
        print scrub_subs

scrub_data(['BM8X'], output_dir_a)
scrub_data(controls_a, output_dir_a)
scrub_data(patients_a, output_dir_a)
scrub_data(patients_b, output_dir_b)