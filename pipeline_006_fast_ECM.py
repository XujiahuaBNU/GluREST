__author__ = 'kanaan'

import os
from network.centrality import *
from utilities.utils import mkdir_path
from denoise.bandpass import *
from variables.subject_list import *
import subprocess
import shutil

def calc_ecm(population, workspace_dir):

    for subject in population:
        print '####################################'
        print 'Running fast ECM Subject %s'%subject

        #############################################################
        #################### Input and output folders
        subject_dir = os.path.join(workspace_dir, 'GluConnectivity', subject)
        outdir = os.path.join(subject_dir, 'FAST_ECM_SCRUBBED')
        mkdir_path(outdir)
        os.chdir(outdir)

        pproc_2mm_scrubbed  = os.path.join(subject_dir , 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/rest_pproc_scrubbed.nii')

        #############################################################
        #################### Run Fast ECM

        mask = '/SCR4/workspace/project_GluRest/OUT_DIR_A/GluConnectivity/GM2mm_bin.nii'

        if os.path.isfile(pproc_2mm_scrubbed):
            if not os.path.isfile(os.path.join(outdir, 'FAST_ECM.nii')):
                matlab_cmd = ['matlab',  '-version', '8.2', '-nodesktop' ,'-nosplash'  ,'-nojvm' ,'-r "fastECM(\'%s\', \'1\', \'1\', \'1\', \'20\', \'%s\') ; quit;"'
                                       %(pproc_2mm_scrubbed, mask)]
                subprocess.call(matlab_cmd)

                shutil.move(os.path.join(subject_dir, 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/rest_pproc_fastECM.nii'), os.path.join(outdir, 'FAST_ECM.nii'))
                shutil.move(os.path.join(subject_dir, 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/rest_pproc_rankECM.nii'), os.path.join(outdir, 'RANK_ECM.nii'))
                shutil.move(os.path.join(subject_dir, 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/rest_pproc_normECM.nii'), os.path.join(outdir, 'NORM_ECM.nii'))
                shutil.move(os.path.join(subject_dir, 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/rest_pproc_degCM.nii'), os.path.join(outdir, 'DCM.nii'))



#calc_ecm(['HCTT'], output_dir_a)
calc_ecm(controls_a, output_dir_a)
calc_ecm(patients_a, output_dir_a)
calc_ecm(patients_b, output_dir_b)
