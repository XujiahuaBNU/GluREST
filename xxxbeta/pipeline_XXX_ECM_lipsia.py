__author__ = 'kanaan'

import os
from network.centrality import *
from utilities.utils import mkdir_path
from denoise.bandpass import *
from variables.subject_list import *


def calc_ecm(population, workspace_dir):

    # all_gm = []
    #
    # if not os.path.isfile(os.path.join(workspace_dir, 'GluConnectivity', 'COMBINED_GM_MASK.nii.gz')):
    #     print 'Creating Group GM mask'
    #     for subject in population:
    #
    #         # input and output folders
    #         subject_dir = os.path.join(workspace_dir, 'GluConnectivity', subject)
    #         out_mask        = os.path.join(workspace_dir, 'GluConnectivity', 'COMBINED_GM_MASK.nii.gz')
    #         MNI2mm_gm   = os.path.join(subject_dir , 'anatomical_MNI2mm_tissue_gm/TISSUE_CLASS_1_GM_OPTIMIZED_resample_warp_thresh.nii.gz')
    #
    #         if os.path.isfile(MNI2mm_gm):
    #             all_gm.append(MNI2mm_gm)
    #
    #     maths_input = []
    #     for i in all_gm:
    #         x = '-add %s'%i
    #         maths_input.append(x)
    #
    #     maths_string = ' '.join(maths_input)[5:]
    #     os.system('fslmaths %s -mul /usr/share/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz %s'%(maths_string, out_mask))
    #
    #     out_mask_4mm = os.path.join(workspace_dir, 'GluConnectivity', 'COMBINED_GM_MASK_4mm.nii.gz')
    #     mni_4mm = '/SCR/ROI/brain_4mm.nii.gz'
    #     os.system('flirt -in %s -ref %s -out %s -applyisoxfm 4' %(out_mask, mni_4mm, out_mask_4mm ))

    for subject in population:
        print 'Running Subject %s'%subject

        # input and output folders
        subject_dir = os.path.join(workspace_dir, 'GluConnectivity', subject)
        outdir = os.path.join(subject_dir, 'ECM_LIPSIA')
        mkdir_path(outdir)
        os.chdir(outdir)

        # TRANSFORM NATIVE IMAGE TO MNI2mm
        mkdir_path( os.path.join(subject_dir , 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp'))
        pproc = os.path.join(subject_dir , 'functional_native_brain_preproc_FWHM_AROMA_residual_bp/bandpassed_demeaned_filtered.nii.gz')
        pproc_2mm  = os.path.join(subject_dir , 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/pproc.nii.gz')
        anat2mni  = os.path.join(subject_dir , 'anatomical_MNI2mm_xfm/MP2RAGE_DESKULL_RPI_resample_ero_fieldwarp.nii.gz')
        func2anat = os.path.join(subject_dir , 'functional_ANAT2mm_xfm/REST_calc_resample_corrected_volreg_maths_tstat_flirt.mat')
        if not os.path.isfile(pproc_2mm):
            if os.path.isfile(pproc):
                print '... Warping to MNI'
                os.system(' '.join([ 'applywarp',
                                 '--in='     +  pproc,
                                 '--ref='    +  '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz',
                                 '--out='    +  pproc_2mm,
                                 '--warp='   +  anat2mni,
                                 '--premat=' +  func2anat]))


        # Convert VISTA to NIFTI
        pproc_2mmv = os.path.join(subject_dir , 'functional_MNI2mm_brain_preproc_FWHM_AROMA_residual_bp/pproc.v')
        if not os.path.isfile(pproc_2mmv):
            print '...Converting NIFTI to VISTA.. make sure you are running this on telemann'
            os.system('isisconv -in %s -out %s' %(pproc_2mm, pproc_2mmv))

        # print '... Calculating ECM'
        # if os.path.isfile(pproc_2mmv):
        #     if not os.path.isfile(os.path.join(outdir, 'ecm.v')):
        #         print ''
        #         # os.system('fslsplit %s slice -z'%rest)
        #         #
        #         # #change from temporal ordering of the slice to a z-ordering of the data
        #         # os.system('for i in slice00*; do mkdir ${i/.nii.gz}; fslsplit $i ${i/.nii.gz}/vol -t; fslmerge -z ${i/.nii.gz/_reorder} ${i/.nii.gz}/vol*;done')
        #         #
        #         # # merge in different orientation
        #         # os.system('fslmerge -t merge slice00*_reorder*')
        #         #
        #         # #convert to vista, change datatype to short and scale by 100
        #         # # os.system('/a/sw/misc/bin/diffusion/vnifti2image merge.nii.gz | vconvert -map linear -repn short -b 10000 | /a/sw/misc/bin/diffusion/vflip3d -x -z -out bandpassed_demeaned_filtered.v')
        #         # os.system('/a/sw/misc/bin/diffusion/vnifti2image merge.nii.gz '
        #         #  os.system('vconvert -map linear -repn short -b 10000 bandpassed_demeaned_filtered.v')
        #         #
        #         # # #get mask
        #         # mask_4mm = os.path.join(workspace_dir, 'GluConnectivity', 'COMBINED_GM_MASK_4mm.nii.gz')
        #         # os.system('/a/sw/misc/bin/diffusion/vnifti2image %s | vbinarize -min 3 | /a/sw/misc/bin/diffusion/vflip3d -x -out mask.v'%mask_4mm)
        #
        #         #clean
        #         # os.system('rm -rf slice* merge.nii.gz slice00*_reorder*')
        #
        #         #calcualte ECM
        #         GMMASK = '/scr/sambesi4/workspace/project_GluRest/OUT_DIR_A/GluConnectivity/GM2mm_bin.v'
        #
        #         os.system('vconvert %s -map linear -repn short -b 5000 bandpassed_demeaned_filtered_short.v' %pproc_2mmv)
        #         os.system('vecm -in %s -out ecm.v -mask %s'%(pproc_2mmv, GMMASK))
        #
        #         # convert to nifti
        #         # os.system('/a/sw/misc/bin/diffusion/vimage2nifti ecm.v ecm.nii.gz')

# calc_ecm(['BM8X'], output_dir_a)
calc_ecm(controls_a, output_dir_a)
# calc_ecm(patients_a, output_dir_a)
# calc_ecm(patients_b, output_dir_b)
