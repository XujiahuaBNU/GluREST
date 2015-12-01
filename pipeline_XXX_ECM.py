__author__ = 'kanaan'

import os
from network.centrality import *
from utilities.utils import mkdir_path
from denoise.bandpass import *
from variables.subject_list import *

mask = '/scr/sambesi1/ROI/mask_4mm_bin.nii.gz'

def calc_ecm(population, workspace_dir):

    for subject in population:

        print 'Running Subject %s'%subject

        # input and output folders
        subject_dir = os.path.join(workspace_dir, 'GluConnectivity', subject)
        outdir = os.path.join(subject_dir, 'ECM')
        mkdir_path(outdir)
        os.chdir(outdir)

        #inputdata
        pproc_4mm = os.path.join(subject_dir , 'functional_MNI4mm_preproc_FWHM_AROMA_residual_high_pass/bandpassed_demeaned_filtered.nii.gz')
        if os.path.isfile(pproc_4mm):
            # bp 4mm pproc data
            if not os.path.isfile(os.path.join(subject_dir, 'ECM/bandpassed_demeaned_filtered.nii.gz')):
                bp_file = bandpass_voxels(pproc_4mm, (0.01,0.1))

            MNI2mm_gm =  os.path.join(subject_dir , 'anatomical_MNI2mm_tissue_gm/TISSUE_CLASS_1_GM_OPTIMIZED_resample_warp_thresh.nii.gz')
            MNI4mm_gm =  os.path.join(subject_dir , 'ECM/mask_4mm.nii.gz')
            MNI4mm_gm_bin = os.path.join(subject_dir , 'ECM/mask_4mm_bin.nii.gz')
            os.system('flirt -in %s -ref %s -out %s -applyxfm'%(MNI2mm_gm, pproc_4mm, MNI4mm_gm))
            os.system('fslmaths %s -thr 0.3 -bin %s' %(MNI4mm_gm,MNI4mm_gm_bin))

            if not os.path.isfile(os.path.join(subject_dir, 'ECM/resting_state_graph/calculate_centrality/eigenvector_centrality_binarize.nii.gz')):
                print '...computing ecm'
                ecm_graph = create_resting_state_graphs()
                ecm_graph.inputs.inputspec.method_option     = 1
                ecm_graph.inputs.inputspec.subject           = bp_file
                ecm_graph.inputs.inputspec.template          = mask
                ecm_graph.inputs.inputspec.threshold_option  = 0
                ecm_graph.inputs.inputspec.threshold         = 0.001
                ecm_graph.inputs.inputspec.weight_options    = [True, True]
                ecm_graph.base_dir                           = outdir
                ecm_graph.run()

if __name__ == "__main__":
    calc_ecm(['HCTT'], output_dir_a)
    # calc_ecm(controls_a, output_dir_a)
    # calc_ecm(patients_a, output_dir_a)
    # calc_ecm(patients_b, output_dir_b)
