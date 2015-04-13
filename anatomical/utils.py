__author__ = 'kanaan'


import nipype.interfaces.freesurfer as freesurfer
from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util

def grab_spm_tissues(list):
    '''
    Function to grab specific files from list
    '''
    gm_probs =  str(list[0])
    wm_probs =  str(list[1])
    cm_probs =  str(list[2])
    return gm_probs, wm_probs, cm_probs


#freesurfer tissue class labels
GM = [3,8,42,17,18,53,54,11,12,13,26,50,51,52,58,9,10,47,48,49,16,28,60]
WM = [2,7,41,46]
CSF = [4,5,14,15,24,31, 43,44,63, 72]


def get_aparc_aseg(files):
    """Return the aparc+aseg.mgz file"""
    for name in files:
        if 'aparc+aseg.mgz' in name:
            return name
    raise ValueError('aparc+aseg.mgz not found')

def freesurfer_nifti():
    '''
    Simple method to convert freesurfer mgz files to nifti format
    '''

    #start with a useful function to grab data
    #define workflow
    flow = Workflow(name = 'freesurfer_nifti')

    inputnode = Node(util.IdentityInterface(fields=['mgz_image', 'anatomical']),
                name='inputnode')

    outputnode = Node(util.IdentityInterface(fields=['aparc_aseg_nifti']),
                name='outputnode')

    #define nodes
    convert_aparc_aseg                  = Node(interface= freesurfer.MRIConvert(), name = 'aparc_aseg_nifti')
    convert_aparc_aseg.inputs.out_type  = 'nii'

    anatomical                          = Node(interface= freesurfer.MRIConvert(), name = 'anatomical_ready')
    anatomical.inputs.out_type                   = 'nii'


    #connect nodes
    return flow
