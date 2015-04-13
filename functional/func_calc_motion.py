__author__ = 'kanaan'

from   nipype.pipeline.engine import Workflow, Node
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype.interfaces.afni import preprocess


def func_calc_motion_affines():
    '''
    Method to run motion correction using MCFLIRT.
    collects motion pars and affines used later to create a unified warp.
    This is done after distortion correction.

    inputs
        inputnode.func_disortion_corrected_3D
    outputs
        outputnode.moco_func       # motion-corrected timeseries
        outputnode.moco_rms        # absolute and relative displacement parameters
        outputnode.moco_param      # text-file with motion parameters
        outputnode.moco_mean       # mean timeseries image
        outputnode.moco_mat        # transformation matrices
    '''

    # Define Workflow
    flow        = Workflow(name='func_motion_correction')
    inputnode   = Node(util.IdentityInterface(fields=['func']),
                          name='inputnode')

    outputnode  = Node(util.IdentityInterface(fields=['moco_func',
                                                      'moco_rms',
                                                      'moco_param',
                                                      'moco_mean',
                                                      'moco_mat']),
                                              name = 'outputnode')

    # Motion Correction with MCFLIRT
    moco                             = Node(interface = fsl.MCFLIRT(),   name = 'mcflirt_func')
    moco.inputs.ref_vol              = 0
    moco.inputs.save_mats            = True
    moco.inputs.save_rms             = True
    moco.inputs.save_plots           = True

    mean                             = Node(interface = fsl.MeanImage(), name = 'func_mean')
    mean.inputs.dimension            = 'T'

    flow.connect( inputnode, 'func',         moco,       'in_file'   )
    flow.connect( moco,      'out_file',     outputnode, 'moco_func'   )
    flow.connect( moco,      'rms_files',    outputnode, 'moco_rms'    )
    flow.connect( moco,      'par_file',     outputnode, 'moco_param'  )
    flow.connect( moco,      'mat_file',     outputnode, 'moco_mat'    )
    flow.connect( moco,      'out_file',     mean,       'in_file'   )
    flow.connect( mean,      'out_file',     outputnode, 'moco_mean'   )

    return flow
