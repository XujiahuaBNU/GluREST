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

    # Run first motion correction iteration to mean volume


    #calculate mean for first run
    orig_mean                        = Node(interface=preprocess.TStat(), name='0_orig_mean')
    orig_mean.inputs.options = '-mean'
    orig_mean.inputs.outputtype = 'NIFTI_GZ'

    moco1 = Node(interface=preprocess.Volreg(), name='moco_step_1')
    moco1.inputs.args = '-Fourier -twopass'
    moco1.inputs.zpad = 4
    moco1.inputs.outputtype = 'NIFTI_GZ'

    moco1_mean                       = Node(interface=preprocess.TStat(), name='moco_step_1_mean')
    moco1_mean.inputs.options = '-mean'
    moco1_mean.inputs.outputtype = 'NIFTI_GZ'

    moco2 = Node(interface=preprocess.Volreg(), name='moco_step_2')
    moco2.inputs.args = '-Fourier -twopass'
    moco2.inputs.zpad = 4
    moco2.inputs.outputtype = 'NIFTI_GZ'

    moco2_mean                       = Node(interface=preprocess.TStat(), name='moco_step_2_mean')
    moco2_mean.inputs.options = '-mean'
    moco2_mean.inputs.outputtype = 'NIFTI_GZ'

    #iter 1
    flow.connect( inputnode,     'func',               orig_mean,    'in_file'    )
    flow.connect( inputnode,     'func',               moco1,        'in_file'    )
    flow.connect( orig_mean,     'out_file',           moco1,        'basefile'   )

    #iter 2
    flow.connect( inputnode,     'func',               moco2,        'in_file'    )
    flow.connect( moco1,         'out_file',           moco1_mean,   'in_file'    )
    flow.connect( moco1_mean,    'out_file',           moco2,        'basefile'   )
    flow.connect( moco2,         'out_file',           moco2_mean,   'in_file' )


    flow.connect( moco2,         'out_file',           outputnode,   'moco_func'  )
    flow.connect( moco2,         'oned_file',          outputnode,   'moco_param' )
    flow.connect( moco2_mean,    'out_file',           outputnode,   'moco_mean' )


    #convert AFNI 1D matrix to FSL-MCFLIRT MATRICES
    #### used later by convert warp

    def create_fsl_mats(afni_aff_1D):
        import numpy as np
        import os
        aff = np.genfromtxt(afni_aff_1D, skip_header=1)
        cur_dir = os.getcwd()
        mat_list =[]

        try:
            os.makedirs(os.path.join(cur_dir, 'MATS'))
        except OSError:
            print 'Matrix output folder already created'
        out_dir  = str(os.path.join(cur_dir, 'MATS'))

        for i, line in enumerate(aff):
            mat =  np.zeros((4, 4))
            mat[0] = line[0:4]
            mat[1] = line[4:8]
            mat[2] = line[8:12]
            mat[3] = (0,0,0,1)
            out_file  = os.path.join('%s/MAT_%s' %(out_dir, '{0:0>4}'.format(i)))
            np.savetxt( out_file, mat, delimiter = ' ', fmt="%s")

            mat_list.append(out_file)

        return mat_list

    makefslmats = Node(util.Function(input_names   = ['afni_aff_1D'],
                                     output_names  = ['mat_list'],
                                     function      = create_fsl_mats),
                                     name          = 'MATS_AFNI_to_FSL')

    flow.connect( moco2,        'oned_matrix_save',   makefslmats,  'afni_aff_1D'  )
    flow.connect( makefslmats,  'mat_list',           outputnode,   'moco_mat'     )


    # moco1                             = Node(interface = fsl.MCFLIRT(),   name = 'mcflirt_iter_1')
    # moco1.inputs.mean_vol             = True
    # moco1.inputs.save_mats            = True
    # moco1.inputs.save_rms             = True
    # moco1.inputs.save_plots           = True
    #
    # #calculate mean for first run
    # moco1_mean                        = Node(interface = fsl.MeanImage(), name = 'mcflirt_iter_1_mean')
    # moco1_mean.inputs.dimension       = 'T'
    #
    # # run second iteration of motion correction using mean of first iteration as a reference volume
    # moco2                             = Node(interface = fsl.MCFLIRT(),   name = 'mcflirt_iter_2')
    # moco2.inputs.save_mats            = True
    # moco2.inputs.save_rms             = True
    # moco2.inputs.save_plots           = True
    #
    # # get mean for second iteration of moco
    # moco2_mean                        = Node(interface = fsl.MeanImage(), name = 'mcflirt_iter_2_mean')
    # moco2_mean.inputs.dimension       = 'T'
    #
    # flow.connect( inputnode,  'func',         moco1,       'in_file'    )
    # flow.connect( moco1,      'out_file',     moco1_mean,  'in_file'    )
    #
    # flow.connect( inputnode,  'func',         moco2,       'in_file'    )
    # flow.connect( moco1_mean, 'out_file',     moco2,       'ref_file'   )
    #
    # flow.connect( moco2,      'out_file',     outputnode,  'moco_func'  )
    # flow.connect( moco2,      'rms_files',    outputnode,  'moco_rms'   )
    # flow.connect( moco2,      'par_file',     outputnode,  'moco_param' )
    # flow.connect( moco2,      'mat_file',     outputnode,  'moco_mat'   )
    #
    # flow.connect( moco2,      'out_file',     moco2_mean,  'in_file'    )
    # flow.connect( moco2_mean, 'out_file',     outputnode,  'moco_mean'  )

    return flow
