__author__ = 'kanaan' 'Dec-17-2014'

from motion_metrics import  calc_DVARS, calc_FD_power, calc_power_motion_params, calc_frames_included, calc_frames_excluded
from   nipype.pipeline.engine import Workflow, Node
import nipype.interfaces.utility as util
from nipype.interfaces.afni import preprocess

# heavily if not totally based on CPAC038
# see https://github.com/FCP-INDI/C-PAC

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

    return flow

def generate_motion_statistics():
    '''
    Workflow to calculate various motion metrics and save them into txt and npy files
    based on CPAC-0.3.8

    inputs
        inputnpode.subject_id                string
        inputnpode.func_preprocessed         string nifti
        inputnpode.func_mask                 string nifti
        inputnpode.mov_par                   string mat
        inputnpode.max_disp_par              string mat

    outputs
        outputnode.FD_power
        outputnode.frames_excluded
        outputnode.frames_included
        outputnode.power_params
    '''

    flow       = Workflow('func_motion_statistics')
    inputnode  = Node(util.IdentityInterface( fields = ['subject_id',
                                                        'mov_par',
                                                        'func_preproc',
                                                        'func_mask']),
                      name = 'inputnode')
    outputnode = Node(util.IdentityInterface( fields = ['FD_power',
                                                        'frames_excluded',
                                                        'frames_included',
                                                        'power_params']),
                      name = 'outputnode')

    calc_dvars = Node(util.Function(input_names  = ['rest', 'mask'],
                                    output_names = ['dvars_out'],
                                    function     = calc_DVARS),                     name = 'calc_DVARS')

    calc_fd    = Node(util.Function(input_names  = ['motion_pars'],
                                    output_names = ['fd_out'],
                                    function     = calc_FD_power),                  name = 'calc_FD_Power')

    frames_ex  = Node(util.Function(input_names  = ['fd_1d'],
                                    output_names = ['out_file'],
                                    function     = calc_frames_excluded),           name = 'calc_frames_excluded')

    frames_in  = Node(util.Function(input_names  = ['fd_1d', 'exclude_list'],
                                    output_names = ['out_file'],
                                    function     = calc_frames_included),           name = 'calc_frames_included')

    power_par  = Node(util.Function(input_names  = ['subject_id','fd_1d', 'DVARS'],
                                    output_names = ['out_file'],
                                    function     = calc_power_motion_params),       name='calc_power_motion_parames')

    flow.connect(inputnode,       'func_preproc' ,      calc_dvars,    'rest'          )
    flow.connect(inputnode,       'func_mask'    ,      calc_dvars,    'mask'          )
    flow.connect(inputnode,       'mov_par'      ,      calc_fd,       'motion_pars'   )
    flow.connect(calc_fd,         'fd_out'       ,      frames_ex,     'fd_1d'         )
    flow.connect(calc_fd,         'fd_out'       ,      frames_in,     'fd_1d'         )
    flow.connect(frames_ex,       'out_file'     ,      frames_in,     'exclude_list'  )
    flow.connect(inputnode,       'subject_id'   ,      power_par,     'subject_id'    )
    flow.connect(calc_fd,         'fd_out'       ,      power_par,     'fd_1d'         )
    flow.connect(calc_dvars,      'dvars_out'    ,      power_par,     'DVARS'         )

    flow.connect(calc_fd,         'fd_out'       ,      outputnode,  'FD_power'        )
    flow.connect(frames_ex,       'out_file'     ,      outputnode,  'frames_excluded' )
    flow.connect(frames_in,       'out_file'     ,      outputnode,  'frames_included' )
    flow.connect(power_par,       'out_file'     ,      outputnode,  'power_params'    )


    return flow

def calc_friston_twenty_four():
    '''
    Worflow/Method to calculate 24 motion parameters based on (Friston, 1996)
    Implementation of CPAC-0.3.8

    Friston, K. J., Williams, S., Howard, R., Frackowiak, R. S., & Turner, R. (1996).
          Movement-related effects in fMRI time-series. Magnetic Resonance in Medicine, 35(3),346-355

    inputs
        inputnode.movement_params
    outputs
        outputnode.movement_friston

    '''

    def calc_friston(mov_par):
        import numpy as np
        import os
        twenty_four   = None

        six           = np.genfromtxt(mov_par)
        six_squared   = six**2

        twenty_four   = np.concatenate((six,six_squared), axis=1)

        six_roll      = np.roll(six, 1, axis=0)
        six_roll[0]   = 0

        twenty_four   = np.concatenate((twenty_four, six_roll), axis=1)

        six_roll_squ  = six_roll**2

        twenty_four   = np.concatenate((twenty_four, six_roll_squ), axis=1)
        updated_mov   = os.path.join(os.getcwd(), 'fristons_twenty_four.1D')
        np.savetxt(updated_mov, twenty_four, fmt='%0.8f', delimiter=' ')

        return updated_mov


    flow        = Workflow('func_motion_friston_par')
    inputnode   = Node(util.IdentityInterface(fields =['mov_par', 'frames_excluded'])    ,      name = 'inputnode')
    outputnode  = Node(util.IdentityInterface(fields =['friston_par', 'friston_par_spikereg']), name = 'outputnode')

    calc        = Node(util.Function(input_names  =['mov_par'],
                                     output_names =['friston_par'],
                                     function     = calc_friston),
                                     name         = 'calc_friston')

    flow.connect(inputnode , 'mov_par'     , calc      ,  'mov_par')
    flow.connect(calc      , 'friston_par' , outputnode,  'friston_par')



    # def combine_motion_parameters_with_outliers(motion_params, outliers_file, spike_reg=True):
    #     """Adapted from rom https://github.com/nipy/nipype/blob/master/examples/
    #     rsfmri_vol_surface_preprocessing_nipy.py
    #     """
    #
    #     import numpy as np
    #     import os
    #     if spike_reg:
    #         out_params = np.genfromtxt(motion_params)
    #         try:
    #             outlier_val = np.nan_to_num(np.genfromtxt(outliers_file, delimiter=','))
    #             outlier_val = outlier_val.astype(int)
    #             outlier_val = np.delete(outlier_val, -1)
    #         except IOError:
    #             outlier_val = np.empty((0))
    #         for index in np.atleast_1d(outlier_val):
    #             outlier_vector = np.zeros((out_params.shape[0], 1))
    #             outlier_vector[int(index)] = 1
    #             out_params = np.hstack((out_params, outlier_vector))
    #
    #         out_file = os.path.join(os.getcwd(), "motion_outlier_regressor.csv") #"filter_regressor%02d.txt" % idx)
    #         np.savetxt(out_file, out_params, fmt="%.8f")
    #     else:
    #         out_file=motion_params
    #
    #     return out_file
    #
    #
    # spike_reg  = Node(util.Function(input_names  =  ['motion_params', 'outliers_file'],
    #                                 output_names =  ['out_file'],
    #                                 function     =    combine_motion_parameters_with_outliers),
    #                                 name         =   'firston_par_with_spike_regressors')
    # spike_reg.inputs.spike_reg                   = True
    #
    # flow.connect(calc,              'friston_par'       ,      spike_reg,   'motion_params'        )
    # flow.connect(inputnode,         'frames_excluded'   ,      spike_reg,   'outliers_file'        )
    # flow.connect(spike_reg,         'out_file'          ,      outputnode,  'friston_par_spikereg' )
    #

    return flow


