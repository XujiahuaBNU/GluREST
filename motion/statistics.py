__author__ = 'kanaan' 'Dec-17-2014'

from metrics import  calc_DVARS, calc_FD_power, calc_power_motion_params, calc_frames_included, calc_frames_excluded
from nipype.pipeline.engine import Workflow, Node
import nipype.interfaces.utility as util

# based on CPAC038
# see https://github.com/FCP-INDI/C-PAC

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
    inputnode   = Node(util.IdentityInterface(fields =['mov_par'])    , name = 'inputnode')
    outputnode  = Node(util.IdentityInterface(fields =['friston_par']), name = 'outputnode')

    calc        = Node(util.Function(input_names  =['mov_par'],
                                     output_names =['friston_par'],
                                     function     = calc_friston),
                                     name         = 'calc_friston')

    flow.connect(inputnode , 'mov_par'     , calc      ,  'mov_par')
    flow.connect(calc      , 'friston_par' , outputnode,  'friston_par')

    return flow


