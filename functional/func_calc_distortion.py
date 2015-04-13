__author__ = 'kanaan'

from   nipype.pipeline.engine import Workflow, Node
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl

def func_calc_disco_warp():
    '''
    Method to calculate and apply susceptibility induced distortion warps to a fresh functional image
    used topup and applytopup

    inputs
        inputnode.func
        inputnode.se_image
        inputnode.se_invpol
        inputnode.encoding_file

    outputnode
          outputnode.out_enc_file       # encoding directions file output for applytopup
          outputnode.movpar             # movpar.txt output file
          outputnode.fieldcoef          # file containing the field coefficients

    '''
    datain     = '/scr/sambesi1/workspace/Projects/GTS/REST/functional/datain.txt'

    #define worflow
    flow       = Workflow('func_distortion_correction')
    inputnode  = Node(util.IdentityInterface(fields =['func',
                                                      'func_se',
                                                      'func_se_inv']), name ='inputnode')
    outputnode = Node(util.IdentityInterface(fields = ['enc_file ',
                                                       'topup_movpar',
                                                       'topup_fieldcoef',
                                                       'topup_field',
                                                       'func_disco']), name = 'outputnode')
    # define nodes
    list_blips                    =  Node(util.Merge(2),               name = 'blips_list')
    make_blips                    = Node(fsl.Merge(),                  name = 'blips_merged')
    make_blips.inputs.dimension   = 't'
    make_blips.inputs.output_type = 'NIFTI_GZ'

    topup                         = Node(interface= fsl.TOPUP(),       name = 'calc_topup')
    topup.inputs.encoding_file    = datain
    topup.inputs.output_type      = "NIFTI_GZ"

    apply                         = Node(interface=fsl.ApplyTOPUP(),   name = 'apply_topup')
    apply.inputs.in_index         = [1]
    apply.inputs.encoding_file    = datain
    apply.inputs.method           = 'jac'
    apply.inputs.output_type      = "NIFTI_GZ"

    # connect nodes
    flow.connect(inputnode    , 'func_se'         , list_blips     , 'in1'                )
    flow.connect(inputnode    , 'func_se_inv'     , list_blips     , 'in2'                )
    flow.connect(list_blips   , 'out'             , make_blips     , 'in_files'           )
    flow.connect(make_blips   , 'merged_file'     , topup          , 'in_file'            )
    flow.connect(inputnode    , 'func'            , apply          , 'in_files'           )
    flow.connect(topup        , 'out_fieldcoef'   , apply          , 'in_topup_fieldcoef' )
    flow.connect(topup        , 'out_movpar'      , apply          , 'in_topup_movpar'    )

    flow.connect(topup        , 'out_enc_file'   , outputnode     , 'enc_file'            )
    flow.connect(topup        , 'out_movpar'     , outputnode     , 'topup_movpar'        )
    flow.connect(topup        , 'out_fieldcoef'  , outputnode     , 'topup_fieldcoef'     )
    flow.connect(topup        , 'out_field'      , outputnode     , 'topup_field'         )
    flow.connect(apply        , 'out_corrected'  , outputnode     , 'func_disco'          )

    return flow

