__author__ = 'kanaan' '26.11.2014'


from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.afni import preprocess
from nipype.interfaces.ants import N4BiasFieldCorrection
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl

def func_equilibrate():
    '''
    Workflow to get the scanner data ready.
    Anatomical and functional images are deobliqued.
    5 TRs are removed from func data.

    inputs
        inputnode.verio_anat
        inputnode.verio_func
        inputnode.verio_func_se
        inputnode.verio_func_se_inv
    outputs
        outputnode.analyze_anat
        outputnode.analyze_func
        outputnode.analyze_func_se
        outputnode.analyze_func_se_inv
    '''

    flow        = Workflow('func_equilibrate')
    inputnode   = Node(util.IdentityInterface(fields= ['verio_func',
                                                       'verio_func_se',
                                                       'verio_func_seinv']),
                       name = 'inputnode')
    outputnode  = Node(util.IdentityInterface(fields= ['analyze_func',
                                                       'func_mask',
                                                       'analyze_func_se',
                                                       'analyze_func_seinv']),
                       name = 'outputnode')

    ## functional image

    # 1. remove TRS
    remove_trs                             = Node(interface = preprocess.Calc(),      name = 'func_drop_trs')
    remove_trs.inputs.start_idx            = 5
    remove_trs.inputs.stop_idx             = 421
    remove_trs.inputs.expr                 = 'a'
    remove_trs.inputs.outputtype           = 'NIFTI_GZ'

    # 2. to RPI
    func_rpi                          = Node(interface= preprocess.Resample(),   name = 'func_rpi')
    func_rpi.inputs.orientation       = 'RPI'
    func_rpi.inputs.outputtype        = 'NIFTI_GZ'

    # 3. func deoblique
    func_deoblique                    = Node(interface=preprocess.Refit(),       name = 'func_deoblique')
    func_deoblique.inputs.deoblique   = True

    flow.connect(inputnode           ,   'verio_func'       ,  remove_trs          ,  'in_file_a'             )
    flow.connect(remove_trs          ,   'out_file'         ,  func_rpi            ,  'in_file'               )
    flow.connect(func_rpi            ,   'out_file'         ,  func_deoblique      ,  'in_file'               )
    flow.connect(func_deoblique      ,   'out_file'         ,  outputnode          ,  'analyze_func'          )


    ###########################################################################################################
    ###########################################################################################################
    # se to RPI
    se_rpi                          = Node(interface= preprocess.Resample(),   name = 'se_rpi')
    se_rpi.inputs.orientation       = 'RPI'
    se_rpi.inputs.outputtype        = 'NIFTI_GZ'

    # 3. func deoblique
    se_deoblique                    = Node(interface=preprocess.Refit(),       name = 'se_deoblique')
    se_deoblique.inputs.deoblique   = True


    flow.connect(inputnode         ,   'verio_func_se'    ,  se_rpi            ,  'in_file'               )
    flow.connect(se_rpi            ,   'out_file'         ,  se_deoblique      ,  'in_file'               )

    flow.connect(se_deoblique      ,   'out_file'         ,  outputnode          ,  'analyze_func_se'          )

    ###########################################################################################################
    ###########################################################################################################
    ###########################################################################################################

    # se_inv to RPI
    se_inv_rpi                          = Node(interface= preprocess.Resample(),   name = 'seinv_rpi')
    se_inv_rpi.inputs.orientation       = 'RPI'
    se_inv_rpi.inputs.outputtype        = 'NIFTI_GZ'

    # 3. func deoblique
    se_inv_deoblique                    = Node(interface=preprocess.Refit(),       name = 'seinv_deoblique')
    se_inv_deoblique.inputs.deoblique   = True



    flow.connect(inputnode            ,   'verio_func_seinv'     ,  se_inv_rpi            ,  'in_file'               )
    flow.connect(se_inv_rpi           ,   'out_file'             ,  se_inv_deoblique      ,  'in_file'               )
    flow.connect(se_inv_deoblique     ,   'out_file'             ,  outputnode            ,  'analyze_func_seinv'          )

    return flow



def func_preprocess(name = 'func_preproc'):

    '''
    Method to preprocess functional data after warping to anatomical space.

    Accomplished after one step Distortion Correction, Motion Correction and Boundary based linear registration to
    anatomical space.

    Precodure includes:
    # 1- skull strip
    # 2- Normalize the image intensity values.
    # 3- Calculate Mean of Skull stripped image
    # 4- Create brain mask from Normalized data.
    '''

    # Define Workflow
    flow        = Workflow(name=name)
    inputnode   = Node(util.IdentityInterface(fields=['func_in']),
                           name='inputnode')
    outputnode  = Node(util.IdentityInterface(fields=['func_preproc',
                                                      'func_preproc_mean',
                                                      'func_preproc_mask']),
                           name = 'outputnode')


    # 2- Normalize the image intensity values.
    norm                               = Node(interface = fsl.ImageMaths(),       name = 'func_normalized')
    norm.inputs.op_string              = '-ing 1000'
    norm.out_data_type                 = 'float'
    norm.output_type                   = 'NIFTI'

    # 4- Create brain mask from Normalized data.
    mask                               = Node(interface = fsl.BET(),  name = 'func_preprocessed')
    mask.inputs.functional             = True
    mask.inputs.mask                   = True
    mask.inputs.frac                   = 0.5
    mask.inputs.vertical_gradient      = 0
    mask.inputs.threshold              = True

    # 3- Calculate Mean of Skull stripped image
    mean                          = Node(interface = preprocess.TStat(),     name = 'func_preprocessed_mean')
    mean.inputs.options           = '-mean'
    mean.inputs.outputtype        = 'NIFTI'


    flow.connect( inputnode  ,   'func_in'           ,   norm,        'in_file'     )
    flow.connect( norm       ,   'out_file'          ,   mask,        'in_file'     )
    flow.connect( norm       ,   'out_file'          ,   mean,        'in_file'     )
    flow.connect( mask       ,   'out_file'          ,   outputnode,  'func_preproc')
    flow.connect( mask       ,   'mask_file'         ,   outputnode,  'func_preproc_mask')
    flow.connect( mean       ,   'out_file'          ,   outputnode,  'func_preproc_mean')

    return flow


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
    datain     = '/scr/sambesi1/workspace/Projects/GluREST/functional/datain.txt'

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
    make_blips                    =  Node(fsl.Merge(),                 name = 'blips_merged')
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

