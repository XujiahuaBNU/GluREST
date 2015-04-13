__author__ = 'kanaan' '26.11.2014'


from nipype.pipeline.engine import Workflow, Node, MapNode
import nipype.interfaces.utility as util
from nipype.interfaces.afni import preprocess
import nipype.interfaces.fsl as fsl
from nipype.interfaces.ants import N4BiasFieldCorrection

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

    remove_trs                             = Node(interface = preprocess.Calc(),      name = 'func_drop_trs')
    remove_trs.inputs.start_idx            = 5
    remove_trs.inputs.stop_idx             = 421
    remove_trs.inputs.expr                 = 'a'
    remove_trs.inputs.outputtype           = 'NIFTI_GZ'

    func_rpi                          = Node(interface= preprocess.Resample(),   name = 'func_rpi')
    func_rpi.inputs.orientation       = 'RPI'
    func_rpi.inputs.outputtype        = 'NIFTI_GZ'

    func_deoblique                    = Node(interface=preprocess.Refit(),       name = 'func_deoblique')
    func_deoblique.inputs.deoblique   = True

    # split functional frames
    func_split                         = Node(interface = fsl.Split(), name = 'func_split')
    func_split.inputs.dimension        = 't'
    func_split.inputs.out_base_name    = 'split'

    func_biasfield                     = MapNode(interface=N4BiasFieldCorrection(), name = 'func_biasfield', iterfield = ['input_image'])

    #merge corrected frames
    func_merge                           = Node(interface = fsl.Merge(), name = 'func_biasfield_merge')
    func_merge.inputs.dimension          = 't'
    func_merge.inputs.output_type        = 'NIFTI'

    mask                               = Node(interface = preprocess.Automask(),  name = 'func_mask')
    mask.inputs.outputtype             = 'NIFTI_GZ'

    se_rpi                            = Node(interface = preprocess.Resample(),  name = 'se_rpi')
    se_rpi.inputs.orientation         = 'RPI'
    se_rpi.inputs.outputtype          = 'NIFTI_GZ'

    seinv_rpi                         = Node(interface= preprocess.Resample(),   name = 'seinv_rpi')
    seinv_rpi.inputs.orientation      = 'RPI'
    seinv_rpi.inputs.outputtype       = 'NIFTI_GZ'

    se_deoblique                      = Node(interface=preprocess.Refit(),       name = 'deoblique_func_se')
    se_deoblique.inputs.deoblique     = True

    seinv_deoblique                   = Node(interface=preprocess.Refit(),      name = 'deoblique_func_seinvpol')
    seinv_deoblique.inputs.deoblique  = True


    flow.connect(inputnode      ,   'verio_func_se'    ,  se_rpi         ,  'in_file'               )
    flow.connect(inputnode      ,   'verio_func_seinv' ,  seinv_rpi      ,  'in_file'               )
    flow.connect(se_rpi         ,   'out_file'         ,  se_deoblique   ,   'in_file'              )
    flow.connect(seinv_rpi      ,   'out_file'         ,  seinv_deoblique,   'in_file'              )

    flow.connect(inputnode      ,   'verio_func'       ,  remove_trs     ,  'in_file_a'             )
    flow.connect(remove_trs     ,   'out_file'         ,  func_rpi       ,  'in_file'               )
    flow.connect(func_rpi       ,   'out_file'         ,  func_deoblique ,  'in_file'               )
    flow.connect(func_deoblique ,   'out_file'         ,  func_split     ,  'in_file'               )
    flow.connect(func_split     ,   'out_files'        ,  func_biasfield ,  'input_image'           )
    flow.connect(func_biasfield ,   'output_image'     ,  func_merge     ,  'in_files'              )
    flow.connect(func_merge     ,   'merged_file'      ,  mask           ,  'in_file'               )

    flow.connect(se_deoblique   ,   'out_file'         ,  outputnode     ,   'analyze_func_se'      )
    flow.connect(seinv_deoblique,   'out_file'         ,  outputnode     ,   'analyze_func_seinv'   )
    flow.connect(mask           ,   'out_file'         ,  outputnode     ,   'func_mask'            )
    flow.connect(mask           ,   'brain_file'       ,  outputnode     ,   'analyze_func'         )

    return flow

def func_preprocess():

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
    flow        = Workflow(name='func2mni_preprocessed')
    inputnode   = Node(util.IdentityInterface(fields=['func_in']),
                           name='inputnode')
    outputnode  = Node(util.IdentityInterface(fields=[#'func_mask'
                                                      #'func_deskull',
                                                      'func_preproc',
                                                      'func_preproc_mean',
                                                      'func_preproc_mask']),
                           name = 'outputnode')


    # 2- Normalize the image intensity values.
    norm                               = Node(interface = fsl.ImageMaths(),       name = 'func_preprocessed')
    norm.inputs.op_string              = '-ing 1000'
    norm.out_data_type                 = 'float'
    norm.output_type                   = 'NIFTI'

    # 3- Calculate Mean of Skull stripped image
    norm_mean                          = Node(interface = preprocess.TStat(),     name = 'func_preprocessed_mean')
    norm_mean.inputs.options           = '-mean'
    norm_mean.inputs.outputtype        = 'NIFTI'

    # 4- Create brain mask from Normalized data.
    norm_mask                          = Node(interface = fsl.ImageMaths(),       name = 'func_preprocessed_mask')
    norm_mask.inputs.op_string         = '-Tmin -bin'
    norm_mask.out_data_type            = 'char'

    #connecting workflow
    # flow.connect( inputnode  ,   'func_in'           ,   mask,        'in_file'     )
    # flow.connect( mask       ,   'out_file'          ,   strip,       'in_file_a'   )
    # flow.connect( inputnode  ,   'func_in'           ,   strip,       'in_file_b'   )
    # flow.connect( strip      ,   'out_file'          ,   norm,        'in_file'     )
    # flow.connect( mask       ,   'out_file'          ,   outputnode,  'func_mask'   )
    # flow.connect( strip      ,   'out_file'          ,   outputnode,  'func_deskull')


    flow.connect( inputnode  ,   'func_in'           ,   norm,        'in_file'     )
    flow.connect( norm       ,   'out_file'          ,   norm_mean,   'in_file'     )
    flow.connect( norm_mean  ,   'out_file'          ,   norm_mask,   'in_file'     )
    flow.connect( norm       ,   'out_file'          ,   outputnode,  'func_preproc')
    flow.connect( norm_mean  ,   'out_file'          ,   outputnode,  'func_preproc_mean')
    flow.connect( norm_mask  ,   'out_file'          ,   outputnode,  'func_preproc_mask')

    return flow
