# coding=utf-8
____author__ = 'kanaan' '26.11.2014'

from nipype.pipeline.engine import Workflow, Node
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype.interfaces.afni import preprocess


def anat2mni_nonlinear():

    config          = '/usr/share/fsl/5.0/etc/flirtsch/T1_2_MNI152_2mm.cnf'
    mni_brain_2mm   = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'
    mni_skull_2mm   = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz'

    #define workflow
    flow  = Workflow('anat2mni')

    inputnode  = Node(util.IdentityInterface(fields=['anat_image',
                                                     'anat_wm',
                                                     'anat_gm',
                                                     'anat_cm']),
                     name = 'inputnode')

    outputnode = Node(util.IdentityInterface(fields=['anat2mni',
                                                     'nonlinear_warp',
                                                     'mni_mask_gm',
                                                     'mni_mask_wm',
                                                     'mni_mask_cm'  ]),
                                    name = 'outputnode')

    flirt = Node(interface= fsl.FLIRT(), name = 'anat2mni_linear_flirt')
    flirt.inputs.cost       = 'mutualinfo'
    flirt.inputs.dof        = 12
    flirt.inputs.reference  = mni_brain_2mm  # without skull here.. see example on fsl website.

    fnirt = Node(interface=fsl.FNIRT(), name = 'anat2mni_nonlinear_fnirt')
    fnirt.inputs.config_file        = config
    fnirt.inputs.fieldcoeff_file    = True
    fnirt.inputs.jacobian_file      = True
    fnirt.inputs.ref_file           = mni_brain_2mm # no skull

    warp_gm = Node(interface=fsl.ApplyWarp(), name='warp_gm')
    warp_gm.inputs.ref_file =mni_brain_2mm
    warp_wm = Node(interface=fsl.ApplyWarp(), name='warp_wm')
    warp_wm.inputs.ref_file =mni_brain_2mm
    warp_cm = Node(interface=fsl.ApplyWarp(), name='warp_csf')
    warp_cm.inputs.ref_file =mni_brain_2mm

    thresh_gm                    = Node(fsl.Threshold(), name= 'mni_mask_gm')
    thresh_gm.inputs.thresh      = 0.1
    thresh_gm.inputs.args        = '-bin'

    thresh_wm                    = Node(fsl.Threshold(), name= 'mni_mask_wm')
    thresh_wm.inputs.thresh      = 0.96
    thresh_wm.inputs.args        = '-bin'

    thresh_csf                   = Node(fsl.Threshold(), name= 'mni_mask_csf')
    thresh_csf.inputs.thresh     = 0.96
    thresh_csf.inputs.args       = '-bin'

    flow.connect(inputnode, 'anat_image'       , flirt,      'in_file'        )
    flow.connect(inputnode, 'anat_image'       , fnirt,      'in_file'        )
    flow.connect(flirt,     'out_matrix_file'  , fnirt,      'affine_file'    )
    flow.connect(inputnode, 'anat_gm'          , warp_gm,    'in_file'        )
    flow.connect(inputnode, 'anat_wm'          , warp_wm,    'in_file'        )
    flow.connect(inputnode, 'anat_cm'          , warp_cm,    'in_file'        )
    flow.connect(fnirt,     'fieldcoeff_file'  , warp_gm,    'field_file'     )
    flow.connect(fnirt,     'fieldcoeff_file'  , warp_wm,    'field_file'     )
    flow.connect(fnirt,     'fieldcoeff_file'  , warp_cm,    'field_file'     )

    flow.connect(warp_gm,   'out_file'         , thresh_gm,  'in_file'        )
    flow.connect(warp_wm,   'out_file'         , thresh_wm,  'in_file'        )
    flow.connect(warp_cm,   'out_file'         , thresh_csf, 'in_file'        )

    flow.connect(fnirt,     'warped_file'      , outputnode, 'anat2mni'    )
    flow.connect(thresh_gm, 'out_file'         , outputnode, 'mni_mask_gm'    )
    flow.connect(thresh_wm, 'out_file'         , outputnode, 'mni_mask_wm'    )
    flow.connect(thresh_csf,'out_file'         , outputnode, 'mni_mask_cm'    )
    flow.connect(fnirt,     'fieldcoeff_file'  , outputnode, 'nonlinear_warp'    )

    return flow

def func2anat_linear():
    '''
    Method to calculate linear registration matrix from a functional image to an anatomical image using FSL-FLIRT
    implements reg in two steps. mutual info and bbr
    freesurfer wm_seg used for bbr
    Inputs::
            inputspec.input_image                   (func)
            inputspec.reference_image               (anat)
            inputspec.wm_seg                        # fs tissue mask, used for bbr
            inputspec.gm_seg                        # fs tissue mask
            inputspec.csf_seg                       # fs tissue mask

    Outputs::
            outputspec.func2anat
            outputspec.func2anat_xfm_
    '''
    import nipype.interfaces.fsl as fsl
    bbr_shedule = '/usr/share/fsl/5.0/etc/flirtsch/bbr.sch'

    #define workflow
    linear  = Workflow('func2anat_linear')

    inputnode  = Node(util.IdentityInterface(fields=['func_image',
                                                     'reference_image',
                                                     'anat_wm']),
                     name = 'inputnode')

    outputnode = Node(util.IdentityInterface(fields=['func2anat',
                                                     'func2anat_xfm',
                                                     'anat_downsample',
                                                     'anat2func_xfm']),
                     name= 'outputnode')

    anatdownsample                      = Node(interface= fsl.FLIRT(), name = 'downsample_anat')
    anatdownsample.inputs.apply_isoxfm  = 2.3
    anatdownsample.inputs.datatype      = 'float'

    # run flirt with mutual info
    mutual_info              = Node(interface= fsl.FLIRT(), name = 'func2anat_flirt0_mutualinfo')
    mutual_info.inputs.cost  = 'mutualinfo'
    mutual_info.inputs.dof   = 6
    mutual_info.inputs.no_resample   = True

    # run flirt boundary based registration on a func_moco_disco using
    # (a) white matter segment as a boundary and (b) the mutualinfo xfm for initialization
    bbr                      = Node(interface= fsl.FLIRT(), name = 'func2anat_flirt1_bbr')
    bbr.inputs.cost          = 'bbr'
    bbr.inputs.dof           = 6
    bbr.inputs.schedule      = bbr_shedule
    # bbr.inputs.no_resample   = True

    convert_xfm                    = Node(interface= fsl.ConvertXFM(), name ='anat2func_xfm')
    convert_xfm.inputs.invert_xfm  = True

    #connect nodes
    linear.connect(inputnode          , 'reference_image'     ,    anatdownsample  , 'in_file'    )
    linear.connect(inputnode          , 'reference_image'     ,    anatdownsample  , 'reference'  )
    linear.connect(inputnode          , 'func_image'          ,    mutual_info     , 'in_file'    )
    linear.connect(anatdownsample     , 'out_file'            ,    mutual_info     , 'reference'  )
    linear.connect(inputnode          , 'func_image'          ,    bbr             , 'in_file'         )
    linear.connect(anatdownsample     , 'out_file'            ,    bbr             , 'reference'       )
    linear.connect(inputnode          , 'anat_wm'             ,    bbr             , 'wm_seg'          )
    linear.connect(mutual_info        , 'out_matrix_file'     ,    bbr             , 'in_matrix_file'  )
    linear.connect(bbr                , 'out_matrix_file'     ,    convert_xfm     , 'in_file'         )
    linear.connect(bbr                , 'out_file'            ,    outputnode      , 'func2anat'       )
    linear.connect(bbr                , 'out_matrix_file'     ,    outputnode      , 'func2anat_xfm'   )
    linear.connect(convert_xfm        , 'out_file'            ,    outputnode      , 'anat2func_xfm'   )
    linear.connect(anatdownsample     , 'out_file'            ,    outputnode      , 'anat_downsample' )

    return linear




def func2mni_convertwarp():
    '''
    inputs
        inputnode.fresh_func                            (moving_image..nifti)
        inputnode.movement_params                       (mcflirt affine for every volume... text files)
        inputnode.topup_mov_params                      (topup text file)
        inputnode.topup_vox_disp_map                    (topup warp)
    outputs
        outputnode.
        outputnode.
        outputnode.
        outputnode.
    '''

    epi_rt = 0.0589587878 # epi_readoutime

    import nipype.interfaces.fsl as fsl
    from nipype.pipeline.engine import Workflow, Node, MapNode

    mni_skull_2mm = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz'
    mni_brain_2mm   = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'

    flow =Workflow('func2mni_nonlinear')
    inputnode = Node(util.IdentityInterface(fields=['fresh_func',
                                                    'anat_preproc',
                                                    'moco_mats',
                                                    'linear_xfm',
                                                    'nonlinear_warp',
                                                    'topup_field',]),
                      name='inputnode')

    outputnode = Node(util.IdentityInterface(fields=['func2mni_allwarps']),
                      name = 'outputnode')

    # #get func mean
    # func_mean                             = Node(interface = fsl.MeanImage(), name = 'func_mean')
    # func_mean.inputs.dimension            = 'T'

    # split functional frames
    func_split                            = Node(interface = fsl.Split(), name = 'func_split')
    func_split.inputs.dimension           = 't'
    func_split.inputs.out_base_name       = 'split'

    #multiply voxel shift may by epi readout time
    VDMxRT                                = Node(interface= fsl.BinaryMaths(), name = 'VDMxRT')
    VDMxRT.inputs.operation               = 'mul'
    VDMxRT.inputs.operand_value           = 0.059

    # unify moco and linear regisrtation warp
    convert_xfm                           = MapNode(interface=fsl.ConvertXFM(), name = 'convert_linear_moco_xfms', iterfield = ['in_file2'])
    convert_xfm.inputs.concat_xfm         = True

    #create unified warp
    convert_warp                          = MapNode(interface = fsl.ConvertWarp(), name = 'convert_warp', iterfield = ['premat'])
    convert_warp.inputs.output_type       = "NIFTI_GZ"
    convert_warp.inputs.out_relwarp       = True
    convert_warp.inputs.shift_direction   = 'y'
    convert_warp.inputs.reference         = mni_skull_2mm

    apply_warp                            = MapNode(interface =fsl.ApplyWarp(), name = 'apply_warp', iterfield = ['in_file', 'field_file'])
    apply_warp.inputs.relwarp             = True
    apply_warp.inputs.ref_file            = mni_skull_2mm

    #merge corrected frames
    func_merge                           = Node(interface = fsl.Merge(), name = 'func2mni_uniwarp')
    func_merge.inputs.dimension          = 't'
    func_merge.inputs.output_type        = 'NIFTI'

    flow.connect(inputnode,      'fresh_func'            ,    func_split,         'in_file'              )
    flow.connect(inputnode,      'topup_field'           ,    VDMxRT,             'in_file'              )

    flow.connect(inputnode,      'linear_xfm'            ,    convert_xfm,        'in_file'              )
    flow.connect(inputnode,      'moco_mats'             ,    convert_xfm,        'in_file2'             )

    flow.connect(VDMxRT,         'out_file'              ,    convert_warp,       'shift_in_file'        )
    flow.connect(convert_xfm,    'out_file'              ,    convert_warp,       'premat'               )
    flow.connect(inputnode,      'nonlinear_warp'        ,    convert_warp,       'warp1'                )

    flow.connect(func_split,     'out_files'             ,    apply_warp,         'in_file'              )
    flow.connect(convert_warp,   'out_file'              ,    apply_warp,         'field_file'           )

    flow.connect(apply_warp,     'out_file'              ,    func_merge,         'in_files'             )
    flow.connect(func_merge,     'merged_file'           ,    outputnode,         'func2mni_allwarps'    )


    return flow






#
# def anat2func_apply_xfm():
#     '''
#     Method for registering anatomical image and tissue masks to functional space.
#     The transformation matrix from the func2anat registeration is inversed and used to go from anat to func.
#     anat2func images are then warped to mni space.
#     This way, func images can be warped to mni space in one resampling step.
#
#     ***tissue masks are used to calcualte denoise residuals in func space. residual images are then warped to mni.
#     '''
#
#
#     import nipype.interfaces.fsl as fsl
#     flow  = Workflow('anat2func_LinearTransform')
#
#     inputnode  = Node(util.IdentityInterface(fields=['anat_image',
#                                                      'anat_wm',
#                                                      'anat_gm',
#                                                      'anat_csf',
#                                                      'func_image',
#                                                      'anat2func_xfm']),
#                      name = 'inputnode')
#
#     outputnode = Node(util.IdentityInterface(fields=['anat2func_image',
#                                                      'func_gm',
#                                                      'func_wm',
#                                                      'func_csf']),
#                       name = 'outputnode')
#
#     xfm_anat2func                    = Node(interface= fsl.ApplyXfm(), name = 'anat2func')
#     xfm_anat2func.inputs.apply_xfm   = True
#     xfm_anat2func.inputs.no_resample = True
#
#     apply_xfm_w                   = Node(interface= fsl.ApplyXfm(), name = 'applyxfm_wm2func')
#     apply_xfm_w.inputs.apply_xfm  = True
#
#     apply_xfm_g                   = Node(interface= fsl.ApplyXfm(), name = 'applyxfm_gm2func')
#     apply_xfm_g.inputs.apply_xfm  = True
#
#     apply_xfm_c                   = Node(interface= fsl.ApplyXfm(), name = 'applyxfm_csf2func')
#     apply_xfm_c.inputs.apply_xfm  = True
#
#     bin_wm                        = Node(interface=fsl.Threshold(), name = 'func_wm')
#     bin_wm.inputs.thresh          = 0.9
#     bin_wm.inputs.args            = '-bin'
#
#     bin_gm                        = Node(interface=fsl.Threshold(), name = 'func_gm')
#     bin_gm.inputs.thresh          = 0.7
#     bin_gm.inputs.args            = '-bin'
#
#     bin_csf                       = Node(interface=fsl.Threshold(), name = 'func_csf')
#     bin_csf.inputs.thresh         = 0.9
#     bin_csf.inputs.args           = '-bin'
#
#
#
#     flow.connect(inputnode          , 'func_image'          ,    xfm_anat2func     , 'reference'         )
#     flow.connect(inputnode          , 'anat_image'          ,    xfm_anat2func     , 'in_file'           )
#     flow.connect(inputnode          , 'anat2func_xfm'       ,    xfm_anat2func     , 'in_matrix_file'    )
#
#     flow.connect(inputnode          , 'anat_wm'             ,    apply_xfm_w     , 'in_file'            )
#     flow.connect(inputnode          , 'func_image'          ,    apply_xfm_w     , 'reference'        )
#     flow.connect(inputnode          , 'anat2func_xfm'       ,    apply_xfm_w     , 'in_matrix_file'   )
#
#     flow.connect(inputnode          , 'anat_gm'             ,    apply_xfm_g     , 'in_file'            )
#     flow.connect(inputnode          , 'func_image'          ,    apply_xfm_g     , 'reference'        )
#     flow.connect(inputnode          , 'anat2func_xfm'       ,    apply_xfm_g     , 'in_matrix_file'   )
#
#     flow.connect(inputnode          , 'anat_csf'           ,    apply_xfm_c     , 'in_file'            )
#     flow.connect(inputnode          , 'func_image'         ,    apply_xfm_c     , 'reference'        )
#     flow.connect(inputnode          , 'anat2func_xfm'      ,    apply_xfm_c     , 'in_matrix_file'   )
#
#     flow.connect(apply_xfm_g        , 'out_file'          ,    bin_gm        , 'in_file'           )
#     flow.connect(apply_xfm_w        , 'out_file'          ,    bin_wm        , 'in_file'           )
#     flow.connect(apply_xfm_c        , 'out_file'          ,    bin_csf       , 'in_file'           )
#
#
#     flow.connect(xfm_anat2func       , 'out_file'            ,    outputnode      , 'anat2func_image'  )
#     flow.connect(bin_wm              , 'out_file'            ,    outputnode      , 'wm_func_mask'          )
#     flow.connect(bin_gm              , 'out_file'            ,    outputnode      , 'gm_func_mask'          )
#     flow.connect(bin_csf             , 'out_file'            ,    outputnode      , 'csf_func_mask'         )
#
#
#
#
#
#     # flow.connect(inputnode          , 'anat_wm'             ,    erode_wm     , 'in_file'          )
#     # flow.connect(inputnode          , 'anat_gm'             ,    erode_gm     , 'in_file'          )
#     # flow.connect(inputnode          , 'anat_csf'            ,    erode_csf     , 'in_file'          )
#
#     #
#     # flow.connect(apply_xfm          , 'out_file'            ,    outputnode      , 'anat2func_image'    )
#     # flow.connect(apply_xfm_w        , 'out_file'            ,    outputnode      , 'wm_func'            )
#     # flow.connect(apply_xfm_g        , 'out_file'            ,    outputnode      , 'gm_func'           )
#     # flow.connect(apply_xfm_c        , 'out_file'            ,    outputnode      , 'csf_func'           )
#
#     return flow
#
#
#
# def ANTS_anat2mni_calc_warpfield():
#     from nipype.pipeline.engine import Workflow, Node
#     import nipype.interfaces.utility as util
#     import nipype.interfaces.ants as ants
#
#     '''
#     Method for calculating anat-->mni registration affine and warpfield using ANTS
#     employs antsRegistration
#
#     _____
#     Notes
#     -ants performs multiple stages of calculation depending on the config
#         e.g.
#             warp_wf.inputs.inputspec.registration = ['Rigid','Affine','SyN']
#             warp_wf.inputs.inputspec.transform_parameters = [[0.1],[0.1],[0.1,3,0]]
#
#         ..where each element in the first list is a metric to be used at each
#         stage, 'Rigid' being for stage 1, 'Affine' for stage 2, etc. The lists
#         within the list for transform_parameters would then correspond to each
#         stage's metric, with [0.1] applying to 'Rigid' and 'Affine' (stages 1 and
#         2), and [0.1,3,0] applying to 'SyN' of stage 3.
#
#     usage
#     reg = calc_warpfield()
#     reg.base_dir                          =  '/SCR2/tmp/ANAT_PREPROC_TEST/working/anat_preprocess/anat_preproc/_subject_id_RB1T/tmp_brain2std'
#     reg.inputs.inputnode.anatomical_brain = ['/SCR2/tmp/ANAT_PREPROC_TEST/working/anat_preprocess/anat_preproc/_subject_id_RB1T/tmp_brain2std/outStripped_resample.nii.gz']
#     reg.inputs.inputnode.reference_brain  = ['/usr/share/fsl/5.0/data/standard/MNI152_T1_1mm_brain.nii.gz']
#     reg.run()
#     '''
#
#     # define workflow
#     flow  = Workflow('anat2mni_calc')
#
#     inputnode = Node(util.IdentityInterface(fields=[
#                 'anatomical_brain',          # Deskulled anatomical image
#                 'reference_brain',]),        # Deskulled reference image
#                 name='inputnode')
#
#     outputnode = Node(util.IdentityInterface(fields=[
#                 'selected_ants_affine',           #  Affine matrix of the registration
#                 'selected_ants_warp',             #  Inverse of Affine matrix
#                 'forward_transforms',             #  List of tranforms.. (affine, warp)
#                 'reverse_transforms',             #  Inverse of registration
#                 'composite_transform',            #  Combined registration (warp field and rigid & affine linear warps)
#                 'inverse_composite_transform',    #  Inverse of combined registration
#                 'warped_image',                   #  warp image
#                 'inverse_warped_image',
#                 'forward_xfms_list']),
#                  name='outputnode')
#
#      # defining node paramters
#     ants_anat2mni = Node(interface=ants.Registration(), name = 'anat2mni_ants')
#     ants_anat2mni.inputs.dimension = 3                                          # Dimesion of input (default is 3)
#     ants_anat2mni.inputs.use_histogram_matching = True                          # Match hists of images before reg
#     ants_anat2mni.inputs.winsorize_lower_quantile = 0.01                        # Winsorize data based on quantilies (lower  value)
#     ants_anat2mni.inputs.winsorize_upper_quantile = 0.99                        # Winsorize data based on quantilies (higher value)
#     ants_anat2mni.inputs.metric = ['MI','CC']                                   # Image metric(s) to be used at each stage
#     ants_anat2mni.inputs.metric_weight = [1,1]                                  # Modulates the per-stage weighting of the corresponding metric
#     ants_anat2mni.inputs.radius_or_number_of_bins = [32,4]                      # Number of bins in each stage for the MI and Mattes metric
#     ants_anat2mni.inputs.sampling_strategy = ['Regular',None]                   # Sampling strategy to use for the metrics {None, Regular, or Random}
#     ants_anat2mni.inputs.sampling_percentage = [0.25,0.25,None]                 # Defines the sampling strategy
#     ants_anat2mni.inputs.number_of_iterations = [[300,200,100], [50,30,20]]   #[[1000,500,250,100],[1000,500,250,100], [100,100,70,20]]  # Determines the convergence
#     ants_anat2mni.inputs.convergence_threshold = [1e-8,1e-9]                    # Threshold compared to the slope of the line fitted in convergence
#     ants_anat2mni.inputs.convergence_window_size = [10,15]                      # Window size of convergence calculations
#     ants_anat2mni.inputs.registration = ['Affine','SyN']                          # Selection of registration options. See antsRegistration documentation
#     ants_anat2mni.inputs.transform_parameters = [[0.1],[0.1,3,0]]               # Selection of registration options. See antsRegistration documentation
#     ants_anat2mni.inputs.transform_parameters = [[0.1],[0.1,3,0]]               # Fine-tuning for the different registration options
#     ants_anat2mni.inputs.shrink_factors = [[4,2,1],[4,2,1]]                     #Specify the shrink factor for the virtual domain (typically the fixed image) at each level
#     ants_anat2mni.inputs.smoothing_sigmas = [[2,1,0],[2,1,0]]                   # Specify the sigma of gaussian smoothing at each level
#
#
#     # grab transformation_series outputs separately (used later in applywarp)
#     anat2mni_forward_affine = Node(util.Function(input_names   = ['warp_list','selection'],
#                                                  output_names  = ['selected_warp'],
#                                                  function      = separate_warps_list),
#                                                  name          = 'anat2mni_forward_affine')
#     anat2mni_forward_affine.inputs.selection = 0
#
#     anat2mni_forward_warp   = Node(util.Function(input_names   = ['warp_list','selection'],
#                                                  output_names  = ['selected_warp'],
#                                                  function      = separate_warps_list),
#                                                  name          = 'anat2mni_forward_warp')
#     anat2mni_forward_warp.inputs.selection = 1
#
#     anat2mni_forward_fix    = Node(util.Function(input_names   = ['file_1', 'file_2'],
#                                                  output_names  = ['warp_list'],
#                                                  function      = join_warps_list),
#                                                  name          = 'anat2mni_forward_fix')
#
#     # connecting nodes
#     flow.connect(inputnode,               'anatomical_brain',            ants_anat2mni,           'moving_image'                )
#     flow.connect(inputnode,               'reference_brain',             ants_anat2mni,           'fixed_image'                 )
#     flow.connect(ants_anat2mni,           'forward_transforms',          anat2mni_forward_affine, 'warp_list'                      )
#     flow.connect(ants_anat2mni,           'forward_transforms',          anat2mni_forward_warp,   'warp_list'                        )
#     flow.connect(anat2mni_forward_affine, 'selected_warp',               anat2mni_forward_fix,    'file_2'                      )
#     flow.connect(anat2mni_forward_warp,   'selected_warp',               anat2mni_forward_fix,    'file_1'                      )
#     flow.connect(ants_anat2mni,           'forward_transforms',          outputnode,              'forward_transforms'          )
#     flow.connect(ants_anat2mni,           'reverse_transforms',          outputnode,              'reverse_transforms'          )
#     flow.connect(ants_anat2mni,           'composite_transform',         outputnode,              'composite_transform'         )
#     flow.connect(ants_anat2mni,           'inverse_composite_transform', outputnode,              'inverse_composite_transform' )
#     flow.connect(ants_anat2mni,           'warped_image',                outputnode,              'warped_image'                )
#     flow.connect(ants_anat2mni,           'inverse_warped_image',        outputnode,              'inverse_warped_image'        )
#     flow.connect(anat2mni_forward_fix,    'warp_list',                   outputnode,              'forward_xfms_list'           )
#
#     return flow
#
#
#
# def ANTS_anat2mni_apply_warpfield():
#     """
#     Method to apply calculated affine and warp field onto an input image.
#     employs WarpImageMultiTransform
#
#     Inputs::
#             inputspec.input_image
#             inputspec.reference_image
#             inputspec.transformation_series          (list of file_paths)
#             #inputsoec.dimension                     (dim of image being registered.. 2, 3 or 4)
#
#     Outputs::
#             outputspec.output_image     (warped brain)
#
#     """
#     import nipype.interfaces.ants as ants
#
#     flow_apply_warp = Workflow('anat2mni_warp')
#
#     inputnode       = Node(util.IdentityInterface(fields = ['input_image',
#                                                             'reference_image',
#                                                             'transformation_series']),
#                                name = 'inputnode')
#     outputnode      = Node(util.IdentityInterface(fields=  ['output_image']),
#                                name = 'outputnode')
#
#     apply_ants_warp = Node(interface=ants.WarpImageMultiTransform(),
#                                name = 'apply_warp')
#
#
#     # Connect nodes
#     flow_apply_warp.connect( inputnode,    'input_image',               apply_ants_warp, 'input_image')
#     flow_apply_warp.connect( inputnode,    'reference_image',           apply_ants_warp, 'reference_image')
#     flow_apply_warp.connect( inputnode,    'transformation_series',     apply_ants_warp, 'transformation_series')
#
#     # connections to outputspec
#     flow_apply_warp.connect(apply_ants_warp, 'output_image', outputnode, 'output_image')
#
#     return flow_apply_warp
#
# def func2mni_apply_warpfield():
#     from nipype.pipeline.engine import Workflow, Node, MapNode
#     import nipype.interfaces.ants as ants
#     import nipype.interfaces.fsl as fsl
#
#     flow = Workflow('func2mni_warp')
#
#     inputnode       = Node(util.IdentityInterface(fields = ['input_image',
#                                                             'reference_image',
#                                                             'transformation_series']),
#                                name = 'inputnode')
#     outputnode      = Node(util.IdentityInterface(fields=  ['func2mni']),
#                                name = 'outputnode')
#
#     # split functional frames
#     func_split                            = Node(interface = fsl.Split(), name = 'func_split')
#     func_split.inputs.dimension           = 't'
#     func_split.inputs.out_base_name       = 'split'
#
#
#     apply_ants_warp                       = MapNode(interface=ants.WarpImageMultiTransform(),  name = 'apply_warp', iterfield = ['input_image'])
#
#     #merge corrected frames
#     func_merge                            = Node(interface = fsl.Merge(), name = 'func_merge')
#     func_merge.inputs.dimension           = 't'
#     func_merge.inputs.output_type         = 'NIFTI_GZ'
#
#     # Connect nodes
#     flow.connect( inputnode,        'input_image',               func_split,      'in_file')
#     flow.connect( func_split,       'out_files',                 apply_ants_warp, 'input_image')
#     flow.connect( inputnode,        'reference_image',           apply_ants_warp, 'reference_image')
#     flow.connect( inputnode,        'transformation_series',     apply_ants_warp, 'transformation_series')
#     flow.connect(apply_ants_warp,   'output_image',              func_merge,      'in_files'           )
#     flow.connect(func_merge,        'merged_file',               outputnode,      'func2mni'   )
#
#     return flow
#
#
# def separate_warps_list(warp_list, selection):
#     '''
#     Simple function to grab specific files from ants transformations list
#     '''
#
#     return warp_list[selection]
#
# def join_warps_list(file_1,file_2):
#     '''
#     Simple function to reorder the ants transformation list
#     ... original WarpImageMultiTransform transformation_series outputs [affine, warp]
#     ... This is needed since WarpImageMultiTransform needs [warp,affine]
#     '''
#     x = []
#     x.append(file_1)
#     x.append(file_2)
#     return  x
#
#
#
