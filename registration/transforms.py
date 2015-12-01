# coding=utf-8
____author__ = 'kanaan' '26.11.2014'

from nipype.pipeline.engine import Workflow, Node
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype.interfaces.afni import preprocess


def anat2mni_nonlinear():

    config          = '/scr/sambesi1/workspace/Projects/GluREST/registration/T1_2_MNI152_2mm.cnf'
    mni_brain_2mm   = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'
    mni_skull_2mm   = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz'

    #mni_1mm = '/usr/share/fsl/5.0/data/standard/MNI152_T1_1mm_brain.nii.gz'
    #define workflow
    flow  = Workflow('anat2mni')

    inputnode  = Node(util.IdentityInterface(fields=['anat_image',
                                                     'anat_wm',
                                                     'anat_gm',
                                                     'anat_cm',
                                                     'anat_first']),
                     name = 'inputnode')

    outputnode = Node(util.IdentityInterface(fields=['anat2mni',
                                                     'nonlinear_warp',
                                                     'mni_mask_gm',
                                                     'mni_mask_wm',
                                                     'mni_mask_cm'  ,
                                                     'mni_mask_first',
                                                     'mni2anat_warp']),
                                    name = 'outputnode')

    flirt = Node(interface= fsl.FLIRT(), name = 'anat2mni_linear_flirt')
    flirt.inputs.cost        = 'mutualinfo'
    flirt.inputs.dof         = 12
    flirt.inputs.output_type = "NIFTI_GZ"
    flirt.inputs.reference   = mni_brain_2mm  # without skull here.. see example on fsl website.

    fnirt = Node(interface=fsl.FNIRT(), name = 'anat2mni_nonlinear_fnirt')
    fnirt.inputs.config_file        = config
    fnirt.inputs.fieldcoeff_file    = True
    fnirt.inputs.jacobian_file      = True
    fnirt.inputs.ref_file           = mni_brain_2mm # no skull

    invwarp  = Node(interface=fsl.InvWarp(), name = 'mni2anat_warp')

    warp_gm = Node(interface=fsl.ApplyWarp(), name='warp_gm')
    warp_gm.inputs.ref_file =mni_brain_2mm
    warp_wm = Node(interface=fsl.ApplyWarp(), name='warp_wm')
    warp_wm.inputs.ref_file =mni_brain_2mm
    warp_cm = Node(interface=fsl.ApplyWarp(), name='warp_csf')
    warp_cm.inputs.ref_file =mni_brain_2mm
    warp_first = Node(interface=fsl.ApplyWarp(), name='warp_first')
    warp_first.inputs.ref_file =mni_brain_2mm

    thresh_gm                    = Node(fsl.Threshold(), name= 'mni_mask_gm')
    thresh_gm.inputs.thresh      = 0.5
    thresh_gm.inputs.args        = '-bin'

    thresh_wm                    = Node(fsl.Threshold(), name= 'mni_mask_wm')
    thresh_wm.inputs.thresh      = 0.6
    thresh_wm.inputs.args        = '-bin'

    thresh_csf                   = Node(fsl.Threshold(), name= 'mni_mask_csf')
    thresh_csf.inputs.thresh     = 0.6
    thresh_csf.inputs.args       = '-bin'

    thresh_first                   = Node(fsl.Threshold(), name= 'mni_mask_first')
    thresh_first.inputs.thresh     = 0.5
    thresh_first.inputs.args       = '-bin'

    flow.connect(inputnode, 'anat_image'       , flirt,      'in_file'        )
    flow.connect(inputnode, 'anat_image'       , fnirt,      'in_file'        )
    flow.connect(flirt,     'out_matrix_file'  , fnirt,      'affine_file'    )
    flow.connect(inputnode, 'anat_gm'          , warp_gm,    'in_file'        )
    flow.connect(inputnode, 'anat_wm'          , warp_wm,    'in_file'        )
    flow.connect(inputnode, 'anat_cm'          , warp_cm,    'in_file'        )
    flow.connect(inputnode, 'anat_first'       , warp_first, 'in_file'        )

    flow.connect(fnirt,     'fieldcoeff_file'  , invwarp,    'warp'           )
    flow.connect(inputnode, 'anat_image'       , invwarp,    'reference'      )

    flow.connect(fnirt,     'fieldcoeff_file'  , warp_gm,    'field_file'     )
    flow.connect(fnirt,     'fieldcoeff_file'  , warp_wm,    'field_file'     )
    flow.connect(fnirt,     'fieldcoeff_file'  , warp_cm,    'field_file'     )
    flow.connect(fnirt,     'fieldcoeff_file'  , warp_first, 'field_file'     )

    flow.connect(warp_gm,   'out_file'         , thresh_gm,  'in_file'        )
    flow.connect(warp_wm,   'out_file'         , thresh_wm,  'in_file'        )
    flow.connect(warp_cm,   'out_file'         , thresh_csf, 'in_file'        )
    flow.connect(warp_first,'out_file'         , thresh_first,'in_file'       )


    flow.connect(fnirt,     'warped_file'      , outputnode, 'anat2mni'       )
    flow.connect(invwarp,   'inverse_warp'     , outputnode, 'mni2anat_warp'  )
    flow.connect(thresh_gm, 'out_file'         , outputnode, 'mni_mask_gm'    )
    flow.connect(thresh_wm, 'out_file'         , outputnode, 'mni_mask_wm'    )
    flow.connect(thresh_csf,'out_file'         , outputnode, 'mni_mask_cm'    )
    flow.connect(thresh_first,'out_file'       , outputnode, 'mni_mask_first' )
    flow.connect(fnirt,     'fieldcoeff_file'  , outputnode, 'nonlinear_warp' )

    return flow

def func2anat_linear():

    import nipype.interfaces.fsl as fsl
    bbr_shedule = '/usr/share/fsl/5.0/etc/flirtsch/bbr.sch'

    #define workflow
    linear  = Workflow('func2anat_linear')

    inputnode  = Node(util.IdentityInterface(fields=['func_image',
                                                     'func_mask',
                                                     'reference_image',
                                                     'anat_wm',
                                                     'anat_csf',
                                                     'anat_gm',
                                                     'anat_first',]),
                     name = 'inputnode')

    outputnode = Node(util.IdentityInterface(fields=['func2anat',
                                                     'func2anat_xfm',
                                                     'anat_downsample',
                                                     'anat2func_xfm',
                                                     'anat2func',
                                                     'func_gm',
                                                     'func_wm',
                                                     'func_csf',
                                                     'func_first']),
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
    bbr.inputs.no_resample   = True

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

    anat_invxfm                = Node(interface= fsl.ApplyXfm(), name ='apply_invxfm_anat')
    anat_invxfm.inputs.apply_xfm = True

    linear.connect(anatdownsample     , 'out_file'            ,    anat_invxfm, 'in_file')
    linear.connect(inputnode          , 'func_image'          ,    anat_invxfm, 'reference')
    linear.connect(convert_xfm        , 'out_file'            ,    anat_invxfm, 'in_matrix_file')
    linear.connect(anat_invxfm        , 'out_file'            ,    outputnode, 'anat2func')

    # flirt tissue masks back to func space
    gm_invxfm                = Node(interface= fsl.ApplyXfm(), name ='apply_invxfm_gm')
    gm_invxfm.inputs.apply_xfm = True
    bin_gm =  Node(interface= fsl.Threshold(), name ='apply_invxfm_gm_bin')
    bin_gm.inputs.thresh      = 0.5
    bin_gm.inputs.args        = '-bin'
    mask_gm = Node(interface=fsl.BinaryMaths(), name='func_gm')
    mask_gm.inputs.operation = 'mul'

    linear.connect(inputnode          , 'anat_gm'             ,    gm_invxfm, 'in_file')
    linear.connect(inputnode          , 'func_image'          ,    gm_invxfm, 'reference')
    linear.connect(convert_xfm        , 'out_file'            ,    gm_invxfm, 'in_matrix_file')
    linear.connect(gm_invxfm          , 'out_file'            ,    bin_gm,    'in_file')
    linear.connect(bin_gm             , 'out_file'            ,    mask_gm,   'in_file')
    linear.connect(inputnode          , 'func_mask'           ,    mask_gm,   'operand_file')
    linear.connect(mask_gm            , 'out_file'            ,    outputnode,'func_gm')

    wm_invxfm = gm_invxfm.clone('apply_invxfm_wm')
    bin_wm    = bin_gm.clone('apply_invxfm_wm_bin')
    mask_wm   = mask_gm.clone('func_wm')

    linear.connect(inputnode          , 'anat_wm'             ,    wm_invxfm, 'in_file')
    linear.connect(inputnode          , 'func_image'          ,    wm_invxfm, 'reference')
    linear.connect(convert_xfm        , 'out_file'            ,    wm_invxfm, 'in_matrix_file')
    linear.connect(wm_invxfm          , 'out_file'            ,    bin_wm,    'in_file')
    linear.connect(bin_wm             , 'out_file'            ,    mask_wm,   'in_file')
    linear.connect(inputnode          , 'func_mask'           ,    mask_wm,   'operand_file')
    linear.connect(mask_wm            , 'out_file'            ,    outputnode,'func_wm')

    cm_invxfm  = gm_invxfm.clone('apply_invxfm_csf')
    bin_cm     = bin_gm.clone('apply_invxfm_csf_bin')
    mask_cm    = mask_gm.clone('func_csf')

    linear.connect(inputnode          , 'anat_csf'            ,    cm_invxfm, 'in_file')
    linear.connect(inputnode          , 'func_image'          ,    cm_invxfm, 'reference')
    linear.connect(convert_xfm        , 'out_file'            ,    cm_invxfm, 'in_matrix_file')
    linear.connect(cm_invxfm          , 'out_file'            ,    bin_cm,    'in_file')
    linear.connect(bin_cm             , 'out_file'            ,    mask_cm,   'in_file')
    linear.connect(inputnode          , 'func_mask'           ,    mask_cm,   'operand_file')
    linear.connect(mask_cm            , 'out_file'            ,    outputnode,'func_csf')

    first_invxfm  = gm_invxfm.clone('apply_invxfm_first')
    bin_first =  Node(interface= fsl.Threshold(), name ='apply_invxfm_first_bin')
    bin_first.inputs.thresh      = 12
    bin_first.inputs.args        = '-bin'
    mask_first    = mask_gm.clone('func_first')

    linear.connect(inputnode          , 'anat_first'          ,    first_invxfm, 'in_file')
    linear.connect(inputnode          , 'func_image'          ,    first_invxfm, 'reference')
    linear.connect(convert_xfm        , 'out_file'            ,    first_invxfm, 'in_matrix_file')
    linear.connect(first_invxfm       , 'out_file'            ,    bin_first,    'in_file')
    linear.connect(bin_first          , 'out_file'            ,    mask_first,   'in_file')
    linear.connect(inputnode          , 'func_mask'           ,    mask_first,   'operand_file')
    linear.connect(mask_first         , 'out_file'            ,    outputnode,   'func_first')

    return linear

def func2mni_wf():

    mni_skull_2mm = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz'
    mni_brain_2mm   = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'

    flow  = Workflow('func2mni_nonlinear')

    inputnode  = Node(util.IdentityInterface(fields=['func_image',
                                                     'reference_image',
                                                     'func2anat_affine',
                                                     'anat2mni_warp']),name = 'inputnode')

    outputnode = Node(util.IdentityInterface(fields=['func2mni_2mm',
                                                     'func2mni_4mm']),name = 'outputnode')

    applywarp = Node(fsl.ApplyWarp(), name = 'apply_warp',)
    applywarp.inputs.ref_file            = mni_brain_2mm

    flirt4mm = Node(fsl.FLIRT(), name = 'resample_4mm')
    flirt4mm.inputs.reference         = mni_brain_2mm
    flirt4mm.inputs.apply_isoxfm      = 4.0

    flow.connect(inputnode, 'func_image'        , applywarp,  'in_file')
    flow.connect(inputnode, 'anat2mni_warp'     , applywarp,  'field_file')
    flow.connect(inputnode, 'func2anat_affine'  , applywarp,  'premat')
    flow.connect(applywarp, 'out_file'          , flirt4mm,   'in_file')

    flow.connect(applywarp, 'out_file'          , outputnode, 'func2mni_2mm')
    flow.connect(flirt4mm,  'out_file'          , outputnode, 'func2mni_4mm')

    return flow
