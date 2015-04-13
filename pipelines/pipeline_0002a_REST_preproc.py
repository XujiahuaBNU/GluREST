__author__ = 'kanaan' '17.03.2015'

import os
from   nipype.pipeline.engine         import Workflow, Node, MapNode
import nipype.interfaces.io           as nio
import nipype.interfaces.utility      as util
import nipype.interfaces.fsl          as fsl
from anatomical.anat_preproc          import  anatomical_preprocessing
from anatomical.anat_segment          import fslfast_segment
from functional.func_preprocess       import func_equilibrate, func_preprocess
from functional.func_calc_motion      import func_calc_motion_affines
from functional.func_calc_distortion  import func_calc_disco_warp
from registration.transforms          import func2anat_linear, anat2mni_nonlinear, func2mni_convertwarp
from motion.statistics                import generate_motion_statistics, calc_friston_twenty_four
from denoise.nuisance                 import create_residuals
from denoise.wavelets                 import WaveletDespike
from denoise.ica_aroma                import ICA_AROMA


from nipype import config
cfg = dict(logging=dict(workflow_level = 'DEBUG'), execution={'remove_unnecessary_outputs': False,
                                                              'job_finished_timeout': 120,
                                                              'stop_on_first_rerun': False,
                                                              'stop_on_first_crash': True} )
config.update_config(cfg)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
===============================================================================================================
                                        Pipeline Definitions
===============================================================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#subject_dir and subject_list
data_dir             = '/xxxx/project'
freesurfer_dir       = '/xxxx/FS_SUBJECTS/'
subject_list = [ "XXXX"]


#processing dirs
working_dir          = '/xxxx/WORKING_DIR'
crash_dir            = '/xxxx/CRASH_DIR'
output_dir           = '/xxxx/OUT_DIR'
pipeline_name        = 'pipeline_xxxxx'

#standard brains
mni_brain_1mm        = '/usr/share/fsl/5.0/data/standard/MNI152_T1_1mm_brain.nii.gz'
mni_brain_2mm        = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'

#Define infosource node (to loop through the subject list) and input files
infosource           = Node(interface=util.IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = ('subject_id', subject_list)
info                 = dict(anat    =[['subject_id',  'MP2RAGE_BRAIN.nii'   ]],
                            func    =[['subject_id',  'REST.nii'            ]],
                            func_rl =[['subject_id',  'REST_SE.nii'         ]],
                            func_lr =[['subject_id',  'REST_SE_INVPOL.nii'  ]],)

#Define datasource to grab infosourced data
datasource                        = Node(interface=nio.DataGrabber(infields=['subject_id'], outfields=info.keys()),name = 'datasource')
datasource.inputs.base_directory  = data_dir
datasource.inputs.template        = "p*/%s/NIFTI/%s"
datasource.inputs.template_args   = info
datasource.inputs.sort_filelist   = True

inputnode  = Node(interface=util.IdentityInterface(fields=["subject_id", "anat", "func", "func_rl","func_lr" ]),name="inputnode")

# define workflow nodes
engine                                      = Workflow(pipeline_name)
pipeline_name                               = pipeline_name
engine.base_dir                             = working_dir
engine.config['execution']['crashdump_dir'] = crash_dir

fsl.FSLCommand.set_default_output_type("NIFTI_GZ")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
===============================================================================================================
                                       Anatomical Pre-processing
===============================================================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

'========================================='
'''     Antomical Pre-processing        '''
'========================================='
# reorient to RPI and create brain mask
anat_preproc =  anatomical_preprocessing()
engine.connect(infosource            ,  'subject_id'     ,  datasource       ,  'subject_id'                  )
engine.connect(datasource            ,  'anat'           ,  anat_preproc     ,  'inputnode.anat'              )
'========================================='
'''        Antomical Segmentation      '''
'========================================='
# Segmentation with SPM12's New Segment --- Prob tissue masks are thresholded at 0.9 and eroded once
anat_segment = fslfast_segment()
engine.connect(anat_preproc          ,  'outputnode.brain'  ,  anat_segment    ,    'inputnode.anat_preproc'  )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
===============================================================================================================
                                    Core functional Image Processing
===============================================================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'========================================='
'''      Prepare Functional Data        '''
'========================================='
# Drop first five TRs
# Deolique
# Reorient to RPI
func_drop_trs  = func_equilibrate()
engine.connect(datasource,          'func'              ,    func_drop_trs ,     'inputnode.verio_func'       )
engine.connect(datasource,          'func_rl'           ,    func_drop_trs ,     'inputnode.verio_func_se'    )
engine.connect(datasource,          'func_lr'           ,    func_drop_trs ,     'inputnode.verio_func_seinv' )

'========================================='
''' Calculate Distortion Correction Warp'''
'========================================='
# runs topup and apply_topup.

func_disco           = func_calc_disco_warp()
engine.connect(func_drop_trs,      'outputnode.analyze_func'       ,    func_disco ,  'inputnode.func'        )
engine.connect(func_drop_trs,      'outputnode.analyze_func_se'    ,    func_disco,   'inputnode.func_se'     )
engine.connect(func_drop_trs,      'outputnode.analyze_func_seinv' ,    func_disco,   'inputnode.func_se_inv' )

'========================================='
''' Calculate Motion Correction Affines '''
'========================================='
func_moco           = func_calc_motion_affines()
engine.connect(func_disco,         'outputnode.func_disco'         ,    func_moco,     'inputnode.func'       )

'========================================='
''' Calculate Native Space Linear Affine'''
'''       FUNC_2.3mm -- > ANAT_1mm      '''
'========================================='
# Calculate func -- > anat linear_xfm
# runs flirt using a two step procedure (Mutual Information and Boundary based registration )

func2anat    = func2anat_linear()
engine.connect(func_moco   ,     'outputnode.moco_mean'         ,  func2anat,  'inputnode.func_image'         )
engine.connect(anat_preproc,     'outputnode.brain'             ,  func2anat,  'inputnode.reference_image'    )
engine.connect(anat_segment,     'outputnode.anat_mask_wm'      ,  func2anat,  'inputnode.anat_wm'            )

'========================================='
'''  Calculate Normalization Warp Field '''
'''       ANAT_1mm  -- > MNI152_2mm     '''
'========================================='
# run fnirt
anat2mni    = anat2mni_nonlinear()
engine.connect(anat_preproc   ,     'outputnode.brain'         ,  anat2mni,  'inputnode.anat_image'           )
engine.connect(anat_segment   ,     'outputnode.anat_mask_gm'  ,  anat2mni,  'inputnode.anat_gm'              )
engine.connect(anat_segment   ,     'outputnode.anat_mask_wm'  ,  anat2mni,  'inputnode.anat_wm'              )
engine.connect(anat_segment   ,     'outputnode.anat_mask_csf' ,  anat2mni,  'inputnode.anat_cm'              )

'========================================='
''' Apply Unifed Normalization transform'''
'''    FUNC_2.3mm  -- > MNI152_2mm      '''
'========================================='
#Creates unified warp for every volume from:
#   (a) distortion voxel shift map
#   (b) motion correction affines
#   (c) Linear func2anat affine
#   (c) nonLinear anat2mni affine
#Apples warp image per volume to a fresh functional image
func_transform = func2mni_convertwarp()
engine.connect(func2anat,          'outputnode.func2anat_xfm'  ,  func_transform,  'inputnode.linear_xfm'     )
engine.connect(func_moco,          'outputnode.moco_mat'       ,  func_transform,  'inputnode.moco_mats'      )
engine.connect(func_disco,         'outputnode.topup_field'    ,  func_transform,  'inputnode.topup_field'    )
engine.connect(anat2mni,           'outputnode.nonlinear_warp' ,  func_transform,  'inputnode.nonlinear_warp' )
engine.connect(func_drop_trs,      'outputnode.analyze_func'   ,  func_transform,  'inputnode.fresh_func'     )

'========================================='
'''    Preprocess transformed func      '''
'========================================='
# Skullstripping
# Intensity normalization to mode 1000.
# Calculates mean image
# Creates brain mask from Normalized data.
func_preproc =  func_preprocess()
engine.connect(func_transform,  'outputnode.func2mni_allwarps',   func_preproc,  'inputnode.func_in'          )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
===============================================================================================================
                                              Motion Statistics
===============================================================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

motion_stats =  generate_motion_statistics()
engine.connect(infosource       ,  'subject_id'                    ,  motion_stats,   'inputnode.subject_id'  )
engine.connect(func_moco        ,  'outputnode.moco_param'         ,  motion_stats,   'inputnode.mov_par'     )
engine.connect(func_preproc     ,  'outputnode.func_preproc'       ,  motion_stats,   'inputnode.func_preproc')
engine.connect(func_preproc     ,  'outputnode.func_preproc_mask'  ,  motion_stats,   'inputnode.func_mask'   )

# compute motion derivatives (Friston24)
friston24 = calc_friston_twenty_four()
engine.connect(func_moco        ,  'outputnode.moco_param'         ,  friston24,      'inputnode.mov_par'     )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
===============================================================================================================
                                            Denoising Timeseries
===============================================================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

'========================================='
''' Type 1: Nuisance Signal Regression Only '''
'========================================='
nuisance1                                      = create_residuals('func_nuisance_only')
nuisance1.inputs.inputnode.compcor_ncomponents = 5
engine.connect(func_preproc,   'outputnode.func_preproc'     ,      nuisance1,         'inputnode.func'       )
engine.connect(anat2mni,       'outputnode.mni_mask_gm'      ,      nuisance1,         'inputnode.gm_mask'    )
engine.connect(anat2mni,       'outputnode.mni_mask_wm'      ,      nuisance1,         'inputnode.wm_mask'    )
engine.connect(anat2mni,       'outputnode.mni_mask_cm'      ,      nuisance1,         'inputnode.csf_mask'   )
engine.connect(friston24,      'outputnode.friston_par'      ,      nuisance1,         'inputnode.motion_pars')
engine.connect(infosource,     'subject_id'                  ,      nuisance1,         'inputnode.subject_id' )


'========================================='
''' Type 2:  Wavelet despike + Nuisance'''
'========================================='
wavelet = WaveletDespike()   #### Dont kill process midway or connectin wont be made on the next run. Delete folder if you do. 
engine.connect(func_preproc,   'outputnode.func_preproc'     ,      wavelet,         'inputnode.func_mni'     )
nuisance2                                      = create_residuals('func_nuisance_wavelet')
nuisance2.inputs.inputnode.compcor_ncomponents = 5
engine.connect(wavelet,        'outputnode.despiked_img'     ,      nuisance2,         'inputnode.func'       )
engine.connect(anat2mni,       'outputnode.mni_mask_gm'      ,      nuisance2,         'inputnode.gm_mask'    )
engine.connect(anat2mni,       'outputnode.mni_mask_wm'      ,      nuisance2,         'inputnode.wm_mask'    )
engine.connect(anat2mni,       'outputnode.mni_mask_cm'      ,      nuisance2,         'inputnode.csf_mask'   )
engine.connect(friston24,      'outputnode.friston_par'      ,      nuisance2,         'inputnode.motion_pars')
engine.connect(infosource,     'subject_id'                  ,      nuisance2,         'inputnode.subject_id' )

'========================================='
''' Type 3:  ICA AROMA  + Nuisance'''
'========================================='
aroma = ICA_AROMA()         #### Dont kill process midway or connectin wont be made on the next run. Delete folder if you do.
aroma.inputs.inputnode.fslDir  = '/usr/share/fsl/5.0/'
aroma.inputs.inputnode.mask    = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
aroma.inputs.inputnode.dim     = 0
aroma.inputs.inputnode.TR      = 1.4
aroma.inputs.inputnode.denType = 'nonaggr'
engine.connect(func_preproc     ,  'outputnode.func_preproc'     ,      aroma,   'inputnode.inFile'           )
engine.connect(func_moco        ,  'outputnode.moco_param'       ,      aroma,   'inputnode.mc'               )
nuisance3                                      = create_residuals('func_nuisance_aroma')
nuisance3.inputs.inputnode.compcor_ncomponents = 5
engine.connect(aroma,          'outputnode.denoised'         ,      nuisance3,         'inputnode.func'       )
engine.connect(anat2mni,       'outputnode.mni_mask_gm'      ,      nuisance3,         'inputnode.gm_mask'    )
engine.connect(anat2mni,       'outputnode.mni_mask_wm'      ,      nuisance3,         'inputnode.wm_mask'    )
engine.connect(anat2mni,       'outputnode.mni_mask_cm'      ,      nuisance3,         'inputnode.csf_mask'   )
engine.connect(friston24,      'outputnode.friston_par'      ,      nuisance3,         'inputnode.motion_pars')
engine.connect(infosource,     'subject_id'                  ,      nuisance3,         'inputnode.subject_id' )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
===============================================================================================================
                                                Data Sink
===============================================================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
sinker = Node(nio.DataSink(base_directory = output_dir), name = 'sinker')
sinker.inputs.base_directory = os.path.join(output_dir, pipeline_name)
sinker.inputs.parameterization = False

engine.connect(infosource,          'subject_id'                  ,  sinker,  'container'                     )
#anatomical sinks
engine.connect(anat_preproc,   'outputnode.brain'            , sinker,  'anatomical_native_brain_1mm'         )
engine.connect(anat_preproc,   'outputnode.brain_mask'       , sinker,  'anatomical_native_brain_mask'        )
engine.connect(anat_segment,   'outputnode.anat_mask_gm'     , sinker,  'anatomical_native_tissue_gm'         )
engine.connect(anat_segment,   'outputnode.anat_mask_wm'     , sinker,  'anatomical_native_tissue_wm'         )
engine.connect(anat_segment,   'outputnode.anat_mask_csf'    , sinker,  'anatomical_native_tissue_csf'        )
#anatomical transforms
engine.connect(anat2mni,       'outputnode.mni_mask_gm'      , sinker,  'anatomical_MNI2mm_tissue_gm'         )
engine.connect(anat2mni,       'outputnode.mni_mask_wm'      , sinker,  'anatomical_MNI2mm_tissue_wm'         )
engine.connect(anat2mni,       'outputnode.mni_mask_cm'      , sinker,  'anatomical_MNI2mm_tissue_csf'        )
engine.connect(anat2mni,       'outputnode.anat2mni'         , sinker,  'anatomical_MNI2mm_brain'             )
#functional transforms
engine.connect(func2anat,      'outputnode.anat_downsample'  , sinker,  'anatomical_native_brain_2mm'         )
engine.connect(func2anat,      'outputnode.anat2func_xfm'    , sinker,  'anatomical_func_2mm_xfm'             )
engine.connect(func2anat,      'outputnode.func2anat'        , sinker,  'functional_anat_2mm'                 )
engine.connect(func2anat,      'outputnode.func2anat_xfm'    , sinker,  'functional_anat_2mm_xfm'             )
engine.connect(func_transform, 'outputnode.func2mni_allwarps', sinker,  'functional_MNI2mm_brain'             )
engine.connect(func_preproc,   'outputnode.func_preproc'     , sinker,  'functional_MNI2mm_brain_preproc'     )
engine.connect(func_preproc,   'outputnode.func_preproc_mean', sinker,  'functional_MNI2mm_brain_preproc_mean')
engine.connect(func_preproc,   'outputnode.func_preproc_mask', sinker,  'functional_MNI2mm_brain_preproc_mask')
#motion statistics
engine.connect(func_moco,      'outputnode.moco_param'       ,  sinker,  'functional_motion_parameters'       )
engine.connect(motion_stats,   'outputnode.FD_power'         ,  sinker,  'functional_motion_FDPower'          )
engine.connect(motion_stats,   'outputnode.frames_excluded'  ,  sinker,  'functional_motion_exclude'          )
engine.connect(motion_stats,   'outputnode.frames_included'  ,  sinker,  'functional_motion_include'          )
engine.connect(motion_stats,   'outputnode.power_params'     ,  sinker,  'functional_motion_statistics'       )
engine.connect(friston24,      'outputnode.friston_par'      ,  sinker,  'functional_motion_friston24'        )
#nuisance regeression
engine.connect(nuisance1,'outputnode.dt_res'         , sinker,'functional_MNI2mm_residual_dt'                      )
engine.connect(nuisance1,'outputnode.dt_mc_res'      , sinker,'functional_MNI2mm_residual_dt_mc'                   )
engine.connect(nuisance1,'outputnode.dt_mc_wmcsf_res', sinker,'functional_MNI2mm_residual_dt_mc_wmcsf'             )
engine.connect(nuisance1,'outputnode.dt_mc_cc_res'   , sinker,'functional_MNI2mm_residual_dt_mc_compcor'           )
engine.connect(nuisance1,'outputnode.dt_mc_cc_gs_res', sinker,'functional_MNI2mm_residual_dt_mc_compcor_gs'        )

engine.connect(nuisance2,'outputnode.dt_res'         , sinker,'functional_MNI2mm_wavelet_residual_dt'              )
engine.connect(nuisance2,'outputnode.dt_mc_res'      , sinker,'functional_MNI2mm_wavelet_residual_dt_mc'           )
engine.connect(nuisance2,'outputnode.dt_mc_wmcsf_res', sinker,'functional_MNI2mm_wavelet_residual_dt_mc_wmcsf'     )
engine.connect(nuisance2,'outputnode.dt_mc_cc_res'   , sinker,'functional_MNI2mm_wavelet_residual_dt_mc_compcor'   )
engine.connect(nuisance2,'outputnode.dt_mc_cc_gs_res', sinker,'functional_MNI2mm_wavelet_residual_dt_mc_compcor_gs')

engine.connect(nuisance3,'outputnode.dt_res'         , sinker,'functional_MNI2mm_icaroma_residual_dt'              )
engine.connect(nuisance3,'outputnode.dt_mc_res'      , sinker,'functional_MNI2mm_icaroma_residual_dt_mc'           )
engine.connect(nuisance3,'outputnode.dt_mc_wmcsf_res', sinker,'functional_MNI2mm_icaroma_residual_dt_mc_wmcsf'     )
engine.connect(nuisance3,'outputnode.dt_mc_cc_res'   , sinker,'functional_MNI2mm_icaroma_residual_dt_mc_compcor'   )
engine.connect(nuisance3,'outputnode.dt_mc_cc_gs_res', sinker,'functional_MNI2mm_icaroma_residual_dt_mc_compcor_gs')
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Run the whole workflow and produce a .dot and .png graph of the processing pipeline.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    engine.run()
    engine.write_graph(graph2use='colored', format='pdf')
