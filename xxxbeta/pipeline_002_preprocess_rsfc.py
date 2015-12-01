__author__ = 'kanaan' '11.08.2015'

import os

from   nipype.pipeline.engine         import Workflow, Node
import nipype.interfaces.io           as nio
import nipype.interfaces.utility      as util
import nipype.interfaces.fsl          as fsl

from variables.subject_list           import working_dir, crash_dir, output_dir, study_a_list
from anatomical.anat_preprocess       import anatomical_preprocessing
from functional.func_preprocess       import func_equilibrate, func_calc_disco_warp, func_preprocess
from motion.motion_calc               import func_calc_motion_affines, generate_motion_statistics, calc_friston_twenty_four
from registration.transforms          import func2anat_linear, anat2mni_nonlinear, func2mni_convertwarp
from denoise.nuisance                 import smooth_data, create_residuals_wf
from xxxbeta.ica_aroma_native import ICA_AROMA
from denoise.bandpass                 import bandpas_voxels_workflow

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                        Pipeline Definitions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#input data
data_dir             = '/scr/sambesi2/TS_EUROTRAIN/VERIO/nmr093a/'
subject_list         = study_a_list

#Define infosource node to loop through the subject list and input files
infosource           = Node(interface=util.IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = ('subject_id', subject_list)
info                 = dict(anat       =[['subject_id',  'MP2RAGE_DESKULL_RPI.nii.gz'          ]],
                            anat_gm    =[['subject_id',  'TISSUE_CLASS_1_GM_OPTIMIZED.nii.gz'  ]],
                            anat_wm    =[['subject_id',  'TISSUE_CLASS_2_WM_OPTIMIZED.nii.gz'  ]],
                            anat_csf   =[['subject_id',  'TISSUE_CLASS_3_CSF_OPTIMIZED.nii.gz' ]],
                            anat_first =[['subject_id',  'FIRST_4d.nii.gz' ]],
                            func       =[['subject_id',  'REST.nii'            ]],
                            func_rl    =[['subject_id',  'REST_SE.nii'         ]],
                            func_lr    =[['subject_id',  'REST_SE_INVPOL.nii'  ]])

#Grab Scanner data
datasource                        = Node(interface=nio.DataGrabber(
                                         infields=['subject_id'], outfields=info.keys()),name = 'datasource')
datasource.inputs.base_directory  = data_dir
datasource.inputs.template        = "p*/%s/NIFTI/%s"
datasource.inputs.template_args   = info
datasource.inputs.sort_filelist   = True

#grab segmented data
inputnode  = Node(interface=util.IdentityInterface(fields=["subject_id", "anat", "anat_gm", "anat_wm", "anat_csf",  "anat_first", "func", "func_rl","func_lr" ]),name="inputnode")

# define workflow nodes
pipeline_id = 'testing_2'
engine                                      = Workflow(pipeline_id)
pipeline_name                               = pipeline_id
engine.base_dir                             = working_dir
engine.config['execution']['crashdump_dir'] = crash_dir

fsl.FSLCommand.set_default_output_type("NIFTI_GZ")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                       Anatomical Pre-processing
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

'========================================='
'''     Antomical Pre-processing        '''
'========================================='
# reorient to RPI and create brain mask
anat_preproc =  anatomical_preprocessing()
engine.connect(infosource            ,  'subject_id'     ,  datasource         ,  'subject_id'            )
engine.connect(datasource            ,  'anat'           ,  anat_preproc       ,  'inputnode.anat'        )
engine.connect(datasource            ,  'anat_gm'        ,  anat_preproc       ,  'inputnode.anat_gm'     )
engine.connect(datasource            ,  'anat_wm'        ,  anat_preproc       ,  'inputnode.anat_wm'     )
engine.connect(datasource            ,  'anat_csf'       ,  anat_preproc       ,  'inputnode.anat_csf'    )
engine.connect(datasource            ,  'anat_first'     ,  anat_preproc       ,  'inputnode.anat_first'     )

'========================================='
'''  Calculate Normalization Warp Field '''
'''       ANAT_1mm  -- > MNI152_2mm     '''
'========================================='
# run fnirt
anat2mni    = anat2mni_nonlinear()
engine.connect(anat_preproc   ,     'outputnode.brain'     ,  anat2mni,  'inputnode.anat_image'           )
engine.connect(anat_preproc   ,     'outputnode.brain_gm'  ,  anat2mni,  'inputnode.anat_gm'              )
engine.connect(anat_preproc   ,     'outputnode.brain_wm'  ,  anat2mni,  'inputnode.anat_wm'              )
engine.connect(anat_preproc   ,     'outputnode.brain_csf' ,  anat2mni,  'inputnode.anat_cm'              )
engine.connect(anat_preproc   ,     'outputnode.brain_first' ,  anat2mni,  'inputnode.anat_first'              )

#
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                    Functional Image Pre-Processing
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'========================================='
'''      Prepare Functional Data        '''
'========================================='
#Drop 5TRs, Deoblique, Reorient to RPI
func_drop_trs = func_equilibrate()
engine.connect(datasource,          'func'              ,    func_drop_trs ,     'inputnode.verio_func'       )
engine.connect(datasource,          'func_rl'           ,    func_drop_trs ,     'inputnode.verio_func_se'    )
engine.connect(datasource,          'func_lr'           ,    func_drop_trs ,     'inputnode.verio_func_seinv' )

'========================================='
''' Calculate Distortion Correction Warp'''
'========================================='
# Topup, applytopup.
func_disco = func_calc_disco_warp()
engine.connect(func_drop_trs,      'outputnode.analyze_func'       ,    func_disco ,  'inputnode.func'        )
engine.connect(func_drop_trs,      'outputnode.analyze_func_se'    ,    func_disco,   'inputnode.func_se'     )
engine.connect(func_drop_trs,      'outputnode.analyze_func_seinv' ,    func_disco,   'inputnode.func_se_inv' )

'========================================='
''' Calculate Motion Correction Affines '''
'========================================='
func_moco = func_calc_motion_affines()
engine.connect(func_disco   ,      'outputnode.func_disco'         ,    func_moco,     'inputnode.func'       )





"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                    Functional Image Normalization
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

'========================================='
''' Calculate Native Space Linear Affine'''
'''       FUNC_2.3mm -- > ANAT_1mm      '''
'========================================='
# Calculate func -- > anat linear_xfm
# runs flirt using a two step procedure (Mutual Information and Boundary based registration )
func2anat    = func2anat_linear()
engine.connect(func_moco   ,     'outputnode.moco_mean'    ,  func2anat,  'inputnode.func_image'         )
engine.connect(anat_preproc,     'outputnode.brain'        ,  func2anat,  'inputnode.reference_image'    )
engine.connect(anat_preproc,     'outputnode.brain_wm'     ,  func2anat,  'inputnode.anat_wm'            )


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

'========================================='
'''         Motion Statistics           '''
'========================================='

motion_stats =  generate_motion_statistics()
engine.connect(infosource       ,  'subject_id'                    ,  motion_stats,   'inputnode.subject_id'  )
engine.connect(func_moco        ,  'outputnode.moco_param'         ,  motion_stats,   'inputnode.mov_par'     )
engine.connect(func_preproc     ,  'outputnode.func_preproc'       ,  motion_stats,   'inputnode.func_preproc')
engine.connect(func_preproc     ,  'outputnode.func_preproc_mask'  ,  motion_stats,   'inputnode.func_mask'   )

# compute motion derivatives (Friston24)
friston24 = calc_friston_twenty_four()
engine.connect(func_moco        ,  'outputnode.moco_param'         ,  friston24,      'inputnode.mov_par'        )
engine.connect(motion_stats     ,  'outputnode.frames_excluded'    ,  friston24,      'inputnode.frames_excluded')



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                             Nuisance Signal Regression
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'========================================='
'''             Smoothing               '''
'========================================='
# smooth image
func_smooth = smooth_data()
engine.connect(func_preproc,   'outputnode.func_preproc'     ,    func_smooth,   'inputnode.func_data'        )

selector = {'compcor' : True  , 'motion' : True  , 'linear' : True  , 'quadratic' : True,
            'wm'      : False , 'csf'    : False , 'gm'     : False , 'global' : False,
            'pc1'     : False }


'########################### NUISANCE ONLY  ####################################################################'
# - nuisance regression, compcor, 24-Motion Parameters

nuisance = create_residuals_wf('func_nuisance_only')
nuisance.inputs.inputnode.compcor_ncomponents = 5
nuisance.inputs.inputnode.selector = selector
engine.connect(func_preproc,   'outputnode.func_preproc'     ,      nuisance,       'inputnode.func_preproc'  )
engine.connect(func_smooth,    'outputnode.func_smoothed'    ,      nuisance,       'inputnode.func_smoothed' )
engine.connect(anat2mni,       'outputnode.mni_mask_gm'      ,      nuisance,       'inputnode.gm_mask'       )
engine.connect(anat2mni,       'outputnode.mni_mask_wm'      ,      nuisance,       'inputnode.wm_mask'       )
engine.connect(anat2mni,       'outputnode.mni_mask_cm'      ,      nuisance,       'inputnode.csf_mask'      )
engine.connect(friston24,      'outputnode.friston_par'      ,      nuisance,       'inputnode.motion_pars'   )
engine.connect(infosource,     'subject_id'                  ,      nuisance,       'inputnode.subject_id'    )

# Band-Pass filtering
bandpass = bandpas_voxels_workflow('func_bandpass_nuisance')
bandpass.inputs.inputnode.bandpass_freqs = (0.009,0.08)
engine.connect(nuisance, 'outputnode.residual'               ,      bandpass,       'inputnode.func_residuals')

#
'########################### AROMA NUISANCE ####################################################################'
aroma = ICA_AROMA()
aroma.inputs.inputnode.fslDir  = '/usr/share/fsl/5.0/'
aroma.inputs.inputnode.mask    = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
aroma.inputs.inputnode.dim     = 0
aroma.inputs.inputnode.TR      = 1.4
aroma.inputs.inputnode.denType = 'aggr'
engine.connect(func_smooth      ,  'outputnode.func_smoothed'     ,      aroma,   'inputnode.inFile'                )
engine.connect(func_moco        ,  'outputnode.moco_param'        ,      aroma,   'inputnode.mc'                    )

# - nuisance regression, compcor, 24-Motion Parameters
nuisance_aroma = create_residuals_wf('func_nuisance_aroma')
nuisance_aroma.inputs.inputnode.compcor_ncomponents = 5
nuisance_aroma.inputs.inputnode.selector = selector
engine.connect(aroma,          'outputnode.denoised'         ,      nuisance_aroma,       'inputnode.func_preproc'  )
engine.connect(aroma,          'outputnode.denoised'         ,      nuisance_aroma,       'inputnode.func_smoothed' )
engine.connect(anat2mni,       'outputnode.mni_mask_gm'      ,      nuisance_aroma,       'inputnode.gm_mask'       )
engine.connect(anat2mni,       'outputnode.mni_mask_wm'      ,      nuisance_aroma,       'inputnode.wm_mask'       )
engine.connect(anat2mni,       'outputnode.mni_mask_cm'      ,      nuisance_aroma,       'inputnode.csf_mask'      )
engine.connect(friston24,      'outputnode.friston_par'      ,      nuisance_aroma,       'inputnode.motion_pars'   )
engine.connect(infosource,     'subject_id'                  ,      nuisance_aroma,       'inputnode.subject_id'    )

# Band-Pass filtering
bandpass_aroma = bandpas_voxels_workflow('func_bandpass_aroma')
bandpass_aroma.inputs.inputnode.bandpass_freqs = (0.009,0.08)
engine.connect(nuisance_aroma, 'outputnode.residual'         ,      bandpass_aroma,       'inputnode.func_residuals')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                Data Sink
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
sinker = Node(nio.DataSink(base_directory = output_dir), name = 'sinker')
sinker.inputs.base_directory = os.path.join(output_dir, pipeline_name)
sinker.inputs.parameterization = False

engine.connect(infosource,     'subject_id'                , sinker,  'container'                    )
#anatomical
engine.connect(anat_preproc,   'outputnode.brain'          , sinker,  'anatomical_native_brain'       )
engine.connect(anat_preproc,   'outputnode.brain_mask'     , sinker,  'anatomical_native_brain_mask'  )
engine.connect(anat_preproc,   'outputnode.brain_gm'       , sinker,  'anatomical_native_gm'          )
engine.connect(anat_preproc,   'outputnode.brain_wm'       , sinker,  'anatomical_native_wm'          )
engine.connect(anat_preproc,   'outputnode.brain_csf'      , sinker,  'anatomical_native_csf'         )
engine.connect(anat2mni,       'outputnode.mni_mask_gm'    , sinker,  'anatomical_MNI2mm_gm'        )
engine.connect(anat2mni,       'outputnode.mni_mask_wm'    , sinker,  'anatomical_MNI2mm_wm'        )
engine.connect(anat2mni,       'outputnode.mni_mask_cm'    , sinker,  'anatomical_MNI2mm_csf'       )
engine.connect(anat2mni,       'outputnode.anat2mni'       , sinker,  'anatomical_MNI2mm_brain'     )
engine.connect(anat2mni,       'outputnode.nonlinear_warp' , sinker,  'anatomical_MNI2mm_warp'      )

#functional
engine.connect(func2anat,      'outputnode.anat_downsample'  , sinker,  'anatomical_native_brain_downsample'  )
engine.connect(func2anat,      'outputnode.anat2func_xfm'    , sinker,  'anatomical_FUNC2mm_xfm'        )
engine.connect(func2anat,      'outputnode.func2anat'        , sinker,  'functional_ANAT2mm'            )
engine.connect(func2anat,      'outputnode.func2anat_xfm'    , sinker,  'functional_ANAT2mm_xfm'        )
engine.connect(func_transform, 'outputnode.func2mni_allwarps', sinker,  'functional_MNI2mm_brain'             )
engine.connect(func_preproc,   'outputnode.func_preproc'     , sinker,  'functional_MNI2mm_brain_preproc'     )
engine.connect(func_preproc,   'outputnode.func_preproc_mean', sinker,  'functional_MNI2mm_brain_preproc_mean')
engine.connect(func_preproc,   'outputnode.func_preproc_mask', sinker,  'functional_MNI2mm_brain_preproc_mask')

# Motion
engine.connect(func_moco,      'outputnode.moco_param'       ,  sinker,  'functional_motion_par'       )
engine.connect(motion_stats,   'outputnode.FD_power'         ,  sinker,  'functional_motion_FD'          )
engine.connect(motion_stats,   'outputnode.frames_excluded'  ,  sinker,  'functional_motion_FD_exclude'          )
engine.connect(motion_stats,   'outputnode.frames_included'  ,  sinker,  'functional_motion_FD_include'          )
engine.connect(motion_stats,   'outputnode.power_params'     ,  sinker,  'functional_motion_stas'       )
engine.connect(friston24,      'outputnode.friston_par'      ,  sinker,  'functional_motion_friston24'        )

#nuisance
engine.connect(func_smooth,    'outputnode.func_smoothed'    ,  sinker,  'functional_MNI2mm_brain_preproc_smooth')
engine.connect(nuisance,       'outputnode.residual'         ,  sinker,  'functional_MNI2mm_brain_preproc_nuisance_residuals')
engine.connect(bandpass,       'outputnode.func_bandpassed'  ,  sinker,  'functional_MNI2mm_brain_preproc_nuisance_bandpass')

engine.connect(aroma,          'outputnode.denoised'         ,  sinker,  'functional_MNI2mm_brain_preproc_aroma')
engine.connect(nuisance_aroma, 'outputnode.residual'         ,  sinker,  'functional_MNI2mm_brain_preproc_aroma_residuals')
engine.connect(bandpass,       'outputnode.func_bandpassed'  ,  sinker,  'functional_MNI2mm_brain_preproc_aroma_bandpass')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                             Run workflow
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    engine.run(plugin='CondorDAGMan')
    engine.write_graph(graph2use='colored', format='pdf')