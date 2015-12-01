__author__ = 'kanaan' '11.08.2015'

import os
from   nipype.pipeline.engine         import Workflow, Node
import nipype.interfaces.io           as nio
import nipype.interfaces.utility      as util
import nipype.interfaces.fsl          as fsl

from variables.subject_list           import working_dir, crash_dir, output_dir_a,  output_dir_b, study_a_list, study_b_list, patients_b
from anatomical.anat_preprocess       import anatomical_preprocessing
from functional.func_preprocess       import func_equilibrate, func_calc_disco_warp, func_preprocess
from functional.func_subcortical      import make_func_subcortical_masks
from motion.motion_calc               import func_calc_motion_affines, generate_motion_statistics, calc_friston_twenty_four
from registration.transforms          import anat2mni_nonlinear, func2anat_linear, func2mni_wf
from denoise.nuisance                 import create_residuals_wf
from denoise.smooth                   import smooth_data
from denoise.run_ica_aroma            import ica_aroma_workflow
from denoise.bandpass                 import bandpas_voxels_workflow

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                        Pipeline Definitions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#input data
output_dir           = output_dir_a
# output_dir            = output_dir_b

data_dir             = '/scr/sambesi2/TS_EUROTRAIN/VERIO/nmr093a/'
# data_dir              = '/scr/sambesi2/TS_EUROTRAIN/VERIO/nmr093b/'

#subject_list         = study_b_list
subject_list         = ['HCTT']
# subject_list = patients_b

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
pipeline_id = 'GluConnectivity'
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
engine.connect(anat_preproc   ,     'outputnode.brain'       ,  anat2mni,  'inputnode.anat_image'           )
engine.connect(anat_preproc   ,     'outputnode.brain_wm'    ,  anat2mni,  'inputnode.anat_wm'              )
engine.connect(anat_preproc   ,     'outputnode.brain_gm'    ,  anat2mni,  'inputnode.anat_gm'              )
engine.connect(anat_preproc   ,     'outputnode.brain_csf'   ,  anat2mni,  'inputnode.anat_cm'              )
engine.connect(anat_preproc   ,     'outputnode.brain_first' ,  anat2mni,  'inputnode.anat_first'         )

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
'''     Distortion Correction Warp      '''
'========================================='
# Topup, applytopup.
func_disco = func_calc_disco_warp()
engine.connect(func_drop_trs,      'outputnode.analyze_func'       ,    func_disco ,  'inputnode.func'        )
engine.connect(func_drop_trs,      'outputnode.analyze_func_se'    ,    func_disco,   'inputnode.func_se'     )
engine.connect(func_drop_trs,      'outputnode.analyze_func_seinv' ,    func_disco,   'inputnode.func_se_inv' )

'========================================='
'''         Motion Correction           '''
'========================================='
func_moco = func_calc_motion_affines()
engine.connect(func_disco   ,      'outputnode.func_disco'         ,    func_moco,     'inputnode.func'       )

'========================================='
'''          Preprocess  func           '''
'========================================='
# intensity normalization and deskulling
func_preproc =  func_preprocess('func_preproc')
engine.connect(func_moco,       'outputnode.moco_func'       ,   func_preproc,  'inputnode.func_in'          )

'========================================='
'''         Motion Statistics           '''
'========================================='

motion_stats =  generate_motion_statistics()
engine.connect(infosource       ,  'subject_id'                    ,  motion_stats,   'inputnode.subject_id'  )
engine.connect(func_moco        ,  'outputnode.moco_param'         ,  motion_stats,   'inputnode.mov_par'     )
engine.connect(func_preproc     ,  'outputnode.func_preproc'       ,  motion_stats,   'inputnode.func_preproc')
engine.connect(func_preproc     ,  'outputnode.func_preproc_mask'  ,  motion_stats,   'inputnode.func_mask'   )
#
# compute motion derivatives (Friston24)
friston24 = calc_friston_twenty_four()
engine.connect(func_moco        ,  'outputnode.moco_param'         ,  friston24,      'inputnode.mov_par'        )
engine.connect(motion_stats     ,  'outputnode.frames_excluded'    ,  friston24,      'inputnode.frames_excluded')

'========================================='
'''       FUNC_2.3mm -- > ANAT_2mm      '''
'========================================='
# Calculate func -- > anat linear_xfm
# runs flirt using a two step procedure (Mutual Information and Boundary based registration )
func2anat    = func2anat_linear()
engine.connect(func_preproc,     'outputnode.func_preproc_mean',  func2anat,  'inputnode.func_image'       )
engine.connect(func_preproc,     'outputnode.func_preproc_mask',  func2anat,  'inputnode.func_mask'        )
engine.connect(anat_preproc,     'outputnode.brain'            ,  func2anat,  'inputnode.reference_image'  )
engine.connect(anat_preproc,     'outputnode.brain_wm'         ,  func2anat,  'inputnode.anat_wm'          )
engine.connect(anat_preproc,     'outputnode.brain_gm'         ,  func2anat,  'inputnode.anat_gm'          )
engine.connect(anat_preproc,     'outputnode.brain_csf'        ,  func2anat,  'inputnode.anat_csf'         )
engine.connect(anat_preproc,     'outputnode.brain_first'      ,  func2anat,  'inputnode.anat_first'       )

#create subcortical tisssue masks in functional space
func_subcortical = make_func_subcortical_masks()
engine.connect(func2anat,     'outputnode.func_first'          , func_subcortical,  'inputnode.func_first' )

'========================================='
'''              Denoising              '''
'========================================='
# Smooth
func_smooth = smooth_data('func_preproc_smoothed')
engine.connect(func_preproc,   'outputnode.func_preproc'     ,    func_smooth,   'inputnode.func_data'     )

aroma = ica_aroma_workflow()
aroma.inputs.inputnode.fslDir = '/usr/share/fsl/5.0/bin/'
aroma.inputs.inputnode.dim = 0
aroma.inputs.inputnode.TR = 1.4
aroma.inputs.inputnode.denType = 'nonaggr'
engine.connect(func_smooth,     'outputnode.func_smoothed'    , aroma     , 'inputnode.inFile'             )
engine.connect(func_preproc,    'outputnode.func_preproc_mask', aroma     , 'inputnode.mask'               )
engine.connect(func_moco,       'outputnode.moco_param'       , aroma     , 'inputnode.mc'                 )
engine.connect(func2anat,       'outputnode.func2anat_xfm'    , aroma     , 'inputnode.affmat'             )
engine.connect(anat2mni,        'outputnode.nonlinear_warp'   , aroma     , 'inputnode.warp'               )

nuisance = create_residuals_wf('func_nuisance')
nuisance.inputs.inputnode.compcor_ncomponents = 5

engine.connect(aroma,          'outputnode.denoised'         ,      nuisance,       'inputnode.func_preproc'  )
engine.connect(func_preproc,   'outputnode.func_preproc'     ,      nuisance,       'inputnode.func_preproc_no_aroma'  )
engine.connect(func2anat,      'outputnode.func_gm'          ,      nuisance,       'inputnode.gm_mask'       )
engine.connect(func2anat,      'outputnode.func_wm'          ,      nuisance,       'inputnode.wm_mask'       )
engine.connect(func2anat,      'outputnode.func_csf'         ,      nuisance,       'inputnode.csf_mask'      )
engine.connect(anat2mni,       'outputnode.mni2anat_warp'    ,      nuisance,       'inputnode.mni2anat_warp'      )
engine.connect(func2anat,      'outputnode.anat2func_xfm'    ,      nuisance,       'inputnode.anat2func_aff'      )
engine.connect(friston24,      'outputnode.friston_par'      ,      nuisance,       'inputnode.motion_pars'   )
engine.connect(infosource,     'subject_id'                  ,      nuisance,       'inputnode.subject_id'    )

# Band-Pass filtering
bandpass = bandpas_voxels_workflow('func_bandpass_nuisance')
bandpass.inputs.inputnode.bandpass_freqs = (0.009,0.08)
engine.connect(nuisance, 'outputnode.residual'               ,      bandpass,       'inputnode.func')

'========================================='
'''         Prepare for GROUP ICA       '''
'========================================='
######### Prep data for group ICA
# go to mni
func2gica = func2mni_wf()
engine.connect(aroma,     'outputnode.denoised'         ,     func2gica,       'inputnode.func_image')
engine.connect(func2anat, 'outputnode.func2anat_xfm'    ,     func2gica,       'inputnode.func2anat_affine')
engine.connect(anat2mni,  'outputnode.nonlinear_warp'   ,     func2gica,       'inputnode.anat2mni_warp'               )

# High pass filtering
func2gica_hp = bandpas_voxels_workflow('func_highpass_gica')
func2gica_hp.inputs.inputnode.bandpass_freqs = (0.01,999999)
engine.connect(func2gica, 'outputnode.func2mni_4mm'         ,      func2gica_hp,       'inputnode.func')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                Data Sink
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
sinker = Node(nio.DataSink(base_directory = output_dir), name = 'sinker')
sinker.inputs.base_directory = os.path.join(output_dir, pipeline_name)
sinker.inputs.parameterization = False

engine.connect(infosource,     'subject_id'                , sinker,  'container'                               )
#anatomical sinks
engine.connect(anat_preproc,   'outputnode.brain'          , sinker,  'anatomical_native_brain_1mm' )
engine.connect(anat_preproc,   'outputnode.brain_mask'     , sinker,  'anatomical_native_brain_mask'            )
engine.connect(anat_preproc,   'outputnode.brain_gm'       , sinker,  'anatomical_native_tissue_gm'         )
engine.connect(anat_preproc,   'outputnode.brain_csf'      , sinker,  'anatomical_native_tissue_wm'         )
engine.connect(anat_preproc,   'outputnode.brain_csf'      , sinker,  'anatomical_native_tissue_csf'        )
engine.connect(anat_preproc,   'outputnode.brain_first'    , sinker,  'anatomical_native_tissue_first'      )

#anatomical transforms
engine.connect(func2anat,      'outputnode.anat2func_xfm'    , sinker,  'anatomical_FUNC2mm_xfm'            )
engine.connect(func2anat,      'outputnode.anat2func'        , sinker,  'anatomical_FUNC2mm_brain'          )
engine.connect(anat2mni,       'outputnode.mni_mask_gm'      , sinker,  'anatomical_MNI2mm_tissue_gm'       )
engine.connect(anat2mni,       'outputnode.mni_mask_wm'      , sinker,  'anatomical_MNI2mm_tissue_wm'       )
engine.connect(anat2mni,       'outputnode.mni_mask_cm'      , sinker,  'anatomical_MNI2mm_tissue_csf'      )
engine.connect(anat2mni,       'outputnode.anat2mni'         , sinker,  'anatomical_MNI2mm_brain'           )
engine.connect(anat2mni,       'outputnode.nonlinear_warp'   , sinker,  'anatomical_MNI2mm_xfm'             )
engine.connect(anat2mni,       'outputnode.mni2anat_warp'    , sinker,  'MNI2mm_ANAT_xfm'                   )

#functional transforms
engine.connect(func2anat,      'outputnode.func2anat'        , sinker,  'functional_ANAT2mm'                 )
engine.connect(func2anat,      'outputnode.func2anat_xfm'    , sinker,  'functional_ANAT2mm_xfm'             )

# functional  preprocessing
engine.connect(func_moco,      'outputnode.moco_mean'        , sinker,  'functional_native_discomoco'          )
engine.connect(func_preproc,   'outputnode.func_preproc'     , sinker,  'functional_native_brain_preproc'     )
engine.connect(func_preproc,   'outputnode.func_preproc_mean', sinker,  'functional_native_brain_preproc_mean')
engine.connect(func_preproc,   'outputnode.func_preproc_mask', sinker,  'functional_native_brain_preproc_mask')

engine.connect(func2anat,      'outputnode.func_gm'          , sinker,  'functional_native_gm')
engine.connect(func2anat,      'outputnode.func_wm'          , sinker,  'functional_native_wm')
engine.connect(func2anat,      'outputnode.func_csf'         , sinker,  'functional_native_csf')
engine.connect(func2anat,      'outputnode.func_first'       , sinker,  'functional_native_first')

engine.connect(func_subcortical, 'outputnode.left_nacc'        , sinker,  'functional_subcortical.@left_nacc')
engine.connect(func_subcortical, 'outputnode.left_amygdala'    , sinker,  'functional_subcortical.@left_amygdala')
engine.connect(func_subcortical, 'outputnode.left_caudate'     , sinker,  'functional_subcortical.@left_caudate')
engine.connect(func_subcortical, 'outputnode.left_hipoocampus' , sinker,  'functional_subcortical.@left_hipoocampus')
engine.connect(func_subcortical, 'outputnode.left_pallidum'    , sinker,  'functional_subcortical.@left_pallidum')
engine.connect(func_subcortical, 'outputnode.left_putamen'     , sinker,  'functional_subcortical.@left_putamen')
engine.connect(func_subcortical, 'outputnode.left_thalamus'    , sinker,  'functional_subcortical.@left_thalamus')
engine.connect(func_subcortical, 'outputnode.right_nacc'       , sinker,  'functional_subcortical.@right_nacc')
engine.connect(func_subcortical, 'outputnode.right_amygdala'   , sinker,  'functional_subcortical.@right_amygdala')
engine.connect(func_subcortical, 'outputnode.right_caudate'    , sinker,  'functional_subcortical.@right_caudate')
engine.connect(func_subcortical, 'outputnode.right_hipoocampus', sinker,  'functional_subcortical.@right_hipoocampus')
engine.connect(func_subcortical, 'outputnode.right_pallidum'   , sinker,  'functional_subcortical.@right_pallidum')
engine.connect(func_subcortical, 'outputnode.right_putamen'    , sinker,  'functional_subcortical.@right_putamen')
engine.connect(func_subcortical, 'outputnode.right_thalamus'   , sinker,  'functional_subcortical.@right_thalamus')
engine.connect(func_subcortical, 'outputnode.midbrain'         , sinker,  'functional_subcortical.@midbrain')
engine.connect(func_subcortical, 'outputnode.left_striatum'    , sinker,  'functional_subcortical.@left_striatum')
engine.connect(func_subcortical, 'outputnode.right_striatum'   , sinker,  'functional_subcortical.@right_striatum')

# Denoising
engine.connect(nuisance,       'outputnode.residual_no_aroma', sinker,  'functional_native_brain_preproc_residual_no_AROMA')
engine.connect(func_smooth,    'outputnode.func_smoothed'    , sinker,  'functional_native_brain_preproc_FWHM')
engine.connect(aroma,          'outputnode.denoised'         , sinker,  'functional_native_brain_preproc_FWHM_AROMA')
engine.connect(nuisance,       'outputnode.residual'         , sinker,  'functional_native_brain_preproc_FWHM_AROMA_residual')
engine.connect(bandpass,       'outputnode.func_bandpassed'  , sinker,  'functional_native_brain_preproc_FWHM_AROMA_residual_bp')

#motion statistics
engine.connect(func_moco,      'outputnode.moco_param'       ,  sinker,  'functional_motion_parameters'       )
engine.connect(motion_stats,   'outputnode.FD_power'         ,  sinker,  'functional_motion_FDPower'          )
engine.connect(motion_stats,   'outputnode.frames_excluded'  ,  sinker,  'functional_motion_exclude'          )
engine.connect(motion_stats,   'outputnode.frames_included'  ,  sinker,  'functional_motion_include'          )
engine.connect(motion_stats,   'outputnode.power_params'     ,  sinker,  'functional_motion_statistics'       )
engine.connect(friston24,      'outputnode.friston_par'      ,  sinker,  'functional_motion_friston24'        )

# Group ICA output
engine.connect(func2gica_hp,   'outputnode.func_bandpassed'  , sinker,   'functional_MNI4mm_preproc_FWHM_AROMA_residual_high_pass')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                             Run workflow
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    engine.run()#plugin='CondorDAGMan')
    engine.write_graph(graph2use='colored', format='pdf')
