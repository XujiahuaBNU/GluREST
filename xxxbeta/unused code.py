# __author__ = 'kanaan'
#
# from  nipype.pipeline.engine import Node, Workflow
# from nipype.interfaces import spm, fsl, afni
# def spm_new_segment():
#
#     """
#     Workflow to run SPM12 segmentation using NewSegment algorithm
#     GM,WM,CSF tissue masks are binned and thresholded.
#
#     Inputs::
#            inputspec.anat_preproc : skullstripped T1_file
#     Outputs::
#            outputspec.mask_gm : Thresholded and Binarized GM mask
#            outputspec.mask_wm : Thresholded and Binarized WM mask
#            outputspec.mask_wm : Thresholded and Binarized CSM mask
#     """
#     import nipype.interfaces.utility as util
#
#     flow                    = Workflow(name='anat_segment')
#
#     inputnode               = Node(util.IdentityInterface(fields=['anat_preproc']), name  = 'inputnode')
#
#     outputnode              = Node(util.IdentityInterface(fields=['mask_gm',
#                                                                   'mask_wm',
#                                                                   'mask_csf']),     name  = 'outputnode')
#
#     # SPM tissue prob maps are in AIL oriention, must switch from RPI
#     # reorient   = Node(interface=preprocess.Resample(),  name = 'anat_reorient_ail')
#     # reorient.inputs.orientation = 'AI'
#     # reorient.inputs.outputtype = 'NIFTI'
#
#     segment                      = Node(spm.NewSegment(), name='spm_NewSegment')
#     segment.inputs.channel_info  = (0.0001, 60, (True, True))
#
#     select_gm  = Node(util.Select(), name = 'select_1')
#     select_gm.inputs.set(inlist=[1, 2, 3], index=[0])
#
#     select_wm  = Node(util.Select(), name = 'select_2')
#     select_wm.inputs.set(inlist=[1, 2, 3], index=[1])
#
#     select_cm  = Node(util.Select(), name = 'select_3')
#     select_cm.inputs.set(inlist=[1, 2, 3], index=[2])
#
#     thresh_gm                    = Node(fsl.Threshold(), name= 'bin_gm')
#     thresh_gm.inputs.thresh      = 0.5
#     thresh_gm.inputs.args        = ' -bin'
#
#     thresh_wm                    = Node(fsl.Threshold(), name= 'bin_wm')
#     thresh_wm.inputs.thresh      = 0.5
#     thresh_wm.inputs.args        = '-bin'
#
#     thresh_csf                   = Node(fsl.Threshold(), name= 'bin_csf')
#     thresh_csf.inputs.thresh     = 0.5
#     thresh_csf.inputs.args       = '-bin'
#
#     reorient_gm                    = Node(interface=afni.preprocess.Resample(), name = 'mask_gm')
#     reorient_gm.inputs.orientation = 'RPI'
#     reorient_gm.inputs.outputtype  = 'NIFTI_GZ'
#     reorient_wm                    = reorient_gm.clone(name= 'mask_wm')
#     reorient_csf                   = reorient_gm.clone(name= 'mask_csf')
#
#     flow.connect(inputnode        ,  "anat_preproc"          ,     segment     ,  "channel_files"  )
#     flow.connect(segment          ,  "native_class_images"   ,     select_gm   ,  "inlist"         )
#     flow.connect(segment          ,  "native_class_images"   ,     select_wm   ,  "inlist"         )
#     flow.connect(segment          ,  "native_class_images"   ,     select_cm   ,  "inlist"         )
#     flow.connect(select_gm        ,  'out'                   ,     thresh_gm   ,  'in_file'        )
#     flow.connect(select_wm        ,  'out'                   ,     thresh_wm   ,  'in_file'        )
#     flow.connect(select_cm        ,  'out'                   ,     thresh_csf  ,  'in_file'        )
#     flow.connect(thresh_gm        ,  "out_file"              ,     reorient_gm ,  "in_file"        )
#     flow.connect(thresh_wm        ,  "out_file"              ,     reorient_wm ,  "in_file"        )
#     flow.connect(thresh_csf       ,  "out_file"              ,     reorient_csf,  "in_file"        )
#     flow.connect(reorient_gm      ,  "out_file"              ,     outputnode  ,  "mask_gm"        )
#     flow.connect(reorient_wm      ,  "out_file"              ,     outputnode  ,  "mask_wm"        )
#     flow.connect(reorient_csf     ,  "out_file"              ,     outputnode  ,  "mask_csf"       )
#     return flow
#
#
#
#
#
# def func_equilibrate_bias():
#     '''
#     Workflow to get the scanner data ready.
#     Anatomical and functional images are deobliqued.
#     5 TRs are removed from func data.
#
#     inputs
#         inputnode.verio_anat
#         inputnode.verio_func
#         inputnode.verio_func_se
#         inputnode.verio_func_se_inv
#     outputs
#         outputnode.analyze_anat
#         outputnode.analyze_func
#         outputnode.analyze_func_se
#         outputnode.analyze_func_se_inv
#     '''
#
#     flow        = Workflow('func_equilibrate')
#     inputnode   = Node(util.IdentityInterface(fields= ['verio_func',
#                                                        'verio_func_se',
#                                                        'verio_func_seinv']),
#                        name = 'inputnode')
#     outputnode  = Node(util.IdentityInterface(fields= ['analyze_func',
#                                                        'func_mask',
#                                                        'analyze_func_se',
#                                                        'analyze_func_seinv']),
#                        name = 'outputnode')
#
#     ## functional image
#
#     # 1. remove TRS
#     remove_trs                             = Node(interface = preprocess.Calc(),      name = 'func_drop_trs')
#     remove_trs.inputs.start_idx            = 5
#     remove_trs.inputs.stop_idx             = 421
#     remove_trs.inputs.expr                 = 'a'
#     remove_trs.inputs.outputtype           = 'NIFTI_GZ'
#
#     # 2. to RPI
#     func_rpi                          = Node(interface= preprocess.Resample(),   name = 'func_rpi')
#     func_rpi.inputs.orientation       = 'RPI'
#     func_rpi.inputs.outputtype        = 'NIFTI_GZ'
#
#     # 3. func deoblique
#     func_deoblique                    = Node(interface=preprocess.Refit(),       name = 'func_deoblique')
#     func_deoblique.inputs.deoblique   = True
#
#
#     # 3- ger mean and splits
#     func_mean                          = Node(interface = preprocess.TStat(),     name = 'func_mean')
#     func_mean.inputs.options           = '-mean'
#     func_mean.inputs.outputtype        = 'NIFTI'
#
#     func_split                         = Node(interface = fsl.Split(), name = 'func_split')
#     func_split.inputs.dimension        = 't'
#     func_split.inputs.out_base_name    = 'split'
#
#     # 4. calculate bias field on mean
#     func_n4                            = Node(interface=N4BiasFieldCorrection(), name = 'func_mean_N4')
#     func_get_bias                      = Node(interface=fsl.BinaryMaths(), name = 'func_mean_getbiasfield')
#     func_get_bias.inputs.operation     = 'div'
#
#     func_apply_bias                    = MapNode(interface=fsl.BinaryMaths(), name = 'func_biasfield_apply_to_splitvols', iterfield = ['in_file'])
#     func_apply_bias.inputs.operation   = 'div'
#
#     # 6. merge corrected frames
#     func_merge                           = Node(interface = fsl.Merge(), name = 'func_biasfield_merged')
#     func_merge.inputs.dimension          = 't'
#     func_merge.inputs.output_type        = 'NIFTI'
#
#
#     flow.connect(inputnode           ,   'verio_func'       ,  remove_trs          ,  'in_file_a'             )
#     flow.connect(remove_trs          ,   'out_file'         ,  func_rpi            ,  'in_file'               )
#     flow.connect(func_rpi            ,   'out_file'         ,  func_deoblique      ,  'in_file'               )
#     flow.connect(func_deoblique      ,   'out_file'         ,  func_mean           ,  'in_file'               )
#     flow.connect(func_deoblique      ,   'out_file'         ,  func_split          ,  'in_file'               )
#     flow.connect(func_mean           ,   'out_file'         ,  func_n4             ,  'input_image'           )
#     flow.connect(func_mean           ,   'out_file'         ,  func_get_bias       ,  'in_file'               )
#     flow.connect(func_n4             ,   'output_image'     ,  func_get_bias       ,  'operand_file'          )
#     flow.connect(func_split          ,   'out_files'        ,  func_apply_bias     ,  'in_file'               )
#     flow.connect(func_get_bias       ,   'out_file'         ,  func_apply_bias     ,  'operand_file'          )
#     flow.connect(func_apply_bias     ,   'out_file'         ,  func_merge          ,  'in_files'              )
#
#     flow.connect(func_merge          ,   'merged_file'      ,  outputnode          ,  'analyze_func'          )
#
#
#     ###########################################################################################################
#     ###########################################################################################################
#     ###########################################################################################################
#     # se to RPI
#     se_rpi                          = Node(interface= preprocess.Resample(),   name = 'se_rpi')
#     se_rpi.inputs.orientation       = 'RPI'
#     se_rpi.inputs.outputtype        = 'NIFTI_GZ'
#
#     # 3. func deoblique
#     se_deoblique                    = Node(interface=preprocess.Refit(),       name = 'se_deoblique')
#     se_deoblique.inputs.deoblique   = True
#
#     # 3- ger mean and splits
#     se_mean                          = Node(interface = preprocess.TStat(),     name = 'se_mean')
#     se_mean.inputs.options           = '-mean'
#     se_mean.inputs.outputtype        = 'NIFTI'
#
#     se_split                         = Node(interface = fsl.Split(), name = 'se_split')
#     se_split.inputs.dimension        = 't'
#     se_split.inputs.out_base_name    = 'split'
#
#     # 4. calculate bias field on mean
#     se_n4                            = Node(interface=N4BiasFieldCorrection(), name = 'se_mean_N4')
#     se_get_bias                      = Node(interface=fsl.BinaryMaths(), name = 'se_mean_getbiasfield')
#     se_get_bias.inputs.operation     = 'div'
#
#     se_apply_bias                    = MapNode(interface=fsl.BinaryMaths(), name = 'se_biasfield_apply_to_splitvols', iterfield = ['in_file'])
#     se_apply_bias.inputs.operation   = 'div'
#
#     # 6. merge corrected frames
#     se_merge                           = Node(interface = fsl.Merge(), name = 'se_biasfield_merged')
#     se_merge.inputs.dimension          = 't'
#     se_merge.inputs.output_type        = 'NIFTI'
#
#
#     flow.connect(inputnode         ,   'verio_func_se'    ,  se_rpi            ,  'in_file'               )
#     flow.connect(se_rpi            ,   'out_file'         ,  se_deoblique      ,  'in_file'               )
#     flow.connect(se_deoblique      ,   'out_file'         ,  se_mean           ,  'in_file'               )
#     flow.connect(se_deoblique      ,   'out_file'         ,  se_split          ,  'in_file'               )
#     flow.connect(se_mean           ,   'out_file'         ,  se_n4             ,  'input_image'           )
#     flow.connect(se_mean           ,   'out_file'         ,  se_get_bias       ,  'in_file'               )
#     flow.connect(se_n4             ,   'output_image'     ,  se_get_bias       ,  'operand_file'          )
#     flow.connect(se_split          ,   'out_files'        ,  se_apply_bias     ,  'in_file'               )
#     flow.connect(se_get_bias     ,   'out_file'           ,  se_apply_bias     ,  'operand_file'          )
#     flow.connect(se_apply_bias   ,   'out_file'           ,  se_merge          ,  'in_files'              )
#     flow.connect(se_merge        ,   'merged_file'      ,  outputnode          ,  'analyze_func_se'          )
#
#     ###########################################################################################################
#     ###########################################################################################################
#     ###########################################################################################################
#
#     # se_inv to RPI
#     se_inv_rpi                          = Node(interface= preprocess.Resample(),   name = 'seinv_rpi')
#     se_inv_rpi.inputs.orientation       = 'RPI'
#     se_inv_rpi.inputs.outputtype        = 'NIFTI_GZ'
#
#     # 3. func deoblique
#     se_inv_deoblique                    = Node(interface=preprocess.Refit(),       name = 'seinv_deoblique')
#     se_inv_deoblique.inputs.deoblique   = True
#
#     # 3- ger mean and splits
#     se_inv_mean                          = Node(interface = preprocess.TStat(),     name = 'seinv_mean')
#     se_inv_mean.inputs.options           = '-mean'
#     se_inv_mean.inputs.outputtype        = 'NIFTI'
#
#     se_inv_split                         = Node(interface = fsl.Split(), name = 'seinv_split')
#     se_inv_split.inputs.dimension        = 't'
#     se_inv_split.inputs.out_base_name    = 'split'
#
#     # 4. calculate bias field on mean
#     se_inv_n4                            = Node(interface=N4BiasFieldCorrection(), name = 'seinv_mean_N4')
#     se_inv_get_bias                      = Node(interface=fsl.BinaryMaths(), name = 'seinv_mean_getbiasfield')
#     se_inv_get_bias.inputs.operation     = 'div'
#
#     se_inv_apply_bias                    = MapNode(interface=fsl.BinaryMaths(), name = 'seinv_biasfield_apply_to_splitvols', iterfield = ['in_file'])
#     se_inv_apply_bias.inputs.operation   = 'div'
#
#     # 6. merge corrected frames
#     se_inv_merge                           = Node(interface = fsl.Merge(), name = 'seinv_biasfield_merged')
#     se_inv_merge.inputs.dimension          = 't'
#     se_inv_merge.inputs.output_type        = 'NIFTI'
#
#
#     flow.connect(inputnode            ,   'verio_func_seinv'     ,  se_inv_rpi            ,  'in_file'               )
#     flow.connect(se_inv_rpi           ,   'out_file'             ,  se_inv_deoblique      ,  'in_file'               )
#
#     flow.connect(se_inv_deoblique     ,   'out_file'             ,  se_inv_mean           ,  'in_file'               )
#     flow.connect(se_inv_deoblique     ,   'out_file'             ,  se_inv_split          ,  'in_file'               )
#
#     flow.connect(se_inv_mean          ,   'out_file'             ,  se_inv_n4             ,  'input_image'           )
#
#     flow.connect(se_inv_mean          ,   'out_file'             ,  se_inv_get_bias       ,  'in_file'               )
#     flow.connect(se_inv_n4            ,   'output_image'         ,  se_inv_get_bias       ,  'operand_file'          )
#
#     flow.connect(se_inv_split         ,   'out_files'            ,  se_inv_apply_bias     ,  'in_file'               )
#     flow.connect(se_inv_get_bias      ,   'out_file'             ,  se_inv_apply_bias     ,  'operand_file'          )
#     flow.connect(se_inv_apply_bias    ,   'out_file'             ,  se_inv_merge          ,  'in_files'              )
#
#     flow.connect(se_inv_merge         ,   'merged_file'          ,  outputnode            ,  'analyze_func_seinv'          )
#
#     return flow

# def change_itk_transform_type(input_affine_file):
#
#     '''
#     this function takes in the affine.txt produced by the c3d_affine_tool
#     (which converted an FSL FLIRT affine.mat into the affine.txt)
#
#     it then modifies the 'Transform Type' of this affine.txt so that it is
#     compatible with the antsApplyTransforms tool and produces a new affine
#     file titled 'updated_affine.txt'
#     '''
#
#     import os
#
#     new_file_lines = []
#
#     with open(input_affine_file) as f:
#
#         for line in f:
#
#             if 'Transform:' in line:
#
#                 if 'MatrixOffsetTransformBase_double_3_3' in line:
#
#                     transform_line = 'Transform: AffineTransform_double_3_3'
#                     new_file_lines.append(transform_line)
#
#             else:
#
#                 new_file_lines.append(line)
#
#
#     updated_affine_file = os.path.join(os.getcwd(), 'updated_affine.txt')
#
#     outfile = open(updated_affine_file, 'wt')
#
#     for line in new_file_lines:
#
#         print >>outfile, line.strip('\n')
#
#     outfile.close()
#
#
#     return updated_affine_file

# # forked from CPAC version 0.39 https://github.com/FCP-INDI/C-PAC
#
# import numpy as np
#
#
# def calc_compcor_components(data, nComponents, wm_sigs, csf_sigs):
#     import scipy.signal as signal
#
#     wmcsf_sigs = np.vstack((wm_sigs, csf_sigs))
#
#     # filter out any voxels whose variance equals 0
#     print 'Removing zero variance components'
#     wmcsf_sigs = wmcsf_sigs[wmcsf_sigs.std(1)!=0,:]
#
#     if wmcsf_sigs.shape.count(0):
#         print 'No wm or csf signals left after removing those with zero variance'
#         raise IndexError
#
#     print 'Detrending and centering data'
#     Y = signal.detrend(wmcsf_sigs, axis=1, type='linear').T
#     Yc = Y - np.tile(Y.mean(0), (Y.shape[0], 1))
#     Yc = Yc / np.tile(np.array(Y.std(0)).reshape(1,Y.shape[1]), (Y.shape[0],1))
#
#     print 'Calculating SVD decomposition of Y*Y\''
#     U, S, Vh = np.linalg.svd(Yc)
#
#     return U[:,:nComponents]
#
# def erode_mask(data):
#     mask = data != 0
#     eroded_mask = np.zeros_like(data, dtype='bool')
#     max_x, max_y, max_z = data.shape
#     x,y,z = np.where(data != 0)
#     for i in range(x.shape[0]):
#         if (max_x-1) == x[i] or \
#            (max_y-1) == y[i] or \
#            (max_z-1) == z[i] or \
#            x[i] == 0 or \
#            y[i] == 0 or \
#            z[i] == 0:
#             eroded_mask[x[i],y[i],z[i]] = False
#         else:
#             eroded_mask[x[i],y[i],z[i]] = mask[x[i], y[i], z[i]] * \
#                                           mask[x[i] + 1, y[i], z[i]] * \
#                                           mask[x[i], y[i] + 1, z[i]] * \
#                                           mask[x[i], y[i], z[i] + 1] * \
#                                           mask[x[i] - 1, y[i], z[i]] * \
#                                           mask[x[i], y[i] - 1, z[i]] * \
#                                           mask[x[i], y[i], z[i] - 1]
#
#     eroded_data = np.zeros_like(data)
#     eroded_data[eroded_mask] = data[eroded_mask]
#
#     return eroded_data
#
#
# def safe_shape(*vol_data):
#     """
#     Checks if the volume (first three dimensions) of multiple ndarrays
#     are the same shape.
#
#     Parameters
#     ----------
#     vol_data0, vol_data1, ..., vol_datan : ndarray
#         Volumes to check
#
#     Returns
#     -------
#     same_volume : bool
#         True only if all volumes have the same shape.
#     """
#     same_volume = True
#
#     first_vol_shape = vol_data[0].shape[:3]
#     for vol in vol_data[1:]:
#         same_volume &= (first_vol_shape == vol.shape[:3])
#
#     return same_volume




def create_residuals(name):

    ##### Ugly... though iterables work, difficult to sink..... revist this with iterables and easy sinking
    '''
    inputs
        inputnode.func_file
        inputnode.wm_mask
        inputnode.gm_mask
        inputnode.csf_mask
        inputnode.motion_pars
        inputnode.selector
        inputnode.compcor_ncomponents
    outputs
        outputnode.residual
        outputnode.regressor_csv
    '''

    flow = Workflow(name=name)
    inputnode = Node(util.IdentityInterface(fields=['func_preproc',
                                                    'func_smoothed',
                                                    'wm_mask',
                                                    'gm_mask',
                                                    'csf_mask',
                                                    'motion_pars',
                                                    'compcor_ncomponents',
                                                    'subject_id']),
                name = 'inputnode')
    outputnode = Node(util.IdentityInterface(fields=['dt_res',
                                                     'dt_reg',
                                                     'dt_mc_res',
                                                     'dt_mc_reg',
                                                     'dt_mc_wmcsf_res',
                                                     'dt_mc_wmcsf_reg',
                                                     'dt_mc_cc_res',
                                                     'dt_mc_cc_reg',
                                                     'dt_mc_cc_gs_res',
                                                     'dt_mc_cc_gs_reg']),
                name = 'outputnode')

    ##########################################################################################################################
                                                  # Extract Tissue Signals  #
    ##########################################################################################################################
    extract_tissue = Node(util.Function(input_names   =[ 'data_file',
                                                         'ventricles_mask_file',
                                                         'wm_seg_file',
                                                         'csf_seg_file',
                                                         'gm_seg_file'],
                                         output_names = ['file_wm',
                                                         'file_csf',
                                                         'file_gm'],
                                         function     =extract_tissue_data),
                        name = 'extract_func_tissues')
    extract_tissue.inputs.ventricles_mask_file = '/scr/sambesi1/workspace/Projects/GluREST/denoise/HarvardOxford-lateral-ventricles-thr25-2mm.nii.gz'



    flow.connect(inputnode,          'wm_mask',                 extract_tissue,   'wm_seg_file'  )
    flow.connect(inputnode,          'gm_mask',                 extract_tissue,   'gm_seg_file'  )
    flow.connect(inputnode,          'csf_mask',                extract_tissue,   'csf_seg_file')
    flow.connect(inputnode,          'func_preproc',            extract_tissue,   'data_file'    )

    # ##########################################################################################################################
    #                                                    # Detrend  #
    # ##########################################################################################################################
    # dt  = Node(util.Function(input_names   =  ['subject','selector','wm_sig_file','csf_sig_file','gm_sig_file','motion_file',
    #                                            'compcor_ncomponents'],
    #                           output_names  = ['residual_file', 'regressors_file'],
    #                           function      = calc_residuals),
    #                           name          = 'residuals_dt')
    # #
    # dt.inputs.selector =  {'compcor' : False , 'wm'     : False , 'csf'    : False , 'gm'        : False,  'global' : False,
    #                         'pc1'    : False , 'motion' : False , 'linear' : True  , 'quadratic'  : False}
    #
    # flow.connect(inputnode,          'func_smoothed',            dt,               'subject'      )
    # #flow.connect(extract_tissue,     'file_wm',                 dt,               'wm_sig_file'  )
    # #flow.connect(extract_tissue,     'file_csf',                dt,               'csf_sig_file' )
    # #flow.connect(extract_tissue,     'file_gm',                 dt,               'gm_sig_file'  )
    # #flow.connect(inputnode,          'motion_pars',             dt,               'motion_file'  )
    # #flow.connect(inputnode,          'compcor_ncomponents',     dt,               'compcor_ncomponents' )
    # flow.connect(dt,                 'residual_file',           outputnode,       'dt_res'          )
    # flow.connect(dt,                 'regressors_file',         outputnode,       'dt_reg'          )
    #
    # ##########################################################################################################################
    #                                                    # Detrend  + Motion #
    # ##########################################################################################################################
    # dt_mc                 =  dt.clone('residuals_dt_mc')
    # dt_mc.inputs.selector =  {'compcor' : False , 'wm'     : False , 'csf'    : False , 'gm'        : False, 'global' : False,
    #                           'pc1'     : False , 'motion' : True , 'linear' : True , 'quadratic' : True}
    #
    # flow.connect(extract_tissue,     'file_wm',                 dt_mc,    'wm_sig_file'  )
    # flow.connect(extract_tissue,     'file_csf',                dt_mc,    'csf_sig_file' )
    # flow.connect(extract_tissue,     'file_gm',                 dt_mc,    'gm_sig_file'  )
    # flow.connect(inputnode,          'func_smoothed',           dt_mc,    'subject'      )
    # flow.connect(inputnode,          'motion_pars',             dt_mc,    'motion_file'  )
    # flow.connect(inputnode,          'compcor_ncomponents',     dt_mc,    'compcor_ncomponents' )
    # flow.connect(dt_mc,              'residual_file',           outputnode,       'dt_mc_res'       )
    # flow.connect(dt_mc,              'regressors_file',         outputnode,       'dt_mc_reg'       )

    # ##########################################################################################################################
    #                                                # Detrend  + Motion  + WMCSF #
    # ##########################################################################################################################
    # dt_mc_wmcsf                 =  dt.clone('residuals_dt_mc_wmcsf')
    # dt_mc_wmcsf.inputs.selector =  {'compcor' : False , 'wm'     : True , 'csf'    : True , 'gm'        : False, 'global' : False,
    #                                 'pc1'     : False , 'motion' : True , 'linear' : True , 'quadratic' : True}
    #
    # flow.connect(extract_tissue,     'file_wm',                 dt_mc_wmcsf,    'wm_sig_file'  )
    # flow.connect(extract_tissue,     'file_csf',                dt_mc_wmcsf,    'csf_sig_file' )
    # flow.connect(extract_tissue,     'file_gm',                 dt_mc_wmcsf,    'gm_sig_file'  )
    # flow.connect(inputnode,          'func_smoothed',                    dt_mc_wmcsf,    'subject'      )
    # flow.connect(inputnode,          'motion_pars',             dt_mc_wmcsf,    'motion_file'  )
    # flow.connect(inputnode,          'compcor_ncomponents',     dt_mc_wmcsf,    'compcor_ncomponents' )
    # flow.connect(dt_mc_wmcsf,        'residual_file',           outputnode,     'dt_mc_wmcsf_res' )
    # flow.connect(dt_mc_wmcsf,        'regressors_file',         outputnode,     'dt_mc_wmcsf_reg' )

    ##########################################################################################################################
                                                   # Detrend  + Motion  + Compcor #
    ##########################################################################################################################

    dt_mc_cc  = Node(util.Function(input_names   =  ['subject','selector','wm_sig_file','csf_sig_file','gm_sig_file','motion_file',
                                               'compcor_ncomponents'],
                                   output_names  = ['residual_file', 'regressors_file'],
                                   function      = calc_residuals),
                                   name          = 'residuals_dt_mc_cc')
    dt_mc_cc.inputs.selector = {'compcor' : True , 'wm'     : False , 'csf'    : False , 'gm'        : False, 'global' : False,
                                'pc1'     : False , 'motion' : True , 'linear' : True , 'quadratic' : True}

    flow.connect(extract_tissue,     'file_wm',                 dt_mc_cc,    'wm_sig_file'  )
    flow.connect(extract_tissue,     'file_csf',                dt_mc_cc,    'csf_sig_file' )
    flow.connect(extract_tissue,     'file_gm',                 dt_mc_cc,    'gm_sig_file'  )
    flow.connect(inputnode,          'func_smoothed',           dt_mc_cc,    'subject'      )
    flow.connect(inputnode,          'motion_pars',             dt_mc_cc,    'motion_file'  )
    flow.connect(inputnode,          'compcor_ncomponents',     dt_mc_cc,    'compcor_ncomponents' )
    flow.connect(dt_mc_cc,           'residual_file',           outputnode,  'dt_mc_cc_res'    )
    flow.connect(dt_mc_cc,           'regressors_file',         outputnode,  'dt_mc_cc_reg'    )

    # ##########################################################################################################################
    #                                          # Detrend  + Motion  + Compcor + GSR #
    # ##########################################################################################################################
    # dt_mc_cc_gsr  = dt.clone('residuals_dt_mc_compcor_gsr')
    # dt_mc_cc_gsr.inputs.selector =   {'compcor' : True , 'wm'      : False , 'csf'    : False , 'gm'       : False, 'global' : True,
    #                                   'pc1'     : False , 'motion' : True  , 'linear' : True , 'quadratic' : True}
    #
    # flow.connect(extract_tissue,     'file_wm',                 dt_mc_cc_gsr,    'wm_sig_file'  )
    # flow.connect(extract_tissue,     'file_csf',                dt_mc_cc_gsr,    'csf_sig_file' )
    # flow.connect(extract_tissue,     'file_gm',                 dt_mc_cc_gsr,    'gm_sig_file'  )
    # flow.connect(inputnode,          'func_smoothed',           dt_mc_cc_gsr,    'subject'      )
    # flow.connect(inputnode,          'motion_pars',             dt_mc_cc_gsr,    'motion_file'  )
    # flow.connect(inputnode,          'compcor_ncomponents',     dt_mc_cc_gsr,    'compcor_ncomponents' )
    # flow.connect(dt_mc_cc_gsr,       'residual_file',           outputnode,       'dt_mc_cc_gs_res' )
    # flow.connect(dt_mc_cc_gsr,       'regressors_file',         outputnode,       'dt_mc_cc_gs_reg' )

    return flow
