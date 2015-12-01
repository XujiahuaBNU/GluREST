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





def anat_segment_fslfast():
    '''
    Inputs:
        Preprocessed Brain

    Workflow:
        1. Segmentation with FSL-FAST
        2. Thresholding of probabilistic maps and binarization
        3. reorienation to RPI

    Returns:
        GM bin RPI
        WM bin RPI
        CSF bin RPI

    '''
    flow                    = Workflow(name='anat_segmentation')

    inputnode               = Node(util.IdentityInterface(fields=['anat_preproc']), name  = 'inputnode')

    outputnode              = Node(util.IdentityInterface(fields=['anat_mask_gm',
                                                                  'anat_mask_wm',
                                                                  'anat_mask_csf']),     name  = 'outputnode')

    segment                         = Node(interface=fsl.FAST(), name='segment')
    segment.inputs.img_type         = 1
    segment.inputs.segments         = True
    segment.inputs.probability_maps = True
    segment.inputs.out_basename     = 'segment'

    select_gm  = Node(util.Select(), name = 'select_1')
    select_gm.inputs.set(inlist=[1, 2, 3], index=[1])

    select_wm  = Node(util.Select(), name = 'select_2')
    select_wm.inputs.set(inlist=[1, 2, 3], index=[2])

    select_cm  = Node(util.Select(), name = 'select_3')
    select_cm.inputs.set(inlist=[1, 2, 3], index=[0])

    thresh_gm                    = Node(fsl.Threshold(), name= 'binned_gm')
    thresh_gm.inputs.thresh      = 0.5
    thresh_gm.inputs.args        = '-bin'

    thresh_wm                    = Node(fsl.Threshold(), name= 'binned_wm')
    thresh_wm.inputs.thresh      = 0.5
    thresh_wm.inputs.args        = '-bin'

    thresh_csf                   = Node(fsl.Threshold(), name= 'binned_csf')
    thresh_csf.inputs.thresh     = 0.5
    thresh_csf.inputs.args       = '-bin'

    reorient_gm                    = Node(interface=preprocess.Resample(), name = 'anat_mask_gm')
    reorient_gm.inputs.orientation = 'RPI'
    reorient_gm.inputs.outputtype  = 'NIFTI_GZ'
    reorient_wm                    = reorient_gm.clone(name= 'anat_mask_wm')
    reorient_csf                   = reorient_gm.clone(name= 'anat_mask_csf')

    flow.connect(inputnode        ,  "anat_preproc"          ,     segment     ,  "in_files"       )
    flow.connect(segment          ,  "partial_volume_files"  ,     select_gm   ,  "inlist"        )
    flow.connect(segment          ,  "partial_volume_files"  ,     select_wm   ,  "inlist"        )
    flow.connect(segment          ,  "partial_volume_files"  ,     select_cm   ,  "inlist"        )
    flow.connect(select_gm        ,  'out'                   ,     thresh_gm   ,  'in_file'        )
    flow.connect(select_wm        ,  'out'                   ,     thresh_wm   ,  'in_file'        )
    flow.connect(select_cm        ,  'out'                   ,     thresh_csf  ,  'in_file'        )
    flow.connect(thresh_gm        ,  "out_file"              ,     reorient_gm ,  "in_file"        )
    flow.connect(thresh_wm        ,  "out_file"              ,     reorient_wm ,  "in_file"        )
    flow.connect(thresh_csf       ,  "out_file"              ,     reorient_csf,  "in_file"        )
    flow.connect(reorient_gm      ,  "out_file"              ,     outputnode  ,  "anat_mask_gm"        )
    flow.connect(reorient_wm      ,  "out_file"              ,     outputnode  ,  "anat_mask_wm"        )
    flow.connect(reorient_csf     ,  "out_file"              ,     outputnode  ,  "anat_mask_csf"       )

    return flow


def anat_segment_subcortical():
    '''
    Inputs:
        Preprocessed Brain

    Workflow:
        1. Segmentation with FSL-FIRST
        2. Thresholding of probabilistic maps and binarization
        3. reorienation to RPI

    Returns:
        GM bin RPI
        WM bin RPI
        CSF bin RPI

    '''
    flow                    = Workflow(name='anat_segmentation_subcortical')
    inputnode               = Node(util.IdentityInterface(fields=['anat_preproc']), name  = 'inputnode')
    outputnode              = Node(util.IdentityInterface(fields=['anat_subcortical']), name  = 'outputnode')

    def run_fsl_first(anat):
        import os
        os.system("run_first_all -b -v -i %s -o ANAT_SUBCORTICAL"%anat)
        curdir =os.curdir
        subcortical = os.path.join(curdir, 'ANAT_SUBCORTICAL_all_fast_firstseg.nii.gz')
        return subcortical

    first = Node(util.Function(input_names=['anat'],
                               output_names=['subcortical'],
                               function=run_fsl_first),
                               name = 'anat_segmentation_first')

    flow.connect(inputnode   , 'anat_preproc'  , first           , 'anat')
    flow.connect(first       , 'subcortical'   , outputnode      , 'anat_subcortical')

    return flow



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
    #print preprocssed_all
    #print 'Number of subjects to concatenate in the time dimension for Group ICA = %s' %len(preprocssed_all)

    #concat_func = os.path.join(group_ica_dir, 'preproc_all_subjects.nii.gz')

    #if not os.path.isfile(concat_func):
    # merger = Merge()
    # merger.inputs.in_files = preprocssed_all
    # #merger.inputs.in_files = ['/scr/sambesi4/workspace/project_GluRest/OUT_DIR_A/GluConnectivity/BM8X/functional_MNI2mm_preproc_FWHM_AROMA_residual_high_pass/bandpassed_demeaned_filtered.nii.gz', '/scr/sambesi4/workspace/project_GluRest/OUT_DIR_A/GluConnectivity/GH4T/functional_MNI2mm_preproc_FWHM_AROMA_residual_high_pass/bandpassed_demeaned_filtered.nii.gz']
    # merger.inputs.dimension = 't'
    # merger.inputs.output_type = 'NIFTI_GZ'
    # merger.inputs.tr = 1.4
    # merger.inputs.merged_file =  '%s/preproc_all_fslmerge.nii.gz'  %group_ica_dir
    # print merger.cmdline
    # merger.run()

    # merge = afni.Merge()
    # #merge.inputs.in_files= preprocssed_all
    # merge.inputs.in_files= ['/scr/sambesi4/workspace/project_GluRest/OUT_DIR_A/GluConnectivity/BM8X/functional_MNI2mm_preproc_FWHM_AROMA_residual_high_pass/bandpassed_demeaned_filtered.nii.gz', '/scr/sambesi4/workspace/project_GluRest/OUT_DIR_A/GluConnectivity/GH4T/functional_MNI2mm_preproc_FWHM_AROMA_residual_high_pass/bandpassed_demeaned_filtered.nii.gz']
    # merge.inputs.doall = True
    # merge.inputs.outputtype = 'NIFTI_GZ'
    # merge.inputs.out_file = '%s/preproc_all_merged_afni_test2subs.nii.gz'  %group_ica_dir
    # merge.base_dir = working_dir
    # print merge.cmdline
    # merge.run()
