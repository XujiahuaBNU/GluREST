__author__ = 'kanaan'

import os
from variables.subject_list import *
from motion.motion_metrics import return_DVARS
from utilities.utils import mkdir_path
from nilearn import input_data
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

import nibabel as nb
from scipy.ndimage import center_of_mass




columnx = [ 'stn_tha' , 'stn_thaX' , 'stn_thaL',  'stn_thaR' , 'stn_hip' , 'stn_amg' , 'stn_acc'  , 'stn_accX' , 'stn_lins'  , 'stn_rins' , 'stn_sma' , 'stn_pal', 'stn_strX', 'stn_str', 'stn_cau', 'stn_put', 'stn_nac',
	        'sn_tha'  , 'sn_thaX'  , 'sn_thaL' ,  'sn_thaR'  , 'sn_hip'  , 'sn_amg'  , 'sn_acc'   , 'sn_accX'  , 'sn_lins'   , 'sn_rins'  , 'sn_sma'  , 'sn_pal' , 'sn_strX', 'sn_str', 'sn_cau', 'sn_put', 'sn_nac',
	        'strX_tha', 'strX_thaX', 'strX_thaL', 'strX_thaR', 'strX_hip', 'strX_amg', 'strX_acc' , 'strX_accX', 'strX_lins' , 'strX_rins', 'strX_sma',
            'str_tha' , 'str_thaX' , 'str_thaL' , 'str_thaR' , 'str_hip' , 'str_amg' , 'str_acc'  , 'str_accX' , 'str_lins'  , 'str_rins' , 'str_sma' ,
            'cau_tha' , 'cau_thaX' , 'cau_thaL' , 'cau_thaR' , 'cau_hip' , 'cau_amg' , 'cau_acc'  , 'cau_accX' , 'cau_lins'  , 'cau_rins' , 'cau_sma' , 'cau_pal',
            'put_tha' , 'put_thaX' , 'put_thaL' , 'put_thaR' , 'put_hip' , 'put_amg' , 'put_acc'  , 'put_accX' , 'put_lins'  , 'put_rins' , 'put_sma' , 'put_pal',
            'nac_tha' , 'nac_thaX' , 'nac_thaL' , 'nac_thaR' , 'nac_hip' , 'nac_amg' , 'nac_acc'  , 'nac_accX' , 'nac_lins'  , 'nac_rins' , 'nac_sma' , 'nac_pal',
            'pal_tha' , 'pal_thaX' , 'pal_thaL' , 'pal_thaR' , 'pal_hip' , 'pal_amg' , 'pal_acc'  , 'pal_accX' , 'pal_lins'  , 'pal_rins' , 'pal_sma' ,
                                                               'tha_hip' , 'tha_amg' , 'tha_acc'  , 'tha_accX' , 'tha_lins'  , 'tha_rins' , 'tha_sma' , 'nac_pal',
                                                               'thaX_hip', 'thaX_amg', 'thaX_acc' , 'thaX_accX', 'thaX_lins' , 'thaX_rins', 'thaX_sma', 'thaX_pal',
                                                               'thaL_hip', 'thaL_amg', 'thaL_acc' , 'thaL_accX', 'thaL_lins' , 'thaL_rins', 'thaL_sma', 'thaL_pal',
                                                               'thaR_hip', 'thaR_amg', 'thaR_acc' , 'thaR_accX', 'thaR_lins' , 'thaR_rins', 'thaR_sma', 'thaR_pal',
                                                                                                                 'acc_lins'  , 'acc_rins' , 'acc_sma'  ,
                                                                                                                 'accX_lins' , 'accX_rins', 'accX_sma' ,
                                                                                                                               'sma_rins' , 'sma_lins' ,
                                                                                                                 'fd'        , 'exclude'  , 'dvars' ]

def run_freesurfer_mask_connectivity(pop_name, population, freesurfer_dir, workspace_dir, mrs_datadir, ):

    df = pd.DataFrame(index = [population], columns= columnx )

    for subject in population:

        print '####################### Subject %s' %subject

        subject_dir = os.path.join(workspace_dir, 'GluConnectivity', subject)
        outdir = os.path.join(subject_dir, 'RSFC_CONNECTIVITY')
        mkdir_path(outdir)

        func_pproc = os.path.join(subject_dir, 'functional_native_brain_preproc_FWHM_AROMA_residual_bp/bandpassed_demeaned_filtered.nii.gz') # 2.3 mm
        func_mean  = os.path.join(subject_dir, 'functional_native_brain_preproc_mean/REST_calc_resample_corrected_volreg_maths_tstat.nii')
        func_aroma = os.path.join(subject_dir, 'functional_native_brain_preproc_FWHM_AROMA/denoised_func_data_nonaggr.nii.gz')
        func_gm    = os.path.join(subject_dir, 'functional_native_gm/TISSUE_CLASS_1_GM_OPTIMIZED_resample_flirt_thresh_maths.nii.gz')
        anat_func  = os.path.join(subject_dir, 'anatomical_FUNC2mm_brain/MP2RAGE_DESKULL_RPI_resample_ero_flirt_flirt.nii.gz') # 2.3mm
        anat_func_xfm = os.path.join(subject_dir, 'anatomical_FUNC2mm_xfm/REST_calc_resample_corrected_volreg_maths_tstat_flirt_inv.mat')
        mni2natwarp    =  os.path.join(subject_dir, 'MNI2mm_ANAT_xfm/MP2RAGE_DESKULL_RPI_resample_ero_fieldwarp_inverse.nii.gz')

       #######################################    Grab ATAG masks   #######################################

        STN_LEFT       =  os.path.join(outdir, 'ATAG_STN_LEFT.nii.gz')
        SN_LEFT        =  os.path.join(outdir, 'ATAG_SN_LEFT.nii.gz')
        GPe_LEFT       =  os.path.join(outdir, 'ATAG_GPE_left.nii.gz')
        GPi_LEFT       =  os.path.join(outdir, 'ATAG_GPi_left.nii.gz')

        os.system('applywarp -i %s -r %s -w %s --postmat=%s -o %s' %(mni_stn_left_1mm, anat_func,mni2natwarp, anat_func_xfm, STN_LEFT ))
        os.system('applywarp -i %s -r %s -w %s --postmat=%s -o %s' %(mni_sn_left_1mm, anat_func,mni2natwarp, anat_func_xfm, SN_LEFT ))

        os.system('fslmaths %s -bin %s' %(STN_LEFT, STN_LEFT ))
        os.system('fslmaths %s -bin %s' %(SN_LEFT, SN_LEFT ))

        #######################################    Grab Subcortical masks   #######################################
        print '1. grabbing FIRST Subcortical masks'

        STR =  os.path.join(subject_dir,  'functional_subcortical', 'left_str.nii.gz')
        CAUx=  os.path.join(subject_dir,  'functional_subcortical', 'left_caudate.nii.gz')
        PUT =  os.path.join(subject_dir,  'functional_subcortical', 'left_putamen.nii.gz')
        PAL =  os.path.join(subject_dir,  'functional_subcortical', 'left_pallidum.nii.gz')
        NAC =  os.path.join(subject_dir,  'functional_subcortical', 'left_nacc.nii.gz')
        HIP =  os.path.join(subject_dir,  'functional_subcortical', 'left_hipoocampus.nii.gz')
        AMG =  os.path.join(subject_dir,  'functional_subcortical', 'left_amygdala.nii.gz')

        THA = os.path.join(subject_dir,  'functional_subcortical', 'thalamus.nii.gz')
        lTHA = os.path.join(subject_dir, 'functional_subcortical', 'left_thalamus.nii.gz')
        rTHA = os.path.join(subject_dir, 'functional_subcortical', 'right_thalamus.nii.gz')

        #######################################    Fill Caudate Holes   #######################################

        CAU = os.path.join(subject_dir,  'functional_subcortical', 'left_caudate_fill.nii.gz')

        if not os.path.isfile(CAU):
            os.system('fslmaths %s -fillh %s' %(CAUx, CAU))

        #######################################    Grab svs masks   #######################################
        print '2. grabbing SVS masks'

        svs_acc_src = os.path.join(mrs_datadir, pop_name, subject, 'svs_voxel_mask', '%s%s_ACC_RDA_MASK.nii' %(subject, mrs_datadir[-1]))
        svs_tha_src = os.path.join(mrs_datadir, pop_name, subject, 'svs_voxel_mask', '%s%s_THA_RDA_MASK.nii' %(subject, mrs_datadir[-1]))
        svs_str_src = os.path.join(mrs_datadir, pop_name, subject, 'svs_voxel_mask', '%s%s_STR_RDA_MASK.nii' %(subject, mrs_datadir[-1]))

        svs_acc = os.path.join(outdir, 'svs_acc.nii.gz')
        svs_tha = os.path.join(outdir, 'svs_tha.nii.gz')
        svs_str = os.path.join(outdir, 'svs_str.nii.gz')

        svs_acc_func = os.path.join(outdir, 'svs_acc_func.nii.gz')
        svs_tha_func = os.path.join(outdir, 'svs_tha_func.nii.gz')
        svs_str_func = os.path.join(outdir, 'svs_str_func.nii.gz')

        if not os.path.isfile(svs_acc_func):
            os.system('fslswapdim %s RL PA IS %s' %(svs_acc_src, svs_acc))
            os.system('fslswapdim %s RL PA IS %s' %(svs_tha_src, svs_tha))
            os.system('fslswapdim %s RL PA IS %s' %(svs_str_src, svs_str))

            os.system('flirt -in %s -ref %s -init %s -applyxfm -out %s' %(svs_acc, anat_func, anat_func_xfm, svs_acc_func))
            os.system('flirt -in %s -ref %s -init %s -applyxfm -out %s' %(svs_tha, anat_func, anat_func_xfm, svs_tha_func))
            os.system('flirt -in %s -ref %s -init %s -applyxfm -out %s' %(svs_str, anat_func, anat_func_xfm, svs_str_func))
            os.system('fslmaths %s -thr 0.5 -bin %s' %(svs_acc_func, svs_acc_func))
            os.system('fslmaths %s -thr 0.5 -bin %s' %(svs_tha_func, svs_tha_func))
            os.system('fslmaths %s -thr 0.5 -bin %s' %(svs_str_func, svs_str_func))

        #######################################   Grab freesurfer masks  #######################################
        print '3. grabbing Freesurfer masks'

        os.system('export SUBJECTS_DIR=%s'%(freesurfer_dir))

        t1mgz  = os.path.join(freesurfer_dir, subject, 'mri', 'T1.mgz')
        segmgz = os.path.join(freesurfer_dir, subject, 'mri', 'aparc.a2009s+aseg.mgz')
        t1nii  = os.path.join(outdir, 'freesurfer_T1.nii.gz')
        segnii = os.path.join(outdir, 'freesurfer_seg.nii.gz')

        fs_la_acc       =  os.path.join(outdir, 'freesurfer_seg_la_MCC_11107.nii.gz')  # 11107  ctx_lh_G_and_S_cingul-Mid-Ant
        fs_ra_acc       =  os.path.join(outdir, 'freesurfer_seg_ra_MCC_12107.nii.gz')  # 12107  ctx_lh_G_and_S_cingul-Mid-Ant
        fs_acc          =  os.path.join(outdir, 'freesurfer_seg_aMCC_11107_12107.nii.gz')

        fs_la_insula =  os.path.join(outdir, 'freesurfer_seg_la_INS_11148.nii.gz') # 11148  ctx_lh_S_circular_insula_ant
        fs_ra_insula =  os.path.join(outdir, 'freesurfer_seg_ra_INS_12148.nii.gz') # 12148  ctx_lh_S_circular_insula_ant

        if not os.path.isfile(fs_acc):
            os.system('mri_convert %s %s' %(t1mgz, t1nii))
            os.system('mri_convert %s %s' %(segmgz, segnii))

            os.system('fslmaths %s -thr 11107 -uthr 11107 %s ' %(segnii, fs_la_acc))
            os.system('fslmaths %s -thr 12107 -uthr 12107 %s ' %(segnii, fs_ra_acc))
            os.system('fslmaths %s -add %s -dilM -bin %s' %(fs_la_acc, fs_ra_acc, fs_acc))

            os.system('fslmaths %s -thr 11148 -uthr 11148 -dilM -bin %s' %(segnii, fs_la_insula))
            os.system('fslmaths %s -thr 12148 -uthr 12148 -dilM -bin %s' %(segnii, fs_ra_insula))

        labels_dir = os.path.join(freesurfer_dir, subject, 'label')
        fs_ba6_rh = os.path.join(outdir, 'freesurfer_seg_SMA_BA6_rh.nii.gz')
        fs_ba6_lh = os.path.join(outdir, 'freesurfer_seg_SMA_BA6_lh.nii.gz')
        fs_sma = os.path.join(outdir, 'freesurfer_seg_SMA_BA6.nii.gz')

        if not os.path.isfile(fs_sma):
            os.system('mri_label2vol --label %s/rh.BA6.thresh.label --subject %s --temp %s --regheader %s --o %s' %(labels_dir, subject,t1mgz, t1mgz,fs_ba6_rh))
            os.system('mri_label2vol --label %s/lh.BA6.thresh.label --subject %s --temp %s --regheader %s --o %s' %(labels_dir, subject,t1mgz, t1mgz,fs_ba6_lh))
            os.system('fslmaths  %s -add %s -dilM -dilM %s' %(fs_ba6_rh,fs_ba6_lh, fs_sma))

        #######################################   TRANSFORM Freesurfer masks to native func space   #######################################
        print '4. Transforming Freesurfer masks to native func space'
        t1nii_rpi         =  os.path.join(outdir, 'freesurfer_T1_RPI.nii.gz')
        fs_acc_rpi        =  os.path.join(outdir, 'freesurfer_seg_aMCC_11107_12107_RPI.nii.gz')
        fs_la_insula_rpi  =  os.path.join(outdir, 'freesurfer_seg_la_INS_11148_RPI.nii.gz')
        fs_ra_insula_rpi  =  os.path.join(outdir, 'freesurfer_seg_ra_INS_12148_RPI.nii.gz')
        fs_sma_rpi        =  os.path.join(outdir, 'freesurfer_seg_SMA_BA6_RPI.nii.gz')

        fst1omat          =  os.path.join(outdir, 'freesurfer2func.mat')
        fst1func          =  os.path.join(outdir, 'freesurfer_T1_func.nii.gz')
        fs_acc_func       =  os.path.join(outdir, 'freesurfer_seg_aMCC_11107_12107_func.nii.gz')
        fs_la_insula_func =  os.path.join(outdir, 'freesurfer_seg_la_INS_11148_func.nii.gz')
        fs_ra_insula_func =  os.path.join(outdir, 'freesurfer_seg_ra_INS_11148_func.nii.gz')
        fs_sma_func       =  os.path.join(outdir, 'freesurfer_seg_SMA_BA6_func.nii.gz')

        if not os.path.isfile(t1nii_rpi):
            os.system('fslswapdim %s RL PA IS %s' %(t1nii, t1nii_rpi))
            os.system('fslswapdim %s RL PA IS %s' %(fs_acc, fs_acc_rpi))
            os.system('fslswapdim %s RL PA IS %s' %(fs_la_insula, fs_la_insula_rpi))
            os.system('fslswapdim %s RL PA IS %s' %(fs_ra_insula, fs_ra_insula_rpi))
            os.system('fslswapdim %s RL PA IS %s' %(fs_sma, fs_sma_rpi))
            os.system('flirt -in %s -ref %s -omat %s -dof 6 -out %s -cost mutualinfo' %(t1nii_rpi, anat_func, fst1omat, fst1func))
            os.system('flirt -in %s -ref %s -init %s -applyxfm -out %s' %(fs_acc_rpi, anat_func, fst1omat, fs_acc_func))
            os.system('flirt -in %s -ref %s -init %s -applyxfm -out %s' %(fs_la_insula_rpi, anat_func, fst1omat, fs_la_insula_func))
            os.system('flirt -in %s -ref %s -init %s -applyxfm -out %s' %(fs_ra_insula_rpi, anat_func, fst1omat, fs_ra_insula_func))
            os.system('flirt -in %s -ref %s -init %s -applyxfm -out %s' %(fs_sma_rpi, anat_func, fst1omat, fs_sma_func))

            os.system('fslmaths  %s -thr 0.5 -bin %s' %(fs_acc_func,fs_acc_func))
            os.system('fslmaths  %s -thr 0.5 -bin %s' %(fs_la_insula_func,fs_la_insula_func))
            os.system('fslmaths  %s -thr 0.5 -bin %s' %(fs_ra_insula_func,fs_ra_insula_func))
            os.system('fslmaths  %s -thr 0.5 -bin %s' %(fs_sma_func,fs_sma_func))

        if os.path.isfile(fs_sma_func):
            sma_load = nb.load(fs_sma_func).get_data()
            x, y, z = center_of_mass(sma_load)
            sma_point = os.path.join(outdir, 'sma_point.nii.gz')
            fs_sma_optimized = os.path.join(outdir, 'freesurfer_seg_SMA_BA6_func_opt.nii.gz')

            os.system('fslmaths %s -mul 0 -add 1 -roi %s 1 %s 1 %s 1 0 1 %s -odt float'%(func_mean, x,y,z, sma_point))
            os.system('fslmaths %s -kernel sphere 10 -fmean -dilM -dilM -ero -ero %s -odt float'%(sma_point, fs_sma_optimized))

        #######################################   GET MOTION PARAMS   #######################################
        print '5. Grabbing motion paramaters'

        motion = os.path.join(subject_dir, 'functional_motion_statistics/motion_power_params.txt')
        if os.path.isfile(motion):
            power   =  pd.read_csv(motion) #n, ignore_index = True)
            exclude =  power.loc[subject][' FD_exclude']
            fd      =  power.loc[subject]['Subject']
        if os.path.isfile(func_aroma) and os.path.isfile(func_gm):
            dvars   =  np.mean(return_DVARS(func_aroma, func_gm))
            print dvars

        #######################################   GEN TIMESERIES OF ROIs   #######################################
        print '6. Extracting timeseries and calculating connectivity'

        if os.path.isfile(func_pproc):

            stn_timeseries = input_data.NiftiLabelsMasker(labels_img= STN_LEFT, standardize=True).fit_transform(func_pproc)
            sn_timeseries = input_data.NiftiLabelsMasker(labels_img= SN_LEFT, standardize=True).fit_transform(func_pproc)

            str_timeseries = input_data.NiftiLabelsMasker(labels_img= STR,   standardize=True).fit_transform(func_pproc)
            tha_timeseries = input_data.NiftiLabelsMasker(labels_img= THA,   standardize=True).fit_transform(func_pproc)
            thaL_timeseries = input_data.NiftiLabelsMasker(labels_img= lTHA, standardize=True).fit_transform(func_pproc)
            thaR_timeseries = input_data.NiftiLabelsMasker(labels_img= rTHA, standardize=True).fit_transform(func_pproc)

            cau_timeseries = input_data.NiftiLabelsMasker(labels_img= CAU, standardize=True).fit_transform(func_pproc)
            put_timeseries = input_data.NiftiLabelsMasker(labels_img= PUT, standardize=True).fit_transform(func_pproc)
            pal_timeseries = input_data.NiftiLabelsMasker(labels_img= PAL, standardize=True).fit_transform(func_pproc)
            nac_timeseries = input_data.NiftiLabelsMasker(labels_img= NAC, standardize=True).fit_transform(func_pproc)
            hip_timeseries = input_data.NiftiLabelsMasker(labels_img= HIP, standardize=True).fit_transform(func_pproc)
            amg_timeseries = input_data.NiftiLabelsMasker(labels_img= AMG, standardize=True).fit_transform(func_pproc)

            mACC_timeseries = input_data.NiftiLabelsMasker(labels_img= fs_acc_func, standardize=True).fit_transform(func_pproc)
            lINS_timeseries = input_data.NiftiLabelsMasker(labels_img= fs_la_insula_func, standardize=True).fit_transform(func_pproc)
            rINS_timeseries = input_data.NiftiLabelsMasker(labels_img= fs_ra_insula_func, standardize=True).fit_transform(func_pproc)
            SMA_timeseries = input_data.NiftiLabelsMasker(labels_img= fs_sma_optimized, standardize=True).fit_transform(func_pproc)

            mACCX_timeseries = input_data.NiftiLabelsMasker(labels_img= svs_acc_func, standardize=True).fit_transform(func_pproc)
            strX_timeseries = input_data.NiftiLabelsMasker(labels_img= svs_str_func, standardize=True).fit_transform(func_pproc)
            thaX_timeseries = input_data.NiftiLabelsMasker(labels_img= svs_tha_func, standardize=True).fit_transform(func_pproc)


            print '......calculating Subthalamic Nucleus connectivity'
            df.loc[subject]['stn_pal']  = float(pearsonr(stn_timeseries, pal_timeseries)[0])
            df.loc[subject]['stn_acc']  = float(pearsonr(stn_timeseries, mACC_timeseries)[0])
            df.loc[subject]['stn_tha']  = float(pearsonr(stn_timeseries, tha_timeseries)[0])
            df.loc[subject]['stn_thaX'] = float(pearsonr(stn_timeseries, thaX_timeseries)[0])
            df.loc[subject]['stn_thaL'] = float(pearsonr(stn_timeseries, thaL_timeseries)[0])
            df.loc[subject]['stn_thaR'] = float(pearsonr(stn_timeseries, thaR_timeseries)[0])
            df.loc[subject]['stn_hip']  = float(pearsonr(stn_timeseries, hip_timeseries)[0])
            df.loc[subject]['stn_amg']  = float(pearsonr(stn_timeseries, amg_timeseries)[0])
            df.loc[subject]['stn_accX'] = float(pearsonr(stn_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['stn_lins'] = float(pearsonr(stn_timeseries, lINS_timeseries)[0])
            df.loc[subject]['stn_rins'] = float(pearsonr(stn_timeseries, rINS_timeseries)[0])
            df.loc[subject]['stn_sma']  = float(pearsonr(stn_timeseries, SMA_timeseries)[0])
            df.loc[subject]['stn_strX'] = float(pearsonr(stn_timeseries, strX_timeseries)[0])
            df.loc[subject]['stn_str']  = float(pearsonr(stn_timeseries, str_timeseries)[0])
            df.loc[subject]['stn_cau']  = float(pearsonr(stn_timeseries, cau_timeseries)[0])
            df.loc[subject]['stn_put']  = float(pearsonr(stn_timeseries, put_timeseries)[0])
            df.loc[subject]['stn_nac']  = float(pearsonr(stn_timeseries, nac_timeseries)[0])

            print '......calculating Substantia Nigra connectivity'
            df.loc[subject]['sn_pal']  = float(pearsonr(sn_timeseries, pal_timeseries)[0])
            df.loc[subject]['sn_acc']  = float(pearsonr(sn_timeseries, mACC_timeseries)[0])
            df.loc[subject]['sn_tha']  = float(pearsonr(sn_timeseries, tha_timeseries)[0])
            df.loc[subject]['sn_thaX'] = float(pearsonr(sn_timeseries, thaX_timeseries)[0])
            df.loc[subject]['sn_thaL'] = float(pearsonr(sn_timeseries, thaL_timeseries)[0])
            df.loc[subject]['sn_thaR'] = float(pearsonr(sn_timeseries, thaR_timeseries)[0])
            df.loc[subject]['sn_hip']  = float(pearsonr(sn_timeseries, hip_timeseries)[0])
            df.loc[subject]['sn_amg']  = float(pearsonr(sn_timeseries, amg_timeseries)[0])
            df.loc[subject]['sn_accX'] = float(pearsonr(sn_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['sn_lins'] = float(pearsonr(sn_timeseries, lINS_timeseries)[0])
            df.loc[subject]['sn_rins'] = float(pearsonr(sn_timeseries, rINS_timeseries)[0])
            df.loc[subject]['sn_sma']  = float(pearsonr(sn_timeseries, SMA_timeseries)[0])
            df.loc[subject]['sn_strX'] = float(pearsonr(sn_timeseries, strX_timeseries)[0])
            df.loc[subject]['sn_str']  = float(pearsonr(sn_timeseries, str_timeseries)[0])
            df.loc[subject]['sn_cau']  = float(pearsonr(sn_timeseries, cau_timeseries)[0])
            df.loc[subject]['sn_put']  = float(pearsonr(sn_timeseries, put_timeseries)[0])
            df.loc[subject]['sn_nac']  = float(pearsonr(sn_timeseries, nac_timeseries)[0])

            print '......calculating STR_SVS connectivity'
            df.loc[subject]['strX_acc']  = float(pearsonr(strX_timeseries, mACC_timeseries)[0])
            df.loc[subject]['strX_tha']  = float(pearsonr(strX_timeseries, tha_timeseries)[0])
            df.loc[subject]['strX_thaX'] = float(pearsonr(strX_timeseries, thaX_timeseries)[0])
            df.loc[subject]['strX_thaL'] = float(pearsonr(strX_timeseries, thaL_timeseries)[0])
            df.loc[subject]['strX_thaR'] = float(pearsonr(strX_timeseries, thaR_timeseries)[0])
            df.loc[subject]['strX_hip']  = float(pearsonr(strX_timeseries, hip_timeseries)[0])
            df.loc[subject]['strX_amg']  = float(pearsonr(strX_timeseries, amg_timeseries)[0])
            df.loc[subject]['strX_accX'] = float(pearsonr(strX_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['strX_lins'] = float(pearsonr(strX_timeseries, lINS_timeseries)[0])
            df.loc[subject]['strX_rins'] = float(pearsonr(strX_timeseries, rINS_timeseries)[0])
            df.loc[subject]['strX_sma'] = float(pearsonr(strX_timeseries, SMA_timeseries)[0])

            print '......calculating STR connetivity'
            df.loc[subject]['str_acc']  = float(pearsonr(str_timeseries, mACC_timeseries)[0])
            df.loc[subject]['str_tha']  = float(pearsonr(str_timeseries, tha_timeseries)[0])
            df.loc[subject]['str_thaX'] = float(pearsonr(str_timeseries, thaX_timeseries)[0])
            df.loc[subject]['str_thaL'] = float(pearsonr(str_timeseries, thaL_timeseries)[0])
            df.loc[subject]['str_thaR'] = float(pearsonr(str_timeseries, thaR_timeseries)[0])
            df.loc[subject]['str_hip']  = float(pearsonr(str_timeseries, hip_timeseries)[0])
            df.loc[subject]['str_amg']  = float(pearsonr(str_timeseries, amg_timeseries)[0])
            df.loc[subject]['str_accX'] = float(pearsonr(str_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['str_lins'] = float(pearsonr(str_timeseries, lINS_timeseries)[0])
            df.loc[subject]['str_rins'] = float(pearsonr(str_timeseries, rINS_timeseries)[0])
            df.loc[subject]['str_sma'] = float(pearsonr(str_timeseries, SMA_timeseries)[0])

            print '......calculating CAUDATE connectivity'
            df.loc[subject]['cau_acc']  = float(pearsonr(cau_timeseries, mACC_timeseries)[0])
            df.loc[subject]['cau_tha']  = float(pearsonr(cau_timeseries, tha_timeseries)[0])
            df.loc[subject]['cau_thaX'] = float(pearsonr(cau_timeseries, thaX_timeseries)[0])
            df.loc[subject]['cau_thaL'] = float(pearsonr(cau_timeseries, thaL_timeseries)[0])
            df.loc[subject]['cau_thaR'] = float(pearsonr(cau_timeseries, thaR_timeseries)[0])
            df.loc[subject]['cau_pal']  = float(pearsonr(cau_timeseries, pal_timeseries)[0])
            df.loc[subject]['cau_hip']  = float(pearsonr(cau_timeseries, hip_timeseries)[0])
            df.loc[subject]['cau_amg']  = float(pearsonr(cau_timeseries, amg_timeseries)[0])
            df.loc[subject]['cau_accX'] = float(pearsonr(cau_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['cau_lins'] = float(pearsonr(cau_timeseries, lINS_timeseries)[0])
            df.loc[subject]['cau_rins'] = float(pearsonr(cau_timeseries, rINS_timeseries)[0])
            df.loc[subject]['cau_sma'] = float(pearsonr(cau_timeseries, SMA_timeseries)[0])

            print '......calculating PUTAMEN connectivity'
            df.loc[subject]['put_tha']  = float(pearsonr(put_timeseries, tha_timeseries)[0])
            df.loc[subject]['put_thaX'] = float(pearsonr(put_timeseries, thaX_timeseries)[0])
            df.loc[subject]['put_thaL'] = float(pearsonr(put_timeseries, thaL_timeseries)[0])
            df.loc[subject]['put_thaR'] = float(pearsonr(put_timeseries, thaR_timeseries)[0])
            df.loc[subject]['put_pal']  = float(pearsonr(put_timeseries, pal_timeseries)[0])
            df.loc[subject]['put_hip']  = float(pearsonr(put_timeseries, hip_timeseries)[0])
            df.loc[subject]['put_amg']  = float(pearsonr(put_timeseries, amg_timeseries)[0])
            df.loc[subject]['put_acc']  = float(pearsonr(put_timeseries, mACC_timeseries)[0])
            df.loc[subject]['put_accX'] = float(pearsonr(put_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['put_lins'] = float(pearsonr(put_timeseries, lINS_timeseries)[0])
            df.loc[subject]['put_rins'] = float(pearsonr(put_timeseries, rINS_timeseries)[0])
            df.loc[subject]['put_sma'] = float(pearsonr(put_timeseries, SMA_timeseries)[0])

            print '......calcualting NUCLESUS ACCUMBENS connectivity'
            df.loc[subject]['nac_tha']  = float(pearsonr(nac_timeseries, tha_timeseries)[0])
            df.loc[subject]['nac_thaX'] = float(pearsonr(nac_timeseries, thaX_timeseries)[0])
            df.loc[subject]['nac_thaL'] = float(pearsonr(nac_timeseries, thaL_timeseries)[0])
            df.loc[subject]['nac_thaR'] = float(pearsonr(nac_timeseries, thaR_timeseries)[0])
            df.loc[subject]['nac_pal']  = float(pearsonr(nac_timeseries, pal_timeseries)[0])
            df.loc[subject]['nac_hip']  = float(pearsonr(nac_timeseries, hip_timeseries)[0])
            df.loc[subject]['nac_amg']  = float(pearsonr(nac_timeseries, amg_timeseries)[0])
            df.loc[subject]['nac_acc']  = float(pearsonr(nac_timeseries, mACC_timeseries)[0])
            df.loc[subject]['nac_accX'] = float(pearsonr(nac_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['nac_lins'] = float(pearsonr(nac_timeseries, lINS_timeseries)[0])
            df.loc[subject]['nac_rins'] = float(pearsonr(nac_timeseries, rINS_timeseries)[0])
            df.loc[subject]['nac_sma'] = float(pearsonr(nac_timeseries, SMA_timeseries)[0])

            print '......calcualting PALLIDUM connectivity'
            df.loc[subject]['pal_tha']  = float(pearsonr(pal_timeseries, tha_timeseries)[0])
            df.loc[subject]['pal_thaX'] = float(pearsonr(pal_timeseries, thaX_timeseries)[0])
            df.loc[subject]['pal_thaL'] = float(pearsonr(pal_timeseries, thaL_timeseries)[0])
            df.loc[subject]['pal_thaR'] = float(pearsonr(pal_timeseries, thaR_timeseries)[0])
            df.loc[subject]['pal_hip']  = float(pearsonr(pal_timeseries, hip_timeseries)[0])
            df.loc[subject]['pal_amg']  = float(pearsonr(pal_timeseries, amg_timeseries)[0])
            df.loc[subject]['pal_acc']  = float(pearsonr(pal_timeseries, mACC_timeseries)[0])
            df.loc[subject]['pal_accX'] = float(pearsonr(pal_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['pal_lins'] = float(pearsonr(pal_timeseries, lINS_timeseries)[0])
            df.loc[subject]['pal_rins'] = float(pearsonr(pal_timeseries, rINS_timeseries)[0])
            df.loc[subject]['pal_sma'] = float(pearsonr(pal_timeseries, SMA_timeseries)[0])

            print '......calcualting THA_SVS connectivity'
            df.loc[subject]['thaX_cau']  = float(pearsonr(thaX_timeseries,  cau_timeseries)[0])
            df.loc[subject]['thaX_put']  = float(pearsonr(thaX_timeseries,  put_timeseries)[0])
            df.loc[subject]['thaX_pal']  = float(pearsonr(thaX_timeseries,  pal_timeseries)[0])
            df.loc[subject]['thaX_nac']  = float(pearsonr(thaX_timeseries,  nac_timeseries)[0])
            df.loc[subject]['thaX_hip']  = float(pearsonr(thaX_timeseries,  hip_timeseries)[0])
            df.loc[subject]['thaX_amg']  = float(pearsonr(thaX_timeseries,  amg_timeseries)[0])
            df.loc[subject]['thaX_acc']  = float(pearsonr(thaX_timeseries,  mACC_timeseries)[0])
            df.loc[subject]['thaX_accX'] = float(pearsonr(thaX_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['thaX_lins'] = float(pearsonr(thaX_timeseries, lINS_timeseries)[0])
            df.loc[subject]['thaX_rins'] = float(pearsonr(thaX_timeseries, rINS_timeseries)[0])
            df.loc[subject]['thaX_sma']  = float(pearsonr(thaX_timeseries, SMA_timeseries)[0])

            print '......calcualting THALAMUS FULL connectivity'
            df.loc[subject]['tha_cau']  = float(pearsonr(tha_timeseries,  cau_timeseries)[0])
            df.loc[subject]['tha_put']  = float(pearsonr(tha_timeseries,  put_timeseries)[0])
            df.loc[subject]['tha_pal']  = float(pearsonr(tha_timeseries,  pal_timeseries)[0])
            df.loc[subject]['tha_nac']  = float(pearsonr(tha_timeseries,  nac_timeseries)[0])
            df.loc[subject]['tha_hip']  = float(pearsonr(tha_timeseries,  hip_timeseries)[0])
            df.loc[subject]['tha_amg']  = float(pearsonr(tha_timeseries,  amg_timeseries)[0])
            df.loc[subject]['tha_acc']  = float(pearsonr(tha_timeseries,  mACC_timeseries)[0])
            df.loc[subject]['tha_accX'] = float(pearsonr(tha_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['tha_lins'] = float(pearsonr(tha_timeseries, lINS_timeseries)[0])
            df.loc[subject]['tha_rins'] = float(pearsonr(tha_timeseries, rINS_timeseries)[0])
            df.loc[subject]['tha_sma']  = float(pearsonr(tha_timeseries, SMA_timeseries)[0])

            print '......calcualting THALAMUS RIGHT connectivity'
            df.loc[subject]['thaR_cau']  = float(pearsonr(thaR_timeseries,  cau_timeseries)[0])
            df.loc[subject]['thaR_put']  = float(pearsonr(thaR_timeseries,  put_timeseries)[0])
            df.loc[subject]['thaR_pal']  = float(pearsonr(thaR_timeseries,  pal_timeseries)[0])
            df.loc[subject]['thaR_nac']  = float(pearsonr(thaR_timeseries,  nac_timeseries)[0])
            df.loc[subject]['thaR_hip']  = float(pearsonr(thaR_timeseries,  hip_timeseries)[0])
            df.loc[subject]['thaR_amg']  = float(pearsonr(thaR_timeseries,  amg_timeseries)[0])
            df.loc[subject]['thaR_acc']  = float(pearsonr(thaR_timeseries,  mACC_timeseries)[0])
            df.loc[subject]['thaR_accX'] = float(pearsonr(thaR_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['thaR_lins'] = float(pearsonr(thaR_timeseries, lINS_timeseries)[0])
            df.loc[subject]['thaR_rins'] = float(pearsonr(thaR_timeseries, rINS_timeseries)[0])
            df.loc[subject]['thaR_sma']  = float(pearsonr(thaR_timeseries, SMA_timeseries)[0])

            print '......calcualting THALAMUS LEFT connectivity'
            df.loc[subject]['thaL_cau']  = float(pearsonr(thaL_timeseries,  cau_timeseries)[0])
            df.loc[subject]['thaL_put']  = float(pearsonr(thaL_timeseries,  put_timeseries)[0])
            df.loc[subject]['thaL_pal']  = float(pearsonr(thaL_timeseries,  pal_timeseries)[0])
            df.loc[subject]['thaL_nac']  = float(pearsonr(thaL_timeseries,  nac_timeseries)[0])
            df.loc[subject]['thaL_hip']  = float(pearsonr(thaL_timeseries,  hip_timeseries)[0])
            df.loc[subject]['thaL_amg']  = float(pearsonr(thaL_timeseries,  amg_timeseries)[0])
            df.loc[subject]['thaL_acc']  = float(pearsonr(thaL_timeseries,  mACC_timeseries)[0])
            df.loc[subject]['thaL_accX'] = float(pearsonr(thaL_timeseries, mACCX_timeseries)[0])
            df.loc[subject]['thaL_lins'] = float(pearsonr(thaL_timeseries, lINS_timeseries)[0])
            df.loc[subject]['thaL_rins'] = float(pearsonr(thaL_timeseries, rINS_timeseries)[0])
            df.loc[subject]['thaL_sma']  = float(pearsonr(thaL_timeseries, SMA_timeseries)[0])

            print '......calcualting ACC connectivity'
            df.loc[subject]['acc_lins']  = float(pearsonr(mACC_timeseries, lINS_timeseries)[0])
            df.loc[subject]['acc_rins']  = float(pearsonr(mACC_timeseries, rINS_timeseries)[0])
            df.loc[subject]['acc_sma']  = float(pearsonr(mACC_timeseries, rINS_timeseries)[0])
            df.loc[subject]['accX_lins'] = float(pearsonr(mACCX_timeseries, lINS_timeseries)[0])
            df.loc[subject]['accX_rins'] = float(pearsonr(mACCX_timeseries, rINS_timeseries)[0])
            df.loc[subject]['accX_sma']  = float(pearsonr(mACCX_timeseries, SMA_timeseries)[0])

            print '......calcualting SMA connectivity'
            df.loc[subject]['sma_lins']  = float(pearsonr(mACCX_timeseries, lINS_timeseries)[0])
            df.loc[subject]['sma_rins']  = float(pearsonr(mACCX_timeseries, rINS_timeseries)[0])

            df.loc[subject]['fd']       = fd
            df.loc[subject]['exclude']  = exclude
            df.loc[subject]['dvars']    = dvars

    df.to_csv(os.path.join(workspace_dir,'GluConnectivity' ,'x4_RSFC_df_%s_%s.csv'%(pop_name, mrs_datadir[-1])))
    print 'done'


#run_freesurfer_mask_connectivity('controls', ['HRPP'], freesurfer_dir_a, output_dir_a, mrs_datadir_a)
run_freesurfer_mask_connectivity('controls', controls_a, freesurfer_dir_a, output_dir_a, mrs_datadir_a)
run_freesurfer_mask_connectivity('patients', patients_a, freesurfer_dir_a, output_dir_a, mrs_datadir_a)
run_freesurfer_mask_connectivity('patients', patients_b, freesurfer_dir_b, output_dir_b, mrs_datadir_b)
