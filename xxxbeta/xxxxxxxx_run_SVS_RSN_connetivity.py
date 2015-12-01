__author__ = 'kanaan'

%matplotlib inline
import os
from nilearn import input_data, plotting, image
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
datadir = '/SCR4/workspace/project_GluRest/OUT_DIR_A/GluConnectivity/'
datadir_mrs = '/SCR3/workspace/project_GLUTAMATE/'

controls_a = ['BM8X', 'GH4T', 'GSNT', 'HCTT', 'HM1X', 'HR8T', 'KDET', 'LL5T', 'LMIT',
              'MJBT', 'NP4T', 'PAHT', 'RB1T', 'RJBT', 'RJJT',  'SDCT', 'SI5T', 'SJBT',
              'SS1X', 'STQT', 'TJ5T', 'TR4T', 'TSCT', 'TV1T', 'ZT5T', 'PU2T', 'GSAT', 'BH5T',
              'SMVX', 'EC9T', 'RA7T', 'HM2X', 'KO4T', 'GHAT']  #'RMNT', 'GF3T',

patients_a = ['BATP', 'BE9P', 'CB4P', 'CF1P', 'CM5P', 'DF2P', 'EB2P', 'EW3P', 'FL3P', 'FMEP',
              'GSAP', 'HHQP', 'HJEP', 'HMXP', 'HRPP', 'HSPP', 'KDDP', 'LA9P',         'LT5P',
              'NL2P', 'NT6P', 'PC5P', 'RA9P', 'RL7P', 'RMJP', 'SA5U', 'SBQP', 'SGKP',
              'SM6U', 'STDP', 'SULP', 'TSEP', 'TT3P', 'WO2P', 'YU1P', 'THCP']#'RRDP'

test = ['LMIT']

pproc_string = 'functional_MNI4mm_preproc_FWHM_AROMA_residual_high_pass/bandpassed_demeaned_filtered.nii.gz'

dlpfc_right     = '/scr/sambesi1/ROI/rDLPFCphere.nii.gz'
dlpfc_left      = '/scr/sambesi1/ROI/lDLPFCphere.nii.gz'
m1_left         = '/scr/sambesi1/ROI/M1lsphere.nii.gz'
m1_right        = '/scr/sambesi1/ROI/M1rsphere.nii.gz'
sma             = '/scr/sambesi1/ROI/SMAsphere.nii.gz'
acc             = '/scr/sambesi1/ROI/ACCsphere.nii.gz'
caudate         = 'functional_subcortical/left_caudate.nii.gz'
putamen         = 'functional_subcortical/left_putamen.nii.gz'
pallidum        = 'functional_subcortical/left_pallidum.nii.gz'
rtha            = 'functional_subcortical/left_thalamus.nii.gz'
ltha            = 'functional_subcortical/right_thalamus.nii.gz'

mni             = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'
warp_mni        = 'anatomical_MNI2mm_xfm/MP2RAGE_DESKULL_RPI_resample_ero_fieldwarp.nii.gz'
premat_anat     = 'functional_ANAT2mm_xfm/REST_calc_resample_corrected_volreg_maths_tstat_flirt.mat'


acc_coord = (0,22,28)
tha_coord = (-10, -18, 10)
str_coord = (-18,22,28)
sma_coord = (0, -4, 58)



def get_fc(population, datadir, out_name):

    df = pd.DataFrame(index = [population], columns=['acc_tha', 'acc_str', 'tha_str',
                                                     'tha_sma', 'sma_str',
                                                     'm1l_tha', 'm1r_tha',
                                                     'm1l_str', 'm1r_str',
                                                     'rpfc_tha', 'lpfc_tha',
                                                     'rpfc_str', 'lpfc_str',
                                                     'rpfc_acc', 'lpfc_acc',
                                                     'fd', 'exclude'])

    for subject in population:
        # cau     = os.path.join(datadir, subject, caudate)
        # put     = os.path.join(datadir, subject, putamen)
        # pal     = os.path.join(datadir, subject, pallidum)

        thar    = os.path.join(datadir, subject, rtha)
        thal    = os.path.join(datadir, subject, ltha)

        pproc   = os.path.join(datadir, subject, pproc_string)
        warp    = os.path.join(datadir, subject, warp_mni)
        premat  = os.path.join(datadir, subject, premat_anat)
        outdir = os.path.join(datadir,subject, 'functional_subcortical')


        motion = os.path.join(datadir, subject, 'functional_motion_statistics/motion_power_params.txt')
        if os.path.isfile(motion):
            power = pd.read_csv(motion)#n, ignore_index = True)

        exclude =  power.loc[subject][' FD_exclude']
        fd =  power.loc[subject]['Subject']


        if os.path.isfile(thar):
            thalamus = os.path.join(outdir, 'thalamus.nii.gz')
            striatum = os.path.join(outdir, 'left_str.nii.gz')

            if not os.path.isfile(thalamus) or not os.path.isfile(striatum):
                os.system('fslmaths %s -add %s -add %s -fillh %s/left_str'%(cau, put, pal,outdir))
                os.system('fslmaths %s -add %s -fillh %s/thalamus'%(thar, thal, outdir))

            mni_thalamus = os.path.join(outdir, 'MNI2mm_THALAMUS.nii.gz')
            mni_striatum = os.path.join(outdir, 'MNI2mm_LEFT_STR.nii.gz')

            os.system(' '.join([ 'applywarp',
                                 '--in='     +  thalamus,
                                 '--ref='    +  mni,
                                 '--out='    +  mni_thalamus,
                                 '--warp='   +  warp,
                                 '--premat=' +  premat]))

            mni_thalamus_bin = os.path.join(outdir, 'MNI2mm_THALAMUS_bin.nii.gz')
            os.system('fslmaths %s -thr 0.5 -bin %s'%(mni_thalamus, mni_thalamus_bin))

            os.system(' '.join([ 'applywarp',
                                 '--in='     +  striatum,
                                 '--ref='    +  mni,
                                 '--out='    +  mni_striatum,
                                 '--warp='   +  warp,
                                 '--premat=' +  premat]))

            mni_striatum_bin = os.path.join(outdir, 'MNI2mm_LEFT_STR_bin.nii.gz')
            os.system('fslmaths %s -thr 0.5 -bin %s'%(mni_striatum, mni_striatum_bin))




            str_timeseries = input_data.NiftiLabelsMasker(labels_img = mni_striatum_bin, standardize=True).fit_transform(pproc)
            tha_timeseries = input_data.NiftiLabelsMasker(labels_img = mni_thalamus_bin, standardize=True).fit_transform(pproc)
            acc_timeseries = input_data.NiftiLabelsMasker(labels_img = acc, standardize=True).fit_transform(pproc)
            sma_timeseries = input_data.NiftiLabelsMasker(labels_img = sma, standardize=True).fit_transform(pproc)
            m1l_timeseries = input_data.NiftiLabelsMasker(labels_img = m1_left, standardize=True).fit_transform(pproc)
            m1r_timeseries = input_data.NiftiLabelsMasker(labels_img = m1_right, standardize=True).fit_transform(pproc)
            rpfc_timeseries = input_data.NiftiLabelsMasker(labels_img = dlpfc_left, standardize=True).fit_transform(pproc)
            lpfc_timeseries = input_data.NiftiLabelsMasker(labels_img = dlpfc_right, standardize=True).fit_transform(pproc)

            print str_timeseries.shape
            print tha_timeseries.shape
            print lpfc_timeseries.shape

            acc_tha = float(pearsonr(acc_timeseries, tha_timeseries)[0])
            acc_str = float(pearsonr(acc_timeseries, str_timeseries)[0])
            tha_str = float(pearsonr(tha_timeseries, str_timeseries)[0])

            tha_sma = float(pearsonr(sma_timeseries, tha_timeseries)[0])
            sma_str = float(pearsonr(sma_timeseries, str_timeseries)[0])

            m1l_tha = float(pearsonr(m1l_timeseries, tha_timeseries)[0])
            m1l_str = float(pearsonr(m1l_timeseries, str_timeseries)[0])

            m1r_tha = float(pearsonr(m1r_timeseries, tha_timeseries)[0])
            m1r_str = float(pearsonr(m1r_timeseries, str_timeseries)[0])

            rpfc_tha = float(pearsonr(rpfc_timeseries , tha_timeseries)[0])
            rpfc_str = float(pearsonr(rpfc_timeseries , str_timeseries)[0])
            lpfc_tha = float(pearsonr(lpfc_timeseries , tha_timeseries)[0])
            lpfc_str = float(pearsonr(lpfc_timeseries , str_timeseries)[0])

            rpfc_acc = float(pearsonr(rpfc_timeseries , acc_timeseries )[0])
            lpfc_acc = float(pearsonr(lpfc_timeseries , acc_timeseries )[0])

            df.loc[subject]['acc_tha'] = acc_tha
            df.loc[subject]['acc_str'] = acc_str
            df.loc[subject]['tha_str'] = tha_str
            df.loc[subject]['tha_sma'] = tha_sma
            df.loc[subject]['sma_str'] = sma_str
            df.loc[subject]['m1l_tha'] = m1l_tha
            df.loc[subject]['m1l_str'] = m1l_str
            df.loc[subject]['m1r_tha'] = m1r_tha
            df.loc[subject]['m1r_str'] = m1r_str

            df.loc[subject]['rpfc_tha'] = rpfc_tha
            df.loc[subject]['rpfc_str'] = rpfc_str
            df.loc[subject]['lpfc_tha'] = lpfc_tha
            df.loc[subject]['lpfc_str'] = lpfc_str

            df.loc[subject]['rpfc_acc'] = rpfc_acc
            df.loc[subject]['lpfc_acc'] = lpfc_acc


            df.loc[subject]['fd']      = fd
            df.loc[subject]['exclude'] = exclude
    else:
        print 'Subject %s dumped'%subject
    df.to_csv(os.path.join(datadir,'%s.csv'%out_name))
    return df


get_fc(test, 'test')
