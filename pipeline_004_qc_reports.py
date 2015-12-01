__author__ = 'kanaan' 'Dec 18 2014'
# -*- coding: utf-8 -*-

import os
from glob import glob
import numpy as np
from nipype.algorithms.misc import TSNR
from variables.subject_list import output_dir_a, study_a_list
from utilities.utils import mkdir_path
from quality_control.mriqc import volumes
from quality_control.qc_utils import make_edge
from quality_control import qc_plots
from quality_control.qc_montage import *
import gc
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

#anatomical
#okay ### IMAGE: visual of skull strip
#okay ### IMAGE: GM-WM-CSF on T1
### IMAGE: MNI edge on anatomical

#functional
# IMAGE:T1 Edge on Mean Functional image
# PLOT: func-T1 similarity
# IMAGE: MNI Edge on Mean Functional image


def return_fd_tsnr_dist(population, out_dir, pipeline_name):

    fd_means=[]
    tsnr_files = []
    mask_files =[]
    missing_subjects = []
    for subject in population:

        subject_dir = os.path.join(out_dir, pipeline_name, subject)
        mkdir_path(os.path.join(subject_dir, 'quality_control'))
        qc_dir = os.path.join(subject_dir, 'quality_control')
        subject_dir = os.path.join(out_dir, pipeline_name, subject)

        fd1d = os.path.join(subject_dir, 'functional_motion_FDPower/FD.1D')
        if os.path.isfile(fd1d):
            fd_means.append(np.mean(np.genfromtxt(fd1d)))

        else:
            print subject,'has no fd1d'
            missing_subjects.append(subject)

        os.chdir(qc_dir)
        pp_file = os.path.join(subject_dir, 'functional_native_brain_preproc/REST_calc_resample_corrected_volreg_maths_brain.nii.gz')

        tsnr_file = os.path.join(qc_dir,'REST_calc_resample_corrected_volreg_maths_brain_tsnr.nii.gz')
        mask_file = os.path.join(subject_dir, 'functional_native_brain_preproc_mask/REST_calc_resample_corrected_volreg_maths_brain_mask.nii.gz')

        if os.path.isfile(tsnr_file):
            tsnr_files.append(tsnr_file)
            mask_files.append(mask_file)
        else:
            if os.path.isfile(pp_file):
                tsnr = TSNR()
                tsnr.inputs.in_file =  pp_file
                res = tsnr.run()
                tsnr_files.append(res.outputs.tsnr_file)
            else:
                print subject,'has no functional_native_preproc'



    tsnr_distributions = volumes.get_median_distribution(tsnr_files, mask_files)
    population_fd_means = fd_means


    np.savetxt(os.path.join(out_dir, 'GluConnectivity', 'population_fd_distributions.txt'), population_fd_means)
    np.savetxt(os.path.join(out_dir, 'GluConnectivity', 'population_tsnr_distributions.txt'), tsnr_distributions)

    print 'FD mean=', population_fd_means
    print 'TSNR_distribution=', tsnr_distributions
    print ''

    #return population_fd_means, tsnr_distributions, missing_subjects

def make_qc_reports(population, out_dir, pipeline_name, population_fd_means, tsnr_distributions):

    for subject in population:
        print 'Creating QC-REPORT for subject',subject
        subject_dir = os.path.join(out_dir, pipeline_name, subject)
        qc_dir = os.path.join(subject_dir, 'quality_control')
        os.chdir(qc_dir)

        mni_brain_2mm   = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'
        mni_skull_2mm   = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz'

        anat_preproc =  glob(os.path.join(subject_dir, 'anatomical_native_brain_1mm/*nii*'))[0]
        anat_wm      =  glob(os.path.join(subject_dir, 'anatomical_native_tissue_wm/*nii*'))[0]
        anat_gm      =  glob(os.path.join(subject_dir, 'anatomical_native_tissue_gm/*nii*'))[0]
        anat_csf     =  glob(os.path.join(subject_dir, 'anatomical_native_tissue_csf/*nii*'))[0]
        anat_first   =  glob(os.path.join(subject_dir, 'anatomical_native_tissue_first/*nii*'))[0]



        #func_orig                   = glob(os.path.join(subject_dir, 'functional_native/*nii*'))[0]
        func_preproc                = glob(os.path.join(subject_dir, 'functional_native_brain_preproc/*nii*'))[0]
        func_preproc_mean           = glob(os.path.join(subject_dir, 'functional_native_brain_preproc_mean/*nii*'))[0]
        func_preproc_mask           = glob(os.path.join(subject_dir, 'functional_native_brain_preproc_mask/*nii*'))[0]
        func_preproc_resid          = glob(os.path.join(subject_dir, 'functional_native_brain_preproc_residual_no_AROMA/*nii*'))[0]
        func_preproc_aroma          = glob(os.path.join(subject_dir, 'functional_native_brain_preproc_FWHM_AROMA/*nii*'))[0]
        func_preproc_aroma_resid    = glob(os.path.join(subject_dir, 'functional_native_brain_preproc_FWHM_AROMA_residual/*nii*'))[0]
        func_preproc_aroma_resid_bp = glob(os.path.join(subject_dir, 'functional_native_brain_preproc_FWHM_AROMA_residual_bp/*nii*'))[0]


        func_right_str    =  glob(os.path.join(subject_dir, 'functional_subcortical/right_caudate_maths.nii.gz'))[0]
        func_left_str     =  glob(os.path.join(subject_dir, 'functional_subcortical/left_caudate_maths.nii.gz'))[0]
        func_gm    = glob(os.path.join(subject_dir, 'functional_native_gm/*nii*'))[0]
        func_wm    = glob(os.path.join(subject_dir, 'functional_native_wm/*nii*'))[0]
        func_csf   = glob(os.path.join(subject_dir, 'functional_native_csf/*nii*'))[0]
        func_first = glob(os.path.join(subject_dir, 'functional_native_first/*nii*'))[0]

        anat2mni  =  glob(os.path.join(subject_dir, 'anatomical_MNI2mm_brain/*nii*'))[0]
        anat2func =  glob(os.path.join(subject_dir, 'anatomical_FUNC2mm_brain/*nii*'))[0]
        func2mni  =  glob(os.path.join(subject_dir, 'functional_MNI2mm_brain/*nii*'))[0]

        mov_param    = glob(os.path.join(subject_dir, 'functional_motion_parameters/*1D*'))[0]
        fd1d         = glob(os.path.join(subject_dir, 'functional_motion_FDPower/*1D*'))[0]
        fdexclude    = os.path.join(subject_dir, 'functional_motion_exclude'    )
        power_params = os.path.join(subject_dir, 'functional_motion_statistics' )


        os.chdir(qc_dir)

        # anatomical QC
        # anat_edge = make_edge(anat_preproc)
        #make_montage_sagittal(overlay = anat_edge, underlay = anat_preproc, png_name = 'ANATOMICAL_skull_vis.png', cbar_name= 'red_to_blue')
        # montage_tissues_sagittal(anat_csf, anat_wm, anat_gm, anat_preproc,  'ANATOMICAL_TISSUE_saggital.png')
        #
        # #
        # # anat2mni_edge = make_edge(anat2mni)
        # make_montage_sagittal(overlay = anat2mni_edge, underlay = mni_skull_2mm, png_name = 'ANATOMICAL_MNI_saggital.png', cbar_name= 'red_to_blue')
        # make_montage_axial(overlay = anat2mni_edge, underlay = mni_skull_2mm, png_name = 'ANATOMICAL_MNI_axial.png', cbar_name= 'green')
        #
        # # #func
        # anat2func_edge = make_edge(anat2func)
        # make_montage_sagittal(overlay = anat2func_edge, underlay = func_preproc_mean, png_name = 'FUNCTIONAL_ANAT_skull_vis_saggital.png', cbar_name= 'red')
        # make_montage_axial(overlay = anat2func_edge, underlay = func_preproc_mean, png_name = 'FUNCTIONAL_ANAT_skull_vis_axial.png', cbar_name= 'red')
        #
        # montage_tissues_sagittal(func_csf, func_wm, func_gm, func_preproc,  'FUNCTIONAL_TISSUE_saggital.png')
        # montage_tissues_axial(func_csf, func_wm, func_gm, func_preproc,  'FUNCTIONAL_TISSUE_axial.png')
        #

        report = PdfPages(os.path.join(qc_dir, 'QC_REPORT_%s.PDF'%subject))


        fig = volumes.plot_mosaic(nifti_file = func_preproc_mean,title="Func_Mean", overlay_mask = None,figsize=(8.3, 11.7))
        report.savefig(fig, dpi=300)
        fig.clf()

        fig = volumes.plot_mosaic(nifti_file = func_preproc_mean,title="Func_GM",overlay_mask = func_gm,figsize=(8.3, 11.7))
        report.savefig(fig, dpi=300)
        fig.clf()

        tsnr_file = glob(os.path.join(qc_dir, '*_tsnr.nii.gz'))[0]
        fig = volumes.plot_mosaic(nifti_file = tsnr_file,title="tSNR",overlay_mask = None,figsize=(8.3, 11.7))
        report.savefig(fig, dpi=300)
        fig.clf()

        fig = qc_plots.plot_distrbution_of_values(main_file = tsnr_file,
                                          mask_file = func_preproc_mask,
                                          xlabel = "Subject %s tSNR inside the mask" % subject,
                                          distribution=  tsnr_distributions,
                                          xlabel2= "%s Distribution over median tSNR of all subjects"%subject,
                                          figsize=(11.7,8.3))
        report.savefig(fig, dpi=300)
        fig.clf()

        anat2func_edge = make_edge(anat2func)
        fig = plot_epi_T1_corregistration(func_preproc_mean, anat2func_edge, figsize=(11.7,8.3))
        report.savefig(fig, dpi=300)
        fig.clf()

        fig = qc_plots.plot_FD(fd1d, population_fd_means, subject, figsize = (8.3,8.3))
        report.savefig(fig, dpi=300)
        fig.clf()

        fig = qc_plots.plot_nuisance_residuals(mov_param,
                                         fd1d,
                                         func_preproc,
                                         func_preproc_mask,
                                         func_gm,
                                         func_preproc_aroma_resid,
                                         func_preproc_aroma_resid,
                                         func_preproc_aroma_resid_bp)
        report.savefig(fig, dpi=900)
        fig.clf()

        fig = qc_plots.plot_3d_overlay(func_preproc_mean, func_left_str, 'right_striatum.png')
        report.savefig(fig, dpi=300)
        fig.clf()



        report.close()
        gc.collect()
        plt.close()


if __name__ == "__main__":
    return_fd_tsnr_dist(study_a_list, output_dir_a, 'GluConnectivity')

    fd_dist   = np.genfromtxt(os.path.join(output_dir_a, 'GluConnectivity', 'population_fd_distributions.txt'))
    tsnr_dist = np.genfromtxt(os.path.join(output_dir_a, 'GluConnectivity', 'population_tsnr_distributions.txt'))

    qc_list = ['BM8X', 'GH4T', 'GSNT', 'HCTT', 'HM1X', 'HR8T', 'KDET', 'LL5T', 'LMIT',
               'MJBT', 'NP4T', 'PAHT', 'RB1T', 'RJBT', 'RJJT', 'SDCT', 'SI5T', 'SJBT',
               'SS1X', 'STQT', 'TJ5T', 'TR4T', 'TSCT', 'TV1T', 'ZT5T', 'PU2T', 'GSAT', 'BH5T',
               'SMVX', 'EC9T', 'RA7T', 'HM2X', 'KO4T', 'GHAT', 'BATP', 'BE9P', 'CB4P', 'CF1P',
               'CM5P', 'DF2P', 'EB2P', 'EW3P', 'FL3P', 'FMEP',
               'GSAP', 'HHQP', 'HJEP', 'HMXP', 'HRPP', 'HSPP', 'KDDP', 'LA9P', 'LT5P',
               'NL2P', 'NT6P', 'PC5P', 'RA9P', 'RL7P', 'RMJP',  'SA5U', 'SBQP', 'SGKP',
               'SM6U', 'STDP', 'SULP', 'TSEP', 'TT3P', 'WO2P', 'YU1P', 'THCP'] # 'RRDP', 'GF3T', 'RMNT',

    make_qc_reports(qc_list, output_dir_a, 'GluConnectivity', fd_dist, tsnr_dist )
