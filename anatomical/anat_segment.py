__author__ = 'kanaan' '26.11.2014'

from   nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.spm as spm
import nipype.interfaces.fsl as fsl
from utils import  grab_spm_tissues
import nipype.interfaces.utility as util
from nipype.interfaces.afni import preprocess


def fslfast_segment():
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

    thresh_gm                    = Node(fsl.Threshold(), name= 'binned_gm_05')
    thresh_gm.inputs.thresh      = 0.5
    thresh_gm.inputs.args        = '-bin'

    thresh_wm                    = Node(fsl.Threshold(), name= 'binned_wm_05')
    thresh_wm.inputs.thresh      = 0.5
    thresh_wm.inputs.args        = '-bin'

    thresh_csf                   = Node(fsl.Threshold(), name= 'binned_csf_05')
    thresh_csf.inputs.thresh     = 0.5
    thresh_csf.inputs.args       = '-bin'

    reorient_gm                    = Node(interface=preprocess.Resample(), name = 'anat_mask_gm')
    reorient_gm.inputs.orientation = 'RPI'
    reorient_gm.inputs.outputtype  = 'NIFTI_GZ'
    reorient_wm                    = reorient_gm.clone(name= 'anat_mask_wm')
    reorient_csf                   = reorient_gm.clone(name= 'anat_mask_csf')

    flow.connect(inputnode        ,  "anat_preproc"          ,     segment     ,  "in_files"       )
    flow.connect(segment          ,  "partial_volume_files"   ,     select_gm   ,  "inlist"        )
    flow.connect(segment          ,  "partial_volume_files"   ,     select_wm   ,  "inlist"        )
    flow.connect(segment          ,  "partial_volume_files"   ,     select_cm   ,  "inlist"        )
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

# def recon_all_skullstripped(name="recon_all_skullstripped"):
#     """Performs recon-all on voulmes that are already skull stripped.
#     FreeSurfer failes to perform skullstrippig on some volumes (especially
#     MP2RAGE). This can be avoided by doing skullstripping before runnig recon-all
#     (using for example SPECTRE algorithm)
#
#     Inputs::
#            inputspec.T1_files : skullstripped T1_files (mandatory)
#            inputspec.subject_id : freesurfer subject id (optional)
#            inputspec.subjects_dir : freesurfer subjects directory (optional)
#
#     Outputs::
#            outputspec.subject_id : freesurfer subject id
#            outputspec.subjects_dir : freesurfer subjects directory
#     """
#
#     def link_masks(subjects_dir, subject_id):
#         import os
#         os.symlink(os.path.join(subjects_dir, subject_id, "mri", "T1.mgz"),
#                    os.path.join(subjects_dir, subject_id, "mri", "brainmask.auto.mgz"))
#         os.symlink(os.path.join(subjects_dir, subject_id, "mri", "brainmask.auto.mgz"),
#                    os.path.join(subjects_dir, subject_id, "mri", "brainmask.mgz"))
#         return subjects_dir, subject_id
#
#     flow                               = Workflow(name=name)
#
#     inputnode                          = Node(util.IdentityInterface(fields=['subject_id',
#                                                                             'subjects_dir',
#                                                                             'T1_files']),
#                                                                     name  = 'inputspec')
#
#     outputnode                         = Node(util.IdentityInterface(fields=['T1_file',
#                                                                             'aseg_file',
#                                                                             'subject_id',
#                                                                             'subjects_dir']),
#                                                                     name  = 'outputspec')
#
#     autorecon1                         =  Node(fs.ReconAll(), name="autorecon1")
#     autorecon1.plugin_args             =  {'submit_specs': 'request_memory = 4000'}
#     autorecon1.inputs.directive        =  "autorecon1"
#     autorecon1.inputs.args             =  "-noskullstrip"
#     autorecon1._interface._can_resume  =  False
#
#     masks                              = Node(util.Function(input_names  =['subjects_dir', 'subject_id'],
#                                                            output_names =['subjects_dir', 'subject_id'],
#                                                            function     = link_masks),
#                                                            name         = "link_masks")
#
#     autorecon_resume                   = Node(fs.ReconAll(), name="autorecon_resume")
#     autorecon_resume.plugin_args       = {'submit_specs': 'request_memory = 4000'}
#     autorecon_resume.inputs.args       = "-no-isrunning"
#
#
#
#     flow.connect(inputnode        ,  "T1_files"      ,     autorecon1       ,  "T1_files"     )
#     flow.connect(inputnode        ,  "subjects_dir"  ,     autorecon1       ,  "subjects_dir" )
#     flow.connect(inputnode        ,  "subject_id"    ,     autorecon1       ,  "subject_id"   )
#     flow.connect(autorecon1       ,  "subjects_dir"  ,     masks            ,  "subjects_dir" )
#     flow.connect(autorecon1       ,  "subject_id"    ,     masks            ,  "subject_id"   )
#     flow.connect(masks            ,  "subjects_dir"  ,     autorecon_resume ,  "subjects_dir" )
#     flow.connect(masks            ,  "subject_id"    ,     autorecon_resume ,  "subject_id"   )
#     flow.connect(autorecon_resume ,  "subjects_dir"  ,     outputnode       ,  "subjects_dir" )
#     flow.connect(autorecon_resume ,  "subject_id"    ,     outputnode       ,  "subject_id"   )
#
#     #    source = Node(interface=nio.FreeSurferSource(), name = 'fs_import')
#     #
#     # # create brainmask from aparc+aseg with single dilation
#     #    def get_aparc_aseg(files):
#     #        for name in files:
#     #            if 'aparc+aseg' in name:
#     #                return name
#     #
#     #    flow.connect(autorecon_resume ,  "subject_id"                  ,     source       ,  "subject_id"   )
#     #    flow.connect(autorecon_resume ,  "subjects_dir"                ,     source       ,  "subjects_dir" )
#     #    flow.connect(source           ,  "subject_id,(get_aparc_aseg)" ,     outputnode   ,  "aseg_file"   )
#
#     return flow
#
#
#
#
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
#     thresh_gm.inputs.thresh      = 0.9
#     thresh_gm.inputs.args        = '-ero -bin'
#
#     thresh_wm                    = Node(fsl.Threshold(), name= 'bin_wm')
#     thresh_wm.inputs.thresh      = 0.9
#     thresh_wm.inputs.args        = '-ero -bin'
#
#     thresh_csf                   = Node(fsl.Threshold(), name= 'bin_csf')
#     thresh_csf.inputs.thresh     = 0.9
#     thresh_csf.inputs.args       = '-ero -bin'
#
#     reorient_gm                    = Node(interface=preprocess.Resample(), name = 'mask_gm')
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
