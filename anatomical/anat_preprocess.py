__author__ = 'kanaan' '26.11.2014'


import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
from nipype.interfaces.afni import preprocess
from  nipype.pipeline.engine import Node, Workflow


def anatomical_preprocessing():
    '''
    Inputs:
        MP2RAGE Skull stripped image using Spectre-2010

    Workflow:
        1. reorient to RPI
        2. create a brain mask

    Returns:
        brain
        brain_mask

    '''
    # define workflow
    flow = Workflow('anat_preprocess')
    inputnode    = Node(util.IdentityInterface(fields=['anat']),                  name = 'inputnode')
    outputnode   = Node(util.IdentityInterface(fields=['brain', 'brain_mask',]),  name = 'outputnode')

    reorient   = Node(interface=preprocess.Resample(),                     name = 'anat_preproc')
    reorient.inputs.orientation = 'RPI'
    reorient.inputs.outputtype = 'NIFTI'

    make_mask    = Node(interface=fsl.UnaryMaths(),                        name = 'anat_preproc_mask')
    make_mask.inputs.operation = 'bin'

    # connect workflow nodes
    flow.connect(inputnode,    'anat'     , reorient,     'in_file'    )
    flow.connect(reorient,     'out_file' , make_mask,    'in_file'    )
    flow.connect(reorient,     'out_file' , outputnode,   'brain'      )
    flow.connect(make_mask,    'out_file' , outputnode,   'brain_mask' )

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


