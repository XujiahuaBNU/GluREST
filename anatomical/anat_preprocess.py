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
    inputnode    = Node(util.IdentityInterface(fields=['anat', 'anat_gm', 'anat_wm', 'anat_csf', 'anat_first']), name = 'inputnode')
    outputnode   = Node(util.IdentityInterface(fields=['brain','brain_gm', 'brain_wm', 'brain_csf', 'brain_first', 'brain_mask',]),  name = 'outputnode')

    reorient   = Node(interface=preprocess.Resample(),                     name = 'anat_reorient')
    reorient.inputs.orientation = 'RPI'
    reorient.inputs.outputtype = 'NIFTI'

    erode = Node(interface=fsl.ErodeImage(),                                 name = 'anat_preproc')

    reorient_gm    = reorient.clone('anat_preproc_gm')
    reorient_wm    = reorient.clone('anat_preproc_wm')
    reorient_cm    = reorient.clone('anat_preproc_csf')
    reorient_first = reorient.clone('anat_preproc_first')

    make_mask    = Node(interface=fsl.UnaryMaths(),                        name = 'anat_preproc_mask')
    make_mask.inputs.operation = 'bin'

    # connect workflow nodes
    flow.connect(inputnode,    'anat'     , reorient,      'in_file'    )
    flow.connect(inputnode,    'anat_gm'  , reorient_gm,   'in_file'    )
    flow.connect(inputnode,    'anat_wm'  , reorient_wm,   'in_file'    )
    flow.connect(inputnode,    'anat_csf' , reorient_cm,   'in_file'    )
    flow.connect(inputnode,    'anat_first' , reorient_first,'in_file'    )
    flow.connect(reorient,     'out_file' , erode,        'in_file'    )
    flow.connect(erode,        'out_file' , make_mask,    'in_file'    )
    flow.connect(make_mask,    'out_file' , outputnode,   'brain_mask' )

    flow.connect(erode,        'out_file' , outputnode,   'brain'      )
    flow.connect(reorient_gm,  'out_file' , outputnode,   'brain_gm'   )
    flow.connect(reorient_wm,  'out_file' , outputnode,   'brain_wm'   )
    flow.connect(reorient_cm,  'out_file' , outputnode,   'brain_csf'  )
    flow.connect(reorient_first,  'out_file' , outputnode,   'brain_first' )

    return flow
