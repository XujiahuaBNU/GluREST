__author__ = 'kanaan'

def anatomical_preprocessing():
    from nipype.pipeline.engine import Workflow, Node
    import nipype.interfaces.utility as util
    from nipype.interfaces.afni import preprocess
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.freesurfer as fs

    # define workflow
    flow = Workflow('anat_preproc')
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
