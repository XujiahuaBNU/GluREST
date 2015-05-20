__author__ = 'kanaan--- Jan 15 2015'


def smooth_data():
    from nipype.pipeline.engine import Node, Workflow
    import nipype.interfaces.utility as util
    import nipype.interfaces.fsl as fsl

    flow        = Workflow('func2mni_preprocessed_fwhm')

    inputnode   = Node(util.IdentityInterface(fields=['func_data']),
                       name = 'inputnode')

    outputnode  =  Node(util.IdentityInterface(fields=['func_smoothed']),
                       name = 'outputnode')

    smooth      = Node(interface=fsl.Smooth(), name='func_smooth_fwhm_4')
    smooth.inputs.fwhm                 = 4.0
    smooth.terminal_output             = 'file'

    flow.connect(inputnode, 'func_data'      , smooth      , 'in_file'    )
    flow.connect(smooth,    'smoothed_file'  , outputnode  , 'func_smoothed'   )


    return flow