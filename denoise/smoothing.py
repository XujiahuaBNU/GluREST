__author__ = 'kanaan--- Jan 15 2015'


def smooth_data():
    from nipype.pipeline.engine import Node, Workflow
    import nipype.interfaces.utility as util
    import nipype.interfaces.fsl as fsl

    flow        = Workflow('func_smoothed')

    inputnode   = Node(util.IdentityInterface(fields=['func_data']),
                       name = 'inputnode')

    outputnode  =  Node(util.IdentityInterface(fields=['func_smoothed']),
                       name = 'outputnode')

    smooth      = Node(interface=fsl.Smooth())
    smooth.inputs.fwhm      = 4


    flow.connect(inputnode, 'func_data'      , smooth      , 'in_file'    )
    flow.connect(smooth,    'smoothed_file'  , outputnode  , 'func_smoothed'   )


    return flow