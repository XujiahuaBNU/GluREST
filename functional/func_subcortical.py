__author__ = 'kanaan'


from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.afni import preprocess
from nipype.interfaces.ants import N4BiasFieldCorrection
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from utilities.utils import return_list


def make_func_subcortical_masks(name = 'func_subcortical'):


    # Define Workflow
    flow        = Workflow(name=name)
    inputnode   = Node(util.IdentityInterface(fields=['func_first']),
                           name='inputnode')
    outputnode  = Node(util.IdentityInterface(fields=['left_nacc', 'left_amygdala',  'left_caudate',  'left_hipoocampus',  'left_pallidum',  'left_putamen', 'left_thalamus',
                                                      'right_nacc','right_amygdala', 'right_caudate', 'right_hipoocampus', 'right_pallidum', 'right_putamen','right_thalamus',
                                                      'midbrain', 'right_striatum', 'left_striatum']),
                           name = 'outputnode')

    left_nacc = Node(interface=fsl.ExtractROI(), name = 'left_nacc')
    left_nacc.inputs.t_min  = 0
    left_nacc.inputs.t_size = 1
    left_nacc.inputs.roi_file = 'left_nacc.nii.gz'

    left_amygdala = Node(interface=fsl.ExtractROI(), name = 'left_amygdala')
    left_amygdala.inputs.t_min  = 1
    left_amygdala.inputs.t_size = 1
    left_amygdala.inputs.roi_file = 'left_amygdala.nii.gz'

    left_caudate = Node(interface=fsl.ExtractROI(), name = 'left_caudate')
    left_caudate.inputs.t_min  = 2
    left_caudate.inputs.t_size = 1
    left_caudate.inputs.roi_file = 'left_caudate.nii.gz'

    left_hipoocampus = Node(interface=fsl.ExtractROI(), name = 'left_hipoocampus')
    left_hipoocampus.inputs.t_min  = 3
    left_hipoocampus.inputs.t_size = 1
    left_hipoocampus.inputs.roi_file = 'left_hipoocampus.nii.gz'

    left_pallidum = Node(interface=fsl.ExtractROI(), name = 'left_pallidum')
    left_pallidum.inputs.t_min  = 4
    left_pallidum.inputs.t_size = 1
    left_pallidum.inputs.roi_file = 'left_pallidum.nii.gz'

    left_putamen = Node(interface=fsl.ExtractROI(), name = 'left_putamen')
    left_putamen.inputs.t_min  = 5
    left_putamen.inputs.t_size = 1
    left_putamen.inputs.roi_file = 'left_putamen.nii.gz'

    left_thalamus = Node(interface=fsl.ExtractROI(), name = 'left_thalamus')
    left_thalamus.inputs.t_min  = 6
    left_thalamus.inputs.t_size = 1
    left_thalamus.inputs.roi_file = 'left_thalamus.nii.gz'

    ###############

    right_nacc = Node(interface=fsl.ExtractROI(), name = 'right_nacc')
    right_nacc.inputs.t_min  = 7
    right_nacc.inputs.t_size = 1
    right_nacc.inputs.roi_file = 'right_nacc.nii.gz'

    right_amygdala = Node(interface=fsl.ExtractROI(), name = 'right_amygdala')
    right_amygdala.inputs.t_min  = 8
    right_amygdala.inputs.t_size = 1
    right_amygdala.inputs.roi_file = 'right_amygdala.nii.gz'

    right_caudate = Node(interface=fsl.ExtractROI(), name = 'right_caudate')
    right_caudate.inputs.t_min  = 9
    right_caudate.inputs.t_size = 1
    right_caudate.inputs.roi_file = 'right_caudate.nii.gz'

    right_hipoocampus = Node(interface=fsl.ExtractROI(), name = 'right_hipoocampus')
    right_hipoocampus.inputs.t_min  = 10
    right_hipoocampus.inputs.t_size = 1
    right_hipoocampus.inputs.roi_file = 'right_hipoocampus.nii.gz'

    right_pallidum = Node(interface=fsl.ExtractROI(), name = 'right_pallidum')
    right_pallidum.inputs.t_min  = 11
    right_pallidum.inputs.t_size = 1
    right_pallidum.inputs.roi_file = 'right_pallidum.nii.gz'

    right_putamen = Node(interface=fsl.ExtractROI(), name = 'right_putamen')
    right_putamen.inputs.t_min  = 12
    right_putamen.inputs.t_size = 1
    right_putamen.inputs.roi_file = 'right_putamen.nii.gz'

    right_thalamus = Node(interface=fsl.ExtractROI(), name = 'right_thalamus')
    right_thalamus.inputs.t_min  = 13
    right_thalamus.inputs.t_size = 1
    right_thalamus.inputs.roi_file = 'right_thalamus.nii.gz'

    midbrain = Node(interface=fsl.ExtractROI(), name = 'midbrain')
    midbrain.inputs.t_min  = 14
    midbrain.inputs.t_size = 1
    midbrain.inputs.roi_file = 'midbrain.nii.gz'

    flow.connect( inputnode  ,   'func_first'   ,   left_nacc,       'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   left_amygdala,   'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   left_caudate,    'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   left_hipoocampus,'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   left_pallidum,   'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   left_putamen,    'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   left_thalamus,   'in_file'     )

    flow.connect( inputnode  ,   'func_first'   ,   right_nacc,       'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   right_amygdala,   'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   right_caudate,    'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   right_hipoocampus,'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   right_pallidum,   'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   right_putamen,    'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   right_thalamus,   'in_file'     )
    flow.connect( inputnode  ,   'func_first'   ,   midbrain,         'in_file'     )

    flow.connect( left_nacc        ,   'roi_file'   ,outputnode   ,   'left_nacc'       )
    flow.connect( left_amygdala    ,   'roi_file'   ,outputnode   ,   'left_amygdala'   )
    flow.connect( left_caudate     ,   'roi_file'   ,outputnode   ,   'left_caudate'    )
    flow.connect( left_hipoocampus ,   'roi_file'   ,outputnode   ,   'left_hipoocampus')
    flow.connect( left_pallidum    ,   'roi_file'   ,outputnode   ,   'left_pallidum')
    flow.connect( left_putamen     ,   'roi_file'   ,outputnode   ,   'left_putamen'    )
    flow.connect( left_thalamus    ,   'roi_file'   ,outputnode   ,   'left_thalamus'   )
    flow.connect( right_nacc       ,   'roi_file'   ,outputnode   ,   'right_nacc'       )
    flow.connect( right_amygdala   ,   'roi_file'   ,outputnode   ,   'right_amygdala'   )
    flow.connect( right_caudate    ,   'roi_file'   ,outputnode   ,   'right_caudate'    )
    flow.connect( right_hipoocampus,   'roi_file'   ,outputnode   ,   'right_hipoocampus')
    flow.connect( right_pallidum   ,   'roi_file'   ,outputnode   ,   'right_pallidum')
    flow.connect( right_putamen    ,   'roi_file'   ,outputnode   ,   'right_putamen'    )
    flow.connect( right_thalamus   ,   'roi_file'   ,outputnode   ,   'right_thalamus'   )
    flow.connect( midbrain         ,   'roi_file'   ,outputnode   ,   'midbrain'         )


    # add images together
    right_striatum = Node(interface=fsl.MultiImageMaths(), name = 'right_striatum')
    right_striatum.inputs.op_string = '-add %s -add %s -bin'
    right_striatum.out_file         = 'right_striatum.nii.gz'
    list_R_str = Node(util.Function(input_names = ['file_1', 'file_2'],
                                    output_names= ['list'],
                                    function    = return_list),
                                    name        = 'list_str_r')

    flow.connect( right_pallidum     ,   'roi_file'   ,list_R_str       ,   'file_1'         )
    flow.connect( right_putamen      ,   'roi_file'   ,list_R_str       ,   'file_2'         )
    flow.connect( right_caudate      ,   'roi_file'   ,right_striatum   ,   'in_file'        )
    flow.connect( list_R_str         ,   'list'       ,right_striatum   ,   'operand_files'  )
    flow.connect( right_striatum     ,   'out_file'   ,outputnode       ,   'right_striatum' )


    left_striatum = Node(interface=fsl.MultiImageMaths(), name = 'left_striatum')
    left_striatum.inputs.op_string = '-add %s -add %s'
    left_striatum.out_file         = 'left_striatum.nii.gz'
    list_L_str =  list_R_str.clone('list_str_l')

    flow.connect( left_pallidum     ,   'roi_file'   ,list_L_str       ,   'file_1'         )
    flow.connect( left_putamen      ,   'roi_file'   ,list_L_str       ,   'file_2'         )
    flow.connect( left_caudate      ,   'roi_file'   ,left_striatum    ,   'in_file'        )
    flow.connect( list_L_str        ,   'list'       ,left_striatum    ,   'operand_files'  )
    flow.connect( left_striatum     ,   'out_file'   ,outputnode       ,   'left_striatum'  )


    return flow
