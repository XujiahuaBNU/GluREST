__author__ = 'kanaan'
from nipype.pipeline.engine import Workflow, Node
import nipype.interfaces.utility as util
from nipype.interfaces.afni import preprocess
import nipype.interfaces.freesurfer as fs


def grabber_util(name):
    '''
    Method to grab data from from out_dir
    inputnode
        input.base_dir
        input.subject_id
        input.folder_name
        input.file_name
        input.pipeline_name
    outputnode
        output.file
    '''

    def grab_filepath(base_dir,subject_id, folder_name, file_name_string):
        import os
        for file in os.listdir(os.path.join(base_dir,
                                             subject_id,
                                             folder_name)):
            if file_name_string in file:
                file_path =os.path.join(base_dir,
                                        subject_id,
                                        folder_name,
                                        file)
                return file_path


    flow      = Workflow(name=name)
    inputnode = Node(util.IdentityInterface(fields = ['subject_id','folder_name', 'file_name_string', 'base_dir']),
                        name = 'inputnode')
    outputnode = Node(util.IdentityInterface(fields=['out_file']),
                        name = 'outputnode')


    grabber  = Node(util.Function(input_names      = ['base_dir', 'subject_id', 'folder_name', 'file_name_string'],
                                  output_names     = ['out_file'],
                                  function         = grab_filepath),
                                  name             = 'grabber')

    flow.connect(inputnode   , 'subject_id'       ,   grabber,     'subject_id'         )
    flow.connect(inputnode   , 'base_dir'         ,   grabber,     'base_dir'           )
    flow.connect(inputnode   , 'folder_name'      ,   grabber,     'folder_name'        )
    flow.connect(inputnode   , 'file_name_string' ,   grabber,     'file_name_string'   )
    flow.connect(grabber     , 'out_file'         ,   outputnode,  'out_file'           )


    return flow


def locate(string, directory):
        import os
        for file in os.listdir(directory):
            x=[]
            if string in file:
                x = os.path.join(directory,file)
                return x



#assert len(sys.argv)== 2
#subject_index=int(sys.argv[1])

# if len(sys.argv) == 1:
#     print 'No argument provided, running internal subject list'
# else:
#     mode=sys.argv[1]
#     if mode == '-sub':
#         population=[sys.argv[2]]
#     elif mode == '-sublist':
#         with open(sys.argv[2], 'r') as f:
#             population = [line.strip() for line in f]
