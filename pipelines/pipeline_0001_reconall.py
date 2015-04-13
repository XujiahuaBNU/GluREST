__author__ = 'kanaan' 'March 10 2015'


import os
import sys
import nipype.interfaces.freesurfer as fs

assert len(sys.argv)== 2
subject_index = int(sys.argv[1])

'========================================================================================'
# define data dirs and subject list
freesurfer_dir   =    '/xxx/FSUBJECTS'
subjects_list    =    ['XXXX']
data_dir         =    '/xxx/probands'
'========================================================================================'



def tourette_reconall(population, data_dir, freesurfer_dir):
    #count = 0
    #for subject in population:
    #count += 1
    subject = population[subject_index]
    print '========================================================================================'
    print 'Runnning FREESURFER reconall on subject %s' %subject
    #print '%s- Runnning FREESURFER reconall on subject %s' %(count, subject)
    print '========================================================================================'

    brain = os.path.join(data_dir, subject, 'NIFTI', 'MP2RAGE_BRAIN.nii')
    if os.path.isfile(brain):
        if os.path.isfile(os.path.join(freesurfer_dir, subject, 'mri', 'aseg.mgz')):
            print 'Brain already segment......... moving on'
            print 'check data here ---> %s' %(os.path.join(freesurfer_dir, subject))
        else:
            print 'Running recon-all'

            '========================= '
            '   Freesurfer Reconall    '
            '========================= '

            autorecon1 = fs.ReconAll()
            autorecon1.plugin_args             =  {'submit_specs': 'request_memory = 4000'}
            autorecon1.inputs.T1_files         =  brain
            autorecon1.inputs.directive        =  "autorecon1"
            autorecon1.inputs.args             =  "-noskullstrip"
            #autorecon1._interface._can_resume  =  False
            autorecon1.inputs.subject_id       =  subject
            autorecon1.inputs.subjects_dir     = freesurfer_dir
            autorecon1.run()

            os.symlink(os.path.join(freesurfer_dir, subject, "mri", "T1.mgz"),
                    os.path.join(freesurfer_dir, subject, "mri", "brainmask.auto.mgz"))
            os.symlink(os.path.join(freesurfer_dir, subject, "mri", "brainmask.auto.mgz"),
                    os.path.join(freesurfer_dir, subject, "mri", "brainmask.mgz"))

            autorecon_resume                     = fs.ReconAll()
            autorecon_resume.plugin_args         = {'submit_specs': 'request_memory = 4000'}
            autorecon_resume.inputs.args         = "-no-isrunning"
            autorecon_resume.inputs.subject_id   =  subject
            autorecon_resume.inputs.subjects_dir = freesurfer_dir
            autorecon_resume.run()

    else:
        print 'Deskull brain and before you come back'
        raise ValueError('MP2RAGE_BRAIN.nii file for subject %s does not exist')


'######################################################################################################################################'
'######################################################################################################################################'


if __name__ == "__main__":
    tourette_reconall(subjects_list, controls_dir_b, freesurfer_dir_b)



