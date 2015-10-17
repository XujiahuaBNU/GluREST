__author__ = 'kanaan' 'March 10 2015'

import os
import nipype.interfaces.freesurfer as fs
from variables.subject_list import *
import sys

assert len(sys.argv)== 2
subject_index=int(sys.argv[1])

def tourette_reconall(population, data_dir, freesurfer_dir):
	#for subject in population:
	subject = population[subject_index]
	print '========================================================================================'
	print '                 Runnning FREESURFER reconall on subject %s' %subject
	print '========================================================================================'

	brain = os.path.join(data_dir, subject, 'NIFTI', 'MP2RAGE_BRAIN.nii')
	if os.path.isfile(brain):
		if os.path.isfile(os.path.join(freesurfer_dir, subject, 'mri', 'aparc.a2009s+aseg.mgz')):
			print 'Brain already segmented......... moving on'
			print 'check data here ---> %s' %(os.path.join(freesurfer_dir, subject))
		else:
			fs_subdir = os.path.join(freesurfer_dir, subject)
			print 'recon all not complete.. deleting incomplete fs_subdir'
			os.system('rm -rf %s'%fs_subdir)
			print 'Running recon-all'

			'========================= '
			'   Freesurfer Reconall    '
			'========================= '

			autorecon1 = fs.ReconAll()
			autorecon1.plugin_args             =  {'submit_specs': 'request_memory = 4000'}
			autorecon1.inputs.T1_files         =  brain
			autorecon1.inputs.directive        =  "autorecon1"
			autorecon1.inputs.args             =  "-noskullstrip"
			#####autorecon1._interface._can_resume  =  False
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
		print 'Deskull brain before and then come back'
		raise ValueError('MP2RAGE_BRAIN.nii file for subject %s does not exist')

	#print needed
    #print len(needed)  
'######################################################################################################################################'
'######################################################################################################################################'
pa_1 = ['DF2P', 'GSAP',  'HHQP', 'HRPP', 'SULP' ]
pa_2 = ['HSPP', 'LA9P',  'NT6P', 'RA9P', 'THCP' ] 
pa_3 = ['RL7P', 'RRDP',  'SA5U', 'STDP' ]


if __name__ == "__main__":
	# tourette_reconall(pa_3, patients_datadir_a, freesurfer_dir_a)
	#tourette_reconall(controls_a, controls_datadir_a, freesurfer_dir_a)
	#tourette_reconall(patients_a, patients_datadir_a, freesurfer_dir_a)
	tourette_reconall(controls_b, controls_datadir_b, freesurfer_dir_b)
	#tourette_reconall(patients_b, patients_datadir_b, freesurfer_dir_b)

