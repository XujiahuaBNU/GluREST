__author__ = 'kanaan'

import os
from utilities.utils import mkdir_path
from variables.subject_list import study_a_list, mni_brain_2mm, working_dir, output_dir
import nibabel as nb
import commands
from nilearn.decomposition.canica import CanICA

def run_CanICA_NILEAN(output_dir, working_dir, population, n_components = 20):

    #create group ica outputdir
    mkdir_path(os.path.join(working_dir, 'CANICA_GROUP_ICA'))
    canica_dir = os.path.join(working_dir, 'CANICA_GROUP_ICA')

    # grab subjects
    preprocssed_all =[]
    for subject in population:
        preprocssed_subject = os.path.join(output_dir, subject, 'xxxx.nii.gz')
        preprocssed_all.append(preprocssed_subject)

    canica = CanICA(n_components=n_components,
                    smoothing_fwhm= 0.,
                    memory= 'nilearn_cashe',
                    memory_level= 5,
                    threshold = 3.,
                    verbose = 10,
                    random_state=10)
    canica.fit(preprocssed_all)

    # save data
    components_img = canica.masker_.inverse_transform(canica.components_)
    components_img.to_filename(os.path.join(canica_dir, 'canica_IC.nii.gz'))

