__author__ = 'kanaan'

import os
from variables.subject_list import *

#population = ['CM5P', 'DF2P', 'FL3P', 'FMEP', 'HHQP', 'HJEP', 'HRPP', 'HSPP', 'KDDP', 'LA9P', 'LT5P', 'SULP', 'TSEP', 'WO2P']

dir_a = '/SCR4/workspace/project_GluRest/OUT_DIR_A/GluConnectivity'
#dir_b = '/SCR4/workspace/project_GluRest/OUT_DIR_B/GluConnectivity'

output_dir = '/SCR4/workspace/project_GluRest/OUT_DIR_B/GluConnectivity/DESIGN'

def get_ecm(population):
    for subject in population:
        ecm_a = os.path.join(dir_a, subject, 'FAST_ECM/FAST_ECM.nii')
        #ecm_b = os.path.join(dir_b, subject, 'FAST_ECM/FAST_ECM.nii')
        #out_file = os.path.join(dir_b, subject, 'ECM/fECM_A_sub_B.nii.gz')

        #os.system('fslmaths %s -sub %s %s' %(ecm_a, ecm_b, out_file))

        if os.path.isfile(ecm_a):
            print ecm_a#

get_ecm(patients_a)
get_ecm(controls_a)