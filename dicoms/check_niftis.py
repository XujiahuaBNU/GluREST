__author__ = 'kanaan'

import os

controls_dir_A   = '/a/projects/nmr093a/probands/'
patients_dir_A   = '/a/projects/nmr093a/patients/'

population_controls_A = []
population_patients_A = []

def get_subject_list(population_dir, population_new_list):
    for folder in os.listdir(population_dir):
        if len(folder) ==4:
            population_new_list.append(folder)

get_subject_list(controls_dir_A, population_controls_A)
get_subject_list(patients_dir_A, population_patients_A)

print population_controls_A
print population_patients_A


# controls_A =  ['SJVT', 'GF3T', 'GSNT', 'BM8X', 'SI5T', 'TSCT', 'HR8T', 'HCTT', 'RB1T',
#                'GH4T', 'ZT5T', 'RJBT', 'SJBT', 'TJ5T', 'WJ3T', 'KDET', 'PAHT', 'LMIT',
#                'NP4T', 'SDCT', 'TR4T', 'TV1T', 'RJJT']


# patients_A = ['TSEP', 'FMEP', 'WO2P', 'LJ9P', 'RSIP', 'HRPP', 'GSAP', 'HMXP',
#               'FL3P', 'HHQP', 'HJEP', 'LA9P', 'LT5P', 'EW3P', 'KDDP', 'EB2P',
#               'CM5P', 'SULP', 'BE9P', 'DF2P', 'PC5P', 'HSPP', 'THCP', 'SA5U',
#               'NT6P', 'CF1P']

def find_all_files(data_dir, population):
    # generic location function
    def locate(string, directory, dir_list):
        for file in dir_list:
            x=[]
            if string in file:
                x = os.path.join(directory,file)
                return x

    for subject in population:
        nifti_dir    = os.path.join(data_dir, subject, 'NIFTI')
        nifti_dir_ls = os.listdir(nifti_dir)

        rest_file = locate('REST.nii', nifti_dir, nifti_dir_ls)
        anat_file = locate('MP2RAGE_UNI.nii', nifti_dir, nifti_dir_ls)
        dwi_file = locate('DWI.nii', nifti_dir, nifti_dir_ls)

        if rest_file is not None:
            continue
        elif rest_file is None:
            print 'Resting file for subject %s is missing' %subject
            print 'dir = %s' %nifti_dir

        if anat_file is not None:
            continue
        elif anat_file is None:
            print 'Anatomical file for subject %s is missing' %subject

        if dwi_file is not None:
            continue
        elif dwi_file is None:
            print 'Anatomical file for subject %s is missing' %subject

find_all_files(controls_dir_A, population_controls_A)
find_all_files(patients_dir_A, population_patients_A)