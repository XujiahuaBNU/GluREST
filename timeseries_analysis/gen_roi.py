__author__ = 'kanaan'


def create_seeds(seedOutputLocation, seed_specification_file, FSLDIR):
    # *** Borrowed from CPAC-0.3.8 **
    # https://github.com/FCP-INDI/C-PAC

    import commands
    import os
    import re

    seed_specifications = [line.rstrip('\r\n') for line in open(seed_specification_file, 'r').readlines() if (not line.startswith('#') and not (line == '\n')) ]

    seed_resolutions = {}

    for specification in seed_specifications:

        seed_label, x, y, z, radius, resolution = re.split(r'[\t| |,]+', specification)

        if resolution not in seed_resolutions.keys():
            seed_resolutions[resolution] = []
        seed_resolutions[resolution].append((seed_label, x, y, z, radius, resolution))

    return_roi_files = []
    for resolution_key in seed_resolutions:


        index = 0
        seed_files = []
        for roi_set in seed_resolutions[resolution_key]:


            seed_label, x, y, z, radius, resolution = roi_set
            if not os.path.exists(seedOutputLocation):
                os.makedirs(seedOutputLocation)

            print 'checking if file exists ', '%s/data/standard/MNI152_T1_%s_brain.nii.gz' % (FSLDIR, resolution)
            assert(os.path.exists('%s/data/standard/MNI152_T1_%s_brain.nii.gz' % (FSLDIR, resolution)))
            cmd = "echo %s %s %s | 3dUndump -prefix %s.nii.gz -master %s/data/standard/MNI152_T1_%s_brain.nii.gz \
-srad %s -orient LPI -xyz -" % (x, y, z, os.path.join(seedOutputLocation, str(index) + '_' + seed_label + '_' + resolution), FSLDIR, resolution, radius)

            print cmd
            try:
                commands.getoutput(cmd)
                seed_files.append((os.path.join(seedOutputLocation, '%s.nii.gz' % (str(index) + '_' + seed_label + '_' + resolution)), seed_label))
                print seed_files
            except:
                raise

            index += 1

        print index, ' ', seed_files
        seed_str = ' '
        intensities = ''
        for idx in range(0, index):

            seed, intensity = seed_files[idx]

            intensities += intensity + '_'

            cmd = "3dcalc -a %s -expr 'a*%s' -prefix %s" % (seed, intensity, os.path.join(seedOutputLocation, 'ic_' + os.path.basename(seed)))
            print cmd
            try:
                commands.getoutput(cmd)

                seed_str += "%s " % os.path.join(seedOutputLocation, 'ic_' + os.path.basename(seed))
            except:
                raise


        cmd = '3dMean  -prefix %s.nii.gz -sum %s' % (os.path.join(seedOutputLocation, 'rois_' + resolution), seed_str)
        print cmd
        try:
            commands.getoutput(cmd)
        except:
            raise

        #try:
        #    cmd = 'rm -f %s' % seed_str
        #    print cmd
        #    commands.getoutput(cmd)
        #    for seed, intensity in seed_files:
        #       try:
        #           cmd = 'rm -f  ' + seed
        #            print cmd
        #           commands.getoutput(cmd)
        #      except:
        #            raise
        #except:
        #    raise
        return_roi_files.append(os.path.join(seedOutputLocation, 'rois_' + resolution + '.nii.gz'))

    print return_roi_files
    return return_roi_files