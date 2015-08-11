__author__ = 'kanaan'

#Pruim, 2015

def runICA(func, outDir, mask, dim, TR):
    """ This function runs MELODIC and merges the mixture modeled thresholded ICs into a single 4D nifti file

    Parameters
    ---------------------------------------------------------------------------------
    func:		Full path to the fMRI data file (nii.gz) on which MELODIC should be run
    outDir:		Full path of the output directory
    melDirIn:	Full path of the MELODIC directory in case it has been run before, otherwise define empty string
    mask:		Full path of the mask to be applied during MELODIC
    dim:		Dimensionality of ICA
    TR:		TR (in seconds) of the fMRI data

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    melodic.ica		MELODIC directory
    melodic_IC_thr.nii.gz	merged file containing the mixture modeling thresholded Z-statistical maps located in melodic.ica/stats/ """

    # Import needed modules
    import os
    import commands

    # Define the 'new' MELODIC directory and predefine some associated files
    melDir = os.path.join(outDir,'melodic.ica')
    melIC = os.path.join(melDir,'melodic_IC.nii.gz')
    melICmix = os.path.join(melDir,'melodic_mix')
    melICthr = os.path.join(outDir,'melodic_IC_thr.nii.gz')

    print '  -  Run Melodic - .'

    # Run MELODIC
    os.system(' '.join(['melodic', '--in=' + func, '--outdir=' + melDir, '--mask=' + mask, '--dim=' + str(dim), '--Ostats --nobet --mmthresh=0.5 --report', '--tr=' + str(TR)]))

    # Get number of components
    cmd = ' '.join(['fslinfo', melIC, '| grep dim4 | head -n1 | awk \'{print $2}\''])
    nrICs=int(float(commands.getoutput(cmd)))

    # Merge mixture modeled thresholded spatial maps.
    # Note! In case that mixture modeling did not converge, the file will contain two spatial maps.
    # The latter being the results from a simple null hypothesis test. In that case, this map will have to be used (first one will be empty).

    for i in range(1,nrICs+1):
        # Define thresholded zstat-map file
        zTemp = os.path.join(melDir,'stats','thresh_zstat' + str(i) + '.nii.gz')
        cmd = ' '.join(['fslinfo', zTemp, '| grep dim4 | head -n1 | awk \'{print $2}\''])
        lenIC=int(float(commands.getoutput(cmd)))

        # Define zeropad for this IC-number and new zstat file
        cmd = ' '.join(['zeropad', str(i), '4'])
        ICnum=commands.getoutput(cmd)
        zstat = os.path.join(outDir,'thr_zstat' + ICnum)

        # Extract last spatial map within the thresh_zstat file
        os.system(' '.join(['fslroi',
            zTemp,		# input
            zstat,		# output
            str(lenIC-1),	# first frame
            '1']))		# number of frames

    # Merge and subsequently remove all mixture modeled Z-maps within the output directory
    os.system(' '.join(['fslmerge','-t',melICthr, os.path.join(outDir,'thr_zstat????.nii.gz')]))

    os.system('rm ' + os.path.join(outDir,'thr_zstat????.nii.gz'))

    # Apply the mask to the merged file (in case a melodic-directory was predefined and run with a different mask)
    os.system(' '.join(['fslmaths', melICthr, '-mas ' + mask, melICthr]))