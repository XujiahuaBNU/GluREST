__author__ = 'kanaan' '22.05.2015'


from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util


# forked from CPAC version 0.39 https://github.com/FCP-INDI/C-PAC
def extract_tissue_data(data_file,
                        ventricles_mask_file,
                        wm_seg_file, csf_seg_file, gm_seg_file,
                        wm_threshold=0.0, csf_threshold=0.0, gm_threshold=0.0):
    import numpy as np
    import nibabel as nb
    import os
    #from nuisance_utils import erode_mask, safe_shape

    #ventricles_mask_file = '/scr/sambesi1/workspace/Projects/GluREST/denoise/HarvardOxford-lateral-ventricles-thr25-2mm.nii.gz'


    def erode_mask(data):
        mask = data != 0
        eroded_mask = np.zeros_like(data, dtype='bool')
        max_x, max_y, max_z = data.shape
        x,y,z = np.where(data != 0)
        for i in range(x.shape[0]):
            if (max_x-1) == x[i] or \
               (max_y-1) == y[i] or \
               (max_z-1) == z[i] or \
               x[i] == 0 or \
               y[i] == 0 or \
               z[i] == 0:
                eroded_mask[x[i],y[i],z[i]] = False
            else:
                eroded_mask[x[i],y[i],z[i]] = mask[x[i], y[i], z[i]] * \
                                              mask[x[i] + 1, y[i], z[i]] * \
                                              mask[x[i], y[i] + 1, z[i]] * \
                                              mask[x[i], y[i], z[i] + 1] * \
                                              mask[x[i] - 1, y[i], z[i]] * \
                                              mask[x[i], y[i] - 1, z[i]] * \
                                              mask[x[i], y[i], z[i] - 1]

        eroded_data = np.zeros_like(data)
        eroded_data[eroded_mask] = data[eroded_mask]

        return eroded_data


    def safe_shape(*vol_data):
        """
        Checks if the volume (first three dimensions) of multiple ndarrays
        are the same shape.

        Parameters
        ----------
        vol_data0, vol_data1, ..., vol_datan : ndarray
            Volumes to check

        Returns
        -------
        same_volume : bool
            True only if all volumes have the same shape.
        """
        same_volume = True

        first_vol_shape = vol_data[0].shape[:3]
        for vol in vol_data[1:]:
            same_volume &= (first_vol_shape == vol.shape[:3])

        return same_volume

    print 'Tissues extraction thresholds wm %d, csf %d, gm %d' % (wm_threshold,
                                                                  csf_threshold,
                                                                  gm_threshold)

    try:
        data = nb.load(data_file).get_data().astype('float64')
    except:
        raise MemoryError('Unable to load %s' % data_file)


    try:
        lat_ventricles_mask = nb.load(ventricles_mask_file).get_data().astype('float64')
    except:
        raise MemoryError('Unable to load %s' % lat_ventricles_mask)


    if not safe_shape(data, lat_ventricles_mask):
        raise ValueError('Spatial dimensions for data and the lateral ventricles mask do not match')

    try:
        wm_seg = nb.load(wm_seg_file).get_data().astype('float64')
    except:
        raise MemoryError('Unable to load %s' % wm_seg)


    if not safe_shape(data, wm_seg):
        raise ValueError('Spatial dimensions for data, white matter segment do not match')

    wm_mask = erode_mask(wm_seg > wm_threshold)
    wm_sigs = data[wm_mask]
    file_wm = os.path.join(os.getcwd(), 'wm_signals.npy')
    np.save(file_wm, wm_sigs)
    del wm_sigs

    try:
        csf_seg = nb.load(csf_seg_file).get_data().astype('float64')
    except:
        raise MemoryError('Unable to load %s' % csf_seg)


    if not safe_shape(data, csf_seg):
        raise ValueError('Spatial dimensions for data, cerebral spinal fluid segment do not match')

    # Only take the CSF at the lateral ventricles as labeled in the Harvard
    # Oxford parcellation regions 4 and 43
    csf_mask = (csf_seg > csf_threshold)*(lat_ventricles_mask==1)
    csf_sigs = data[csf_mask]
    file_csf = os.path.join(os.getcwd(), 'csf_signals.npy')
    np.save(file_csf, csf_sigs)
    del csf_sigs


    try:
        gm_seg = nb.load(gm_seg_file).get_data().astype('float64')
    except:
        raise MemoryError('Unable to load %s' % gm_seg)


    if not safe_shape(data, gm_seg):
        raise ValueError('Spatial dimensions for data, gray matter segment do not match')


    gm_mask = erode_mask(gm_seg > gm_threshold)
    gm_sigs = data[gm_mask]
    file_gm = os.path.join(os.getcwd(), 'gm_signals.npy')
    np.save(file_gm, gm_sigs)
    del gm_sigs



    nii = nb.load(wm_seg_file)
    wm_mask_file = os.path.join(os.getcwd(), 'wm_mask.nii.gz')
    csf_mask_file = os.path.join(os.getcwd(), 'csf_mask.nii.gz')
    gm_mask_file = os.path.join(os.getcwd(), 'gm_mask.nii.gz')
    nb.Nifti1Image(wm_mask, header=nii.get_header(), affine=nii.get_affine()).to_filename(wm_mask_file)
    nb.Nifti1Image(csf_mask, header=nii.get_header(), affine=nii.get_affine()).to_filename(csf_mask_file)
    nb.Nifti1Image(gm_mask, header=nii.get_header(), affine=nii.get_affine()).to_filename(gm_mask_file)

    return file_wm, file_csf, file_gm


# forked from CPAC version 0.39 https://github.com/FCP-INDI/C-PAC
def calc_residuals(subject,
                   selector,
                   wm_sig_file = None,
                   csf_sig_file = None,
                   gm_sig_file = None,
                   motion_file = None,
                   compcor_ncomponents = 0):
    """
    Calculates residuals of nuisance regressors for every voxel for a subject.

    Parameters
    ----------
    subject : string
        Path of a subject's realigned nifti file.
    selector : dictionary
        Dictionary of selected regressors.  Keys are  represented as a string of the regressor name and keys
        are True/False.  See notes for an example.
    wm_mask_file : string, optional
        Path to subject's white matter mask (in the same space as the subject's functional file)
    csf_mask_file : string, optional
        Path to subject's cerebral spinal fluid mask (in the same space as the subject's functional file)
    gm_mask_file : string, optional
        Path to subject's grey matter mask (in the same space as the subject's functional file)
    compcor_ncomponents : integer, optional
        The first `n` principal of CompCor components to use as regressors.  Default is 0.

    Returns
    -------
    residual_file : string
        Path of residual file in nifti format
    regressors_file : string
        Path of csv file of regressors used.  Filename corresponds to the name of each
        regressor in each column.

    Notes
    -----

    Example of selector parameter:

    >>> selector = {'compcor' : True,
    >>> 'wm' : True,
    >>> 'csf' : True,
    >>> 'gm' : True,
    >>> 'global' : True,
    >>> 'pc1' : True,
    >>> 'motion' : True,
    >>> 'linear' : True,
    >>> 'quadratic' : True}


    """
    import numpy as np
    import nibabel as nb
    import os
    import scipy
    #from nuisance_utils import calc_compcor_components


    def calc_compcor_components(data, nComponents, wm_sigs, csf_sigs):
        import scipy.signal as signal

        wmcsf_sigs = np.vstack((wm_sigs, csf_sigs))

        # filter out any voxels whose variance equals 0
        print 'Removing zero variance components'
        wmcsf_sigs = wmcsf_sigs[wmcsf_sigs.std(1)!=0,:]

        if wmcsf_sigs.shape.count(0):
            print 'No wm or csf signals left after removing those with zero variance'
            raise IndexError

        print 'Detrending and centering data'
        Y = signal.detrend(wmcsf_sigs, axis=1, type='linear').T
        Yc = Y - np.tile(Y.mean(0), (Y.shape[0], 1))
        Yc = Yc / np.tile(np.array(Y.std(0)).reshape(1,Y.shape[1]), (Y.shape[0],1))

        print 'Calculating SVD decomposition of Y*Y\''
        U, S, Vh = np.linalg.svd(Yc)

        return U[:,:nComponents]


    nii = nb.load(subject)
    data = nii.get_data().astype(np.float64)
    global_mask = (data != 0).sum(-1) != 0


    #Check and define regressors which are provided from files
    if wm_sig_file is not None:
        wm_sigs = np.load(wm_sig_file)
        if wm_sigs.shape[1] != data.shape[3]:
            raise ValueError('White matter signals length %d do not match data timepoints %d' % (wm_sigs.shape[1], data.shape[3]))
    if csf_sig_file is not None:
        csf_sigs = np.load(csf_sig_file)
        if csf_sigs.shape[1] != data.shape[3]:
            raise ValueError('CSF signals length %d do not match data timepoints %d' % (csf_sigs.shape[1], data.shape[3]))
    if gm_sig_file is not None:
        gm_sigs = np.load(gm_sig_file)
        if gm_sigs.shape[1] != data.shape[3]:
            raise ValueError('Grey matter signals length %d do not match data timepoints %d' % (gm_sigs.shape[1], data.shape[3]))

    if motion_file is not None:
        motion = np.genfromtxt(motion_file)
        if motion.shape[0] != data.shape[3]:
            raise ValueError('Motion parameters %d do not match data timepoints %d' % (motion.shape[0], data.shape[3]) )

    #Calculate regressors
    regressor_map = {'constant' : np.ones((data.shape[3],1))}
    if(selector['compcor']):
        print 'compcor_ncomponents ', compcor_ncomponents
        regressor_map['compcor'] = calc_compcor_components(data, compcor_ncomponents, wm_sigs, csf_sigs)

    if(selector['wm']):
        regressor_map['wm'] = wm_sigs.mean(0)

    if(selector['csf']):
        regressor_map['csf'] = csf_sigs.mean(0)

    if(selector['gm']):
        regressor_map['gm'] = gm_sigs.mean(0)

    if(selector['global']):
        regressor_map['global'] = data[global_mask].mean(0)

    if(selector['pc1']):
        bdata = data[global_mask].T
        bdatac = bdata - np.tile(bdata.mean(0), (bdata.shape[0], 1))
        U, S, Vh = np.linalg.svd(bdatac, full_matrices=False)
        regressor_map['pc1'] = U[:,0]

    if(selector['motion']):
        regressor_map['motion'] = motion

    if(selector['linear']):
        regressor_map['linear'] = np.arange(0, data.shape[3])

    if(selector['quadratic']):
        regressor_map['quadratic'] = np.arange(0, data.shape[3])**2

    print 'Regressors include: ', regressor_map.keys()

    X = np.zeros((data.shape[3], 1))
    csv_filename = ''
    for rname, rval in regressor_map.items():
        X = np.hstack((X, rval.reshape(rval.shape[0],-1)))
        csv_filename += '_' + rname
    X = X[:,1:]

    csv_filename = csv_filename[1:]
    csv_filename += '.csv'
    csv_filename = os.path.join(os.getcwd(), csv_filename)
    np.savetxt(csv_filename, X, delimiter='\t')

    print 'Regressors dim: ', X.shape, ' starting regression'

    Y = data[global_mask].T
    B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    Y_res = Y - X.dot(B)

    data[global_mask] = Y_res.T

    print 'Writing residual and regressors'
    img = nb.Nifti1Image(data, header=nii.get_header(), affine=nii.get_affine())
    residual_file = os.path.join(os.getcwd(), 'residual.nii.gz')
    img.to_filename(residual_file)

    #Easier to read for debugging purposes
    regressors_file = os.path.join(os.getcwd(), 'nuisance_regressors.mat')

    if scipy.__version__ == '0.7.0':
        scipy.io.savemat(regressors_file, regressor_map)                        ### for scipy v0.7.0
    else:
        scipy.io.savemat(regressors_file, regressor_map, oned_as='column')   ### for scipy v0.12: OK



    return residual_file, csv_filename




def create_residuals_wf(name):

    '''
    inputs
        inputnode.func_file
        inputnode.wm_mask
        inputnode.gm_mask
        inputnode.csf_mask
        inputnode.motion_pars
        inputnode.selector
        inputnode.compcor_ncomponents
    outputs
        outputnode.residual
        outputnode.regressor_csv
    '''

    flow = Workflow(name=name)
    inputnode = Node(util.IdentityInterface(fields=['func_preproc',
                                                    'func_smoothed',
                                                    'wm_mask',
                                                    'gm_mask',
                                                    'csf_mask',
                                                    'motion_pars',
                                                    'compcor_ncomponents',
                                                    'subject_id']),
                name = 'inputnode')
    outputnode = Node(util.IdentityInterface(fields=['residual',
                                                     'regressors',]),
                name = 'outputnode')

    ##########################################################################################################################
                                                  # Extract Tissue Signals  #
    ##########################################################################################################################
    extract_tissue = Node(util.Function(input_names   =[ 'data_file',
                                                         'ventricles_mask_file',
                                                         'wm_seg_file',
                                                         'csf_seg_file',
                                                         'gm_seg_file'],
                                         output_names = ['file_wm',
                                                         'file_csf',
                                                         'file_gm'],
                                         function     =extract_tissue_data),
                        name = 'extract_func_tissues')
    extract_tissue.inputs.ventricles_mask_file = '/scr/sambesi1/workspace/Projects/GluREST/denoise/HarvardOxford-lateral-ventricles-thr25-2mm.nii.gz'



    flow.connect(inputnode,          'wm_mask',                 extract_tissue,   'wm_seg_file'  )
    flow.connect(inputnode,          'gm_mask',                 extract_tissue,   'gm_seg_file'  )
    flow.connect(inputnode,          'csf_mask',                extract_tissue,   'csf_seg_file')
    flow.connect(inputnode,          'func_preproc',            extract_tissue,   'data_file'    )


    ##########################################################################################################################
                                                   # Detrend  + Motion  + Compcor #
    ##########################################################################################################################

    dt_mc_cc  = Node(util.Function(input_names   =  ['subject','selector','wm_sig_file','csf_sig_file','gm_sig_file','motion_file',
                                                     'compcor_ncomponents'],
                                   output_names  = ['residual_file', 'regressors_file'],
                                   function      = calc_residuals),
                                   name          = 'residuals_dt_mc_cc')
    dt_mc_cc.inputs.selector = {'compcor' : True , 'wm'     : False , 'csf'    : False , 'gm'        : False, 'global' : False,
                                'pc1'     : False , 'motion' : True , 'linear' : True , 'quadratic' : True}

    flow.connect(extract_tissue,     'file_wm',                 dt_mc_cc,    'wm_sig_file'  )
    flow.connect(extract_tissue,     'file_csf',                dt_mc_cc,    'csf_sig_file' )
    flow.connect(extract_tissue,     'file_gm',                 dt_mc_cc,    'gm_sig_file'  )
    flow.connect(inputnode,          'func_smoothed',           dt_mc_cc,    'subject'      )
    flow.connect(inputnode,          'motion_pars',             dt_mc_cc,    'motion_file'  )
    flow.connect(inputnode,          'compcor_ncomponents',     dt_mc_cc,    'compcor_ncomponents' )
    flow.connect(dt_mc_cc,           'residual_file',           outputnode,  'residual'    )
    flow.connect(dt_mc_cc,           'regressors_file',         outputnode,  'regressors'    )


    return flow


def smooth_data():
    from nipype.pipeline.engine import Node, Workflow
    import nipype.interfaces.utility as util
    import nipype.interfaces.fsl as fsl

    flow        = Workflow('func2mni_preprocessed_fwhm')

    inputnode   = Node(util.IdentityInterface(fields=['func_data']),
                       name = 'inputnode')

    outputnode  =  Node(util.IdentityInterface(fields=['func_smoothed']),
                       name = 'outputnode')

    smooth      = Node(interface=fsl.Smooth(), name='func_smooth_fwhm_4')
    smooth.inputs.fwhm                 = 6.0
    smooth.terminal_output             = 'file'

    flow.connect(inputnode, 'func_data'      , smooth      , 'in_file'    )
    flow.connect(smooth,    'smoothed_file'  , outputnode  , 'func_smoothed'   )


    return flow