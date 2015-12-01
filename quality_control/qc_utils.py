__author__ = 'kanaan' 'Dec 18 2014'

# forked from CPAC-0.3.8
import commands
import numpy as np
import matplotlib
import math
import pkg_resources as p
matplotlib.use('Agg')

def calculate_DVARS(rest, mask):

    import numpy as np
    import nibabel as nib
    rest_data = nib.load(rest).get_data().astype(np.float32)
    mask_data = nib.load(mask).get_data().astype('bool')
    #square of relative intensity value for each voxel across
    #every timepoint
    data = np.square(np.diff(rest_data, axis = 3))
    #applying mask, getting the data in the brain only
    data = data[mask_data]
    #square root and mean across all timepoints inside mask
    DVARS = np.sqrt(np.mean(data, axis=0))
    return DVARS

def gen_realignment_params(realignment_parameters_file):
    data = np.loadtxt(realignment_parameters_file)
    data_t = data.T
    x = data_t[0]
    y = data_t[1]
    z = data_t[2]
    for i in range(3, 6):
        for j in range(len(data_t[i])):
            data_t[i][j] = math.degrees(data_t[i][j])
    roll = data_t[3]
    pitch= data_t[4]
    yaw = data_t[5]
    return x,y,z,roll, pitch, yaw

def timeseries(rest, grey):
    import numpy as np
    import nibabel as nib
    import os
    rest_data = nib.load(rest).get_data().astype(np.float32)
    gm_mask = nib.load(grey).get_data().astype('bool')
    rest_gm = rest_data[gm_mask]

    return rest_gm



def find_cut_coords(img, mask=None, activation_threshold=None):
    import warnings
    import numpy as np
    from scipy import ndimage
    from nilearn._utils import as_ndarray, new_img_like
    from nilearn._utils.ndimage import largest_connected_component
    from nilearn._utils.extmath import fast_abs_percentile
    """ Find the center of the largest activation connected component.
        Parameters
        -----------
        img : 3D Nifti1Image
            The brain map.
        mask : 3D ndarray, boolean, optional
            An optional brain mask.
        activation_threshold : float, optional
            The lower threshold to the positive activation. If None, the
            activation threshold is computed using the 80% percentile of
            the absolute value of the map.
        Returns
        -------
        x : float
            the x world coordinate.
        y : float
            the y world coordinate.
        z : float
            the z world coordinate.
    """
    data = img.get_data()
    # To speed up computations, we work with partial views of the array,
    # and keep track of the offset
    offset = np.zeros(3)

    # Deal with masked arrays:
    if hasattr(data, 'mask'):
        not_mask = np.logical_not(data.mask)
        if mask is None:
            mask = not_mask
        else:
            mask *= not_mask
        data = np.asarray(data)

    # Get rid of potential memmapping
    data = as_ndarray(data)
    my_map = data.copy()
    if mask is not None:
        slice_x, slice_y, slice_z = ndimage.find_objects(mask)[0]
        my_map = my_map[slice_x, slice_y, slice_z]
        mask = mask[slice_x, slice_y, slice_z]
        my_map *= mask
        offset += [slice_x.start, slice_y.start, slice_z.start]

    # Testing min and max is faster than np.all(my_map == 0)
    if (my_map.max() == 0) and (my_map.min() == 0):
        return .5 * np.array(data.shape)
    if activation_threshold is None:
        activation_threshold = fast_abs_percentile(my_map[my_map != 0].ravel(),
                                                   80)
    mask = np.abs(my_map) > activation_threshold - 1.e-15
    # mask may be zero everywhere in rare cases
    if mask.max() == 0:
        return .5 * np.array(data.shape)
    mask = largest_connected_component(mask)
    slice_x, slice_y, slice_z = ndimage.find_objects(mask)[0]
    my_map = my_map[slice_x, slice_y, slice_z]
    mask = mask[slice_x, slice_y, slice_z]
    my_map *= mask
    offset += [slice_x.start, slice_y.start, slice_z.start]

    # For the second threshold, we use a mean, as it is much faster,
    # althought it is less robust
    second_threshold = np.abs(np.mean(my_map[mask]))
    second_mask = (np.abs(my_map) > second_threshold)
    if second_mask.sum() > 50:
        my_map *= largest_connected_component(second_mask)
    cut_coords = ndimage.center_of_mass(np.abs(my_map))
    x_map, y_map, z_map = cut_coords + offset

    coords = []
    coords.append(x_map)
    coords.append(y_map)
    coords.append(z_map)

    # Return as a list of scalars
    return coords


def make_edge(file_):

    """
    CPAC 0.3.8 implementation
    Make edge file from a scan image

    Parameters
    ----------
    file_ :    string
        path to the scan

    Returns
    -------
    new_fname : string
        path to edge file
    """
    import commands
    import os
    remainder, ext_ = os.path.splitext(file_)
    remainder, ext1_ = os.path.splitext(remainder)
    ext = ''.join([ext1_, ext_])
    new_fname = ''.join([remainder, '_edge', ext])
    new_fname = os.path.join(os.getcwd(), os.path.basename(new_fname))
    cmd = "3dedge3 -input %s -prefix %s -fscale" % (file_, new_fname)
    print cmd
    print commands.getoutput(cmd)
    return new_fname


def determine_start_and_end(data, direction, percent):

    """
    Determine start slice and end slice in data file in
    given direction with at least threshold percent of voxels
    at start and end slices.

    Parameters
    ----------

    data : string
        input nifti file

    direction : string
        axial or sagittal

    percent : float
        percent(from total) of non zero voxels at starting and ending slice


    Returns
    -------

    start : integer
            Index of starting slice

    end : integer
            Index of the last slice

    """

    x, y, z = data.shape

    xx1 = 0
    xx2 = x - 1
    zz1 = 0
    zz2 = z - 1
    total_non_zero_voxels = len(np.nonzero(data.flatten())[0])
    thresh = percent * float(total_non_zero_voxels)
    start = None
    end = None

    if 'axial' in direction:
        while(zz2 > 0):
            d = len(np.nonzero(data[:, :, zz2].flatten())[0])
            if float(d) > thresh:
                break
            zz2 -= 1
        while(zz1 < zz2):
            d = len(np.nonzero(data[:, :, zz1].flatten())[0])
            if float(d) > thresh:
                break
            zz1 += 1
        start =  zz1
        end = zz2
    else:
        while(xx2 > 0):
            d = len(np.nonzero(data[xx2, :, :].flatten())[0])
            if float(d) > thresh:
                break
            xx2 -= 1
        while(xx1 < xx2):
            d = len(np.nonzero(data[xx1, :, :].flatten())[0])
            if float(d) > thresh:
                break
            xx1 += 1
        start = xx1
        end = xx2
    return start, end

def get_spacing(across, down, dimension):

    """
    Get Spacing in slices to be selected for montage
    display varying in given dimension

    Parameters
    ----------
    across : integer
        # images placed horizontally in montage

    down : integer
        # images stacked vertically in montage

    Returns
    -------

    space : integer
        # of images to skip before displaying next one

    """
#    across = 6
#    down = 3
    space = 10
    prod = (across*down*space)
    if prod > dimension:
        while(across*down*space) > dimension:
            space -= 1
    else:
        while(across*down*space) < dimension:
            space += 1
    return space


def drange(min_, max_):

    """
    Generate list of float values in a specified range.

    Parameters
    ----------

    min_ : float
        Min value

    max_ : float
        Max value


    Returns
    -------

    range_ : list
        list of float values in the min_ max_ range

    """

    step = float(max_ - min_) /8.0
    range_ = []

    while min_ <= max_:

        range_.append(float('%.3f' % round(min_, 3)))
        min_ += step
    return range_
