__author__ = 'kanaan' 'Dec 18 2014'

# forked from CPAC-0.3.8
import commands
import numpy as np
import matplotlib
import pkg_resources as p
matplotlib.use('Agg')


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
