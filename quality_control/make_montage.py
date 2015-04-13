__author__ = 'kanaan' 'Dec 18 2014'

# Make montage functions are borrowed from CPAC 0.3.8 with various manipulations

import os
import sys
import commands
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio
import nipype.interfaces.utility as util
#from CPAC.qc.qc import *
#from CPAC.qc.utils import determine_start_and_end, drange

from qc_utils import drange

def make_montage_axial(overlay, underlay, png_name, cbar_name):

    """

    Draws Montage using overlay on Anatomical brain in Axial Direction
    Parameters
    ----------
    overlay : string
            Nifi file
    underlay : string
            Nifti for Anatomical Brain
    cbar_name : string
            name of the cbar
    png_name : string
            Proposed name of the montage plot
    Returns
    -------
    png_name : Path to generated PNG

    """
    import matplotlib
    import commands
#    matplotlib.use('Agg')
    import os
    matplotlib.rcParams.update({'font.size': 5})
    import matplotlib.cm as cm
    ###
    try:
        from mpl_toolkits.axes_grid1 import ImageGrid
    except:
        from mpl_toolkits.axes_grid import ImageGrid
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    import nibabel as nb
    import numpy as np

    def determine_start_and_end(data, direction, percent):
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
        space = 10
        prod = (across*down*space)
        if prod > dimension:
            while(across*down*space) > dimension:
                space -= 1
        else:
            while(across*down*space) < dimension:
                space += 1
        return space

    Y = nb.load(underlay).get_data()
    X = nb.load(overlay).get_data()
    X = X.astype(np.float16)
    Y = Y.astype(np.float16)

    if  'skull_vis' in png_name:
        X[X < 20.0] = 0.0
    if 'skull_vis' in png_name or 't1_edge_on_mean_func_in_t1' in png_name or 'MNI_edge_on_mean_func_mni' in png_name:
        max_ = np.nanmax(np.abs(X.flatten()))
        X[X != 0.0] = max_
        print '^^', np.unique(X)
    z1, z2 = determine_start_and_end(Y, 'axial', 0.0001)
    spacing = get_spacing(6, 3, z2 - z1)
    x, y, z = Y.shape
    fig = plt.figure(1)
    max_ = np.max(np.abs(Y))

    if ('snr' in png_name) or  ('reho' in png_name) or ('vmhc' in png_name) or ('sca_' in png_name) or ('alff' in png_name) or ('centrality' in png_name) or ('temporal_regression_sca' in png_name)  or ('temporal_dual_regression' in png_name):
        grid = ImageGrid(fig, 111, nrows_ncols=(3, 6), share_all=True, aspect=True, cbar_mode="single", cbar_pad=0.2, direction="row")
    else:
        grid = ImageGrid(fig, 111, nrows_ncols=(3, 6), share_all=True, aspect=True, direction="row")

    zz = z1
    for i in range(6*3):

        if zz >= z2:
            break

        im = grid[i].imshow(np.rot90(Y[:, :, zz]), cmap=cm.Greys_r)
        zz += spacing

    x, y, z = X.shape
    X[X == 0.0] = np.nan
    max_ = np.nanmax(np.abs(X.flatten()))
    print '~~', max_


    zz = z1
    im = None
    print '~~~', z1, ' ', z2
    for i in range(6*3):


        if zz >= z2:
            break
        if cbar_name is 'red_to_blue':

            im = grid[i].imshow(np.rot90(X[:, :, zz]), cmap=cm.get_cmap(cbar_name), alpha=0.82, vmin=0, vmax=max_)   ###
        elif cbar_name is 'green':
            im = grid[i].imshow(np.rot90(X[:, :, zz]), cmap=cm.get_cmap(cbar_name), alpha=0.82, vmin=0, vmax=max_)
        else:
            im = grid[i].imshow(np.rot90(X[:, :, zz]), cmap=cm.get_cmap(cbar_name), alpha=0.82, vmin=- max_, vmax=max_)

        grid[i].axes.get_xaxis().set_visible(False)
        grid[i].axes.get_yaxis().set_visible(False)
        zz += spacing

    cbar = grid.cbar_axes[0].colorbar(im)

    if 'snr' in png_name:
        cbar.ax.set_yticks(drange(0, max_))

    elif  ('reho' in png_name) or ('vmhc' in png_name) or ('sca_' in png_name) or ('alff' in png_name) or ('centrality' in png_name) or ('temporal_regression_sca' in png_name) or ('temporal_dual_regression' in png_name):
        cbar.ax.set_yticks(drange(-max_, max_))


    plt.show()
    plt.axis("off")
    png_name = str(os.path.join(os.getcwd(), png_name))
    plt.savefig(png_name, dpi=300, bbox_inches='tight')
    plt.close()

    matplotlib.rcdefaults()

    return png_name

def make_montage_sagittal(overlay, underlay, png_name, cbar_name):

    """
    Draws Montage using overlay on Anatomical brain in Sagittal Direction
    Parameters
    ----------
    overlay : string
            Nifi file
    underlay : string
            Nifti for Anatomical Brain
    cbar_name : string
            name of the cbar
    png_name : string
            Proposed name of the montage plot
    Returns
    -------
    png_name : Path to generated PNG
    """
    import matplotlib
    import commands
    import os
    import numpy as np
    matplotlib.rcParams.update({'font.size': 5})
    ###
    try:
        from mpl_toolkits.axes_grid1 import ImageGrid
    except:
        from mpl_toolkits.axes_grid import ImageGrid
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    import nibabel as nb
    def determine_start_and_end(data, direction, percent):
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
        space = 10
        prod = (across*down*space)
        if prod > dimension:
            while(across*down*space) > dimension:
                space -= 1
        else:
            while(across*down*space) < dimension:
                space += 1
        return space

    Y = nb.load(underlay).get_data()
    X = nb.load(overlay).get_data()
    X = X.astype(np.float16)
    Y = Y.astype(np.float16)


    if  'skull_vis' in png_name:
        X[X < 20.0] = 0.0
    if 'skull_vis' in png_name or 't1_edge_on_mean_func_in_t1' in png_name or 'MNI_edge_on_mean_func_mni' in png_name:
        max_ = np.nanmax(np.abs(X.flatten()))
        X[X != 0.0] = max_
        print '^^', np.unique(X)

    x1, x2 = determine_start_and_end(Y, 'sagittal', 0.0001)
    spacing = get_spacing(6, 3, x2 - x1)
    x, y, z = Y.shape
    fig = plt.figure(1)
    max_ = np.max(np.abs(Y))

    if ('snr' in png_name) or  ('reho' in png_name) or ('vmhc' in png_name) or ('sca_' in png_name) or ('alff' in png_name) or ('centrality' in png_name) or ('temporal_regression_sca' in png_name)  or ('temporal_dual_regression' in png_name):
        grid = ImageGrid(fig, 111, nrows_ncols=(3, 6), share_all=True, aspect=True, cbar_mode="single", cbar_pad=0.5, direction="row")
    else:
        grid = ImageGrid(fig, 111, nrows_ncols=(3, 6), share_all=True, aspect=True, cbar_mode="None", direction="row")

    xx = x1
    for i in range(6*3):

        if xx >= x2:
            break

        im = grid[i].imshow(np.rot90(Y[xx, :, :]), cmap=cm.Greys_r)
        grid[i].get_xaxis().set_visible(False)
        grid[i].get_yaxis().set_visible(False)
        xx += spacing

    x, y, z = X.shape
    X[X == 0.0] = np.nan
    max_ = np.nanmax(np.abs(X.flatten()))
    print '~~', max_
    xx = x1
    for i in range(6*3):


        if xx >= x2:
            break
        im = None
        if cbar_name is 'red_to_blue':

            im = grid[i].imshow(np.rot90(X[xx, :, :]), cmap=cm.get_cmap(cbar_name), alpha=0.82, vmin=0, vmax=max_)   ###
        elif cbar_name is 'green':
            im = grid[i].imshow(np.rot90(X[xx, :, :]), cmap=cm.get_cmap(cbar_name), alpha=0.82, vmin=0, vmax=max_)
        else:
            im = grid[i].imshow(np.rot90(X[xx, :, :]), cmap=cm.get_cmap(cbar_name), alpha=0.82, vmin=- max_, vmax=max_)
        xx += spacing
    cbar = grid.cbar_axes[0].colorbar(im)

    if 'snr' in png_name:
        cbar.ax.set_yticks(drange(0, max_))

    elif  ('reho' in png_name) or ('vmhc' in png_name) or ('sca_' in png_name) or ('alff' in png_name) or ('centrality' in png_name) or ('temporal_regression_sca' in png_name)  or ('temporal_dual_regression' in png_name):
        cbar.ax.set_yticks(drange(-max_, max_))


    plt.show()
    plt.axis("off")
    png_name = os.path.join(os.getcwd(), png_name)
    plt.savefig(png_name, dpi=300, bbox_inches='tight')
    plt.close()
    matplotlib.rcdefaults()

    return png_name




def montage_tissues_axial(overlay_csf, overlay_wm, overlay_gm, underlay, png_name):

    """
    Draws Montage using GM WM and CSF overlays on Anatomical brain in Sagittal Direction
    Parameters
    ----------
    overlay_csf : string
            Nifti file CSF MAP
    overlay_wm : string
            Nifti file WM MAP
    overlay_gm : string
            Nifti file GM MAP
    underlay : string
            Nifti for Anatomical Brain
    png_name : string
            Proposed name of the montage plot
    Returns
    -------
    png_name : Path to generated PNG
    """
    def determine_start_and_end(data, direction, percent):
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
        space = 10
        prod = (across*down*space)
        if prod > dimension:
            while(across*down*space) > dimension:
                space -= 1
        else:
            while(across*down*space) < dimension:
                space += 1
        return space

    import os
    import matplotlib
    import commands
#    matplotlib.use('Agg')
    import numpy as np
    ###
    try:
        from mpl_toolkits.axes_grid1 import ImageGrid
    except:
        from mpl_toolkits.axes_grid import ImageGrid
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    import nibabel as nb
    import matplotlib.cm as cm

    Y = nb.load(underlay).get_data()
    z1, z2 = determine_start_and_end(Y, 'axial', 0.0001)
    spacing = get_spacing(6, 3, z2 - z1)
    X_csf = nb.load(overlay_csf).get_data()
    X_wm = nb.load(overlay_wm).get_data()
    X_gm = nb.load(overlay_gm).get_data()
    X_csf = X_csf.astype(np.float16)
    X_wm = X_wm.astype(np.float16)
    X_gm = X_gm.astype(np.float16)
    Y = Y.astype(np.float16)

    max_csf = np.nanmax(np.abs(X_csf.flatten()))
    X_csf[X_csf != 0.0] = max_csf
    max_wm = np.nanmax(np.abs(X_wm.flatten()))
    X_wm[X_wm != 0.0] = max_wm
    max_gm = np.nanmax(np.abs(X_gm.flatten()))
    X_gm[X_gm != 0.0] = max_gm
    x, y, z = Y.shape
    fig = plt.figure(1)
    max_ = np.max(np.abs(Y))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 6), share_all=True, aspect=True, cbar_mode="None", direction="row")

    zz = z1
    for i in range(6*3):

        if zz >= z2:
            break

        im = grid[i].imshow(np.rot90(Y[:, :, zz]), cmap=cm.Greys_r)
        zz += spacing

    x, y, z = X_csf.shape
    X_csf[X_csf == 0.0] = np.nan
    X_wm[X_wm == 0.0] = np.nan
    X_gm[X_gm == 0.0] = np.nan
    print '~~', max_


    zz = z1
    im = None
    for i in range(6*3):


        if zz >= z2:
            break

        im = grid[i].imshow(np.rot90(X_csf[:, :, zz]), cmap=cm.winter, alpha=0.7, vmin=0, vmax=max_csf)   ###
        im = grid[i].imshow(np.rot90(X_wm[:, :, zz]), cmap=cm.GnBu  , alpha=0.7, vmin=0, vmax=max_wm)
        im = grid[i].imshow(np.rot90(X_gm[:, :, zz]), cmap=cm.bwr, alpha=0.7, vmin=0, vmax=max_gm)

        grid[i].axes.get_xaxis().set_visible(False)
        grid[i].axes.get_yaxis().set_visible(False)
        zz += spacing

    cbar = grid.cbar_axes[0].colorbar(im)

    #plt.show()
    plt.axis("off")
    png_name = str(os.path.join(os.getcwd(), png_name))
    plt.savefig(png_name, dpi=300, bbox_inches='tight', )
    plt.close()

    return png_name

def montage_tissues_sagittal(overlay_csf, overlay_wm, overlay_gm, underlay, png_name):

    """
    Draws Montage using GM WM and CSF overlays on Anatomical brain in Sagittal Direction
    Parameters
    ----------
    overlay_csf : string
            Nifi file CSF MAP
    overlay_wm : string
            Nifti file WM MAP
    overlay_gm : string
            Nifti file GM MAP
    underlay : string
            Nifti for Anatomical Brain
    png_name : string
            Proposed name of the montage plot
    Returns
    -------
    png_name : Path to generated PNG
    """
    def determine_start_and_end(data, direction, percent):
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
        space = 10
        prod = (across*down*space)
        if prod > dimension:
            while(across*down*space) > dimension:
                space -= 1
        else:
            while(across*down*space) < dimension:
                space += 1
        return space

    import os
    import matplotlib
    import commands
#    matplotlib.use('Agg')
    import numpy as np
    ###
    try:
        from mpl_toolkits.axes_grid1 import ImageGrid
    except:
        from mpl_toolkits.axes_grid import ImageGrid
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    import matplotlib.cm as cm
    import nibabel as nb

    Y = nb.load(underlay).get_data()
    x1, x2 = determine_start_and_end(Y, 'sagittal', 0.0001)
    spacing = get_spacing(6, 3, x2 - x1)
    X_csf = nb.load(overlay_csf).get_data()
    X_wm = nb.load(overlay_wm).get_data()
    X_gm = nb.load(overlay_gm).get_data()
    X_csf = X_csf.astype(np.float16)
    X_wm = X_wm.astype(np.float16)
    X_gm = X_gm.astype(np.float16)
    Y = Y.astype(np.float16)

    max_csf = np.nanmax(np.abs(X_csf.flatten()))
    X_csf[X_csf != 0.0] = max_csf
    max_wm = np.nanmax(np.abs(X_wm.flatten()))
    X_wm[X_wm != 0.0] = max_wm
    max_gm = np.nanmax(np.abs(X_gm.flatten()))
    X_gm[X_gm != 0.0] = max_gm
    x, y, z = Y.shape
    fig = plt.figure(1)
    max_ = np.max(np.abs(Y))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 6), share_all=True, aspect=True, cbar_mode="None", direction="row")

    zz = x1
    for i in range(6*3):

        if zz >= x2:
            break

        im = grid[i].imshow(np.rot90(Y[zz, :, :]), cmap=cm.Greys_r)
        zz += spacing

    x, y, z = X_csf.shape
    X_csf[X_csf == 0.0] = np.nan
    X_wm[X_wm == 0.0] = np.nan
    X_gm[X_gm == 0.0] = np.nan
    print '~~', max_


    zz = x1
    im = None
    for i in range(6*3):


        if zz >= x2:
            break

        im = grid[i].imshow(np.rot90(X_csf[zz, :, :]), cmap=cm.winter, alpha=0.7, vmin=0, vmax=max_csf)   ###
        im = grid[i].imshow(np.rot90(X_wm[zz, :, :]),  cmap=cm.GnBu, alpha=0.7, vmin=0, vmax=max_wm)
        im = grid[i].imshow(np.rot90(X_gm[zz, :, :]),  cmap=cm.bwr, alpha=0.7, vmin=0, vmax=max_gm)

        grid[i].axes.get_xaxis().set_visible(False)
        grid[i].axes.get_yaxis().set_visible(False)
        zz += spacing

    cbar = grid.cbar_axes[0].colorbar(im)

    #plt.show()
    plt.axis("off")
    png_name = str(os.path.join(os.getcwd(), png_name))
    plt.savefig(png_name, dpi=300, bbox_inches='tight')
    plt.close()

    return png_name
