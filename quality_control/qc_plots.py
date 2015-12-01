__author__ = 'kanaan' 'Dec 19 2014'

import seaborn as sns
from numpy import loadtxt
import os
from matplotlib.gridspec import GridSpec
from mriqc.misc import plot_vline
from matplotlib import pyplot as plt
import os
import matplotlib.cm as cm
import numpy as np
from matplotlib.ticker import FixedLocator
import math
from qc_utils import calculate_DVARS, gen_realignment_params, timeseries
from mriqc.volumes import _get_values_inside_a_mask
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
from matplotlib.figure import Figure

def plot_FD(fd1d, mean_FD_distribution, subject,figsize = (8.3,8.3)):

    threshold = 0.2
    FD_power      = np.genfromtxt(fd1d)
    meanFD        = np.round(np.mean(FD_power), decimals = 2)
    rmsFD         = np.sqrt(np.mean(FD_power))
    count         = np.int(FD_power[FD_power>threshold].size)
    percentFD     = (count*100/(len(FD_power)+1))

    fig = plt.figure(figsize=figsize)

    fig.subplots_adjust(wspace=0.3)
    fig.set_size_inches(12, 8)

    grid = GridSpec(2, 4)

    ax = plt.subplot(grid[0,:-1])
    ax.plot(FD_power)
    ylim = ax.get_ylim()
    ax.set_xlim((0, len(FD_power)))
    ax.set_ylabel("%s Frame Displacement [mm]" %subject)
    ax.set_xlabel("Frame #")
    ax.text(20, (ylim[1] - ylim[1]*0.05), 'FD mean = %s'%meanFD, va='center', size = 18, color = 'r')
    ax.text(20, (ylim[1] - ylim[1]*0.2), '%s%% (%s) > threshold  '%(percentFD, count), va='center', size = 18, color = 'r')
    #ax.text(20, (ylim[1] - ylim[1]*0.15), 'are above the threshold '%, va='center', size = 18, color = 'r')
    plt.axhline(threshold, linestyle='dashed', linewidth=2)#,color='r')

    ax = plt.subplot(grid[0,-1])
    sns.distplot(FD_power, vertical = True, ax = ax)
    ax.set_ylim(ylim)

    ax= plt.subplot(grid[1,:])
    sns.distplot(mean_FD_distribution, ax=ax)
    ax.set_xlabel("%s Mean Frame Dispalcement (over all subjects) [mm]"%subject)
    label = "MeanFD = %g"%meanFD
    plot_vline(meanFD, label, ax=ax)

    png_name = str(os.path.join(os.getcwd(), 'qc_fd_plot.png'))
    plt.savefig(png_name, dpi=190, bbox_inches='tight')

    return fig


def plot_nuisance_residuals(mov_params,
                            fd1d,
                            func_preprocessed,
                            func_preprocessed_mask,
                            func_gm,
                            residuals_dt,
                            residuals,
                            residuals_bp,
                            figsize = (8.3,8.3)):

    sns.set_style('darkgrid')

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0.05)
    fig.set_size_inches(12, 8)

    # open figure
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.05)
    fig.set_size_inches(12, 8)

    # 1 plot translations
    movement = gen_realignment_params(mov_params)
    ax1 = plt.subplot2grid((6,2), (0,0),  colspan = 1, rowspan =1)
    ax1.yaxis.set_tick_params(labelsize=5)
    ax1.plot(movement[0])
    ax1.plot(movement[1])
    ax1.plot(movement[2])
    ax1.set_xticklabels([])
    ax1.grid(True)
    ax1.set_color_cycle(['red', 'green', 'blue'])
    ax1.set_ylabel('trans')

    # 2 plot rotations
    ax2 = plt.subplot2grid((6,2), (0,1),  colspan = 1, rowspan =1)
    ax2.set_color_cycle(['red', 'green', 'blue'])
    ax2.plot(movement[3])
    ax2.plot(movement[4])
    ax2.plot(movement[5])
    ax2.yaxis.set_tick_params(labelsize=5)
    ax2.set_xticklabels([])
    ax2.grid(True)
    ax2.set_ylabel('rot')

    # 3 plot FD
    FD_power = np.loadtxt(fd1d)
    ax3 = plt.subplot2grid((6,2), (1, 0),  colspan = 2, rowspan =1)
    #ax3.axes.get_yaxis().set_visible(True)
    ax3.yaxis.set_tick_params(labelsize=5)
    ax3.set_xticklabels([])
    ax3.grid(True)
    ax3.plot(FD_power)
    ax3.set_xlim([0, 422])
    plt.axhline(0.2, linestyle='dashed', linewidth=2, color='r')
    ax3.set_ylabel('FD')
    #ax3.yaxis.set_major_locator(FixedLocator((ax3.get_ylim())))

    # 4 plot DVARS
    DVARS = calculate_DVARS(func_preprocessed, func_preprocessed_mask)
    # DVARS = calc_dvars(func_preprocessed, output_all=False, interp="fraction")
    ax4 = plt.subplot2grid((6,2), (2, 0),  colspan = 2, rowspan =1)
    ax4.plot(DVARS, color='red')
    ax4.set_xlim([0, 422])
    ax4.set_ylabel('DVARS')
    ax4.set_xticklabels([])
    ax4.yaxis.set_tick_params(labelsize=5)
    ax4.grid(True)

    #plot aroma_detrend
    n1  = timeseries(residuals_dt, func_gm)
    ax5 = plt.subplot2grid((6,2), (3,0),  colspan = 2, rowspan =1)
    ax5.imshow(n1, interpolation = 'none', aspect = 'auto', cmap=cm.gray, vmin =-50, vmax = 50)
    ax5.set_title('Detrend + Compocor + Friston 24', fontsize = 8 )
    ax5.axes.get_xaxis().set_visible(False)
    ax5.axes.get_yaxis().set_visible(False)

    #plot aroma_detrend
    n2  = timeseries(residuals, func_gm)
    ax6 = plt.subplot2grid((6,2), (4,0),  colspan = 2, rowspan =1)
    ax6.imshow(n2, interpolation = 'none', aspect = 'auto', cmap=cm.gray, vmin =-50, vmax = 50)
    ax6.set_title('AROMA + Detrend + Compocor + Friston 24', fontsize = 8 )
    ax6.axes.get_xaxis().set_visible(False)
    ax6.axes.get_yaxis().set_visible(False)

    # #plot residuals
    n3  = timeseries(residuals_bp, func_gm)
    ax7 = plt.subplot2grid((6,2), (5,0),  colspan = 2, rowspan =1)
    ax7.imshow(n3, interpolation = 'none', aspect = 'auto', cmap=cm.gray, vmin =-50, vmax = 50)
    ax7.set_title('AROMA + Detrend + Compocor + Friston 24 + BP', fontsize = 8)
    ax7.axes.get_xaxis().set_visible(False)
    ax7.axes.get_yaxis().set_visible(False)

    png_name = str(os.path.join(os.getcwd(), 'qc_nuisance_plot.png'))
    plt.savefig(png_name, dpi=190, bbox_inches='tight')
    plt.close()

    return fig



def plot_distrbution_of_values(main_file, mask_file, xlabel, distribution=None, xlabel2=None, figsize=(11.7,8.3)):

    data = _get_values_inside_a_mask(main_file, mask_file)

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0.3)
    fig.set_size_inches(12, 8)

    gs = GridSpec(2, 1)
    ax = plt.subplot(gs[0, 0])
    sns.distplot(data.astype(np.double), kde=False, bins=100, ax=ax)
    ax.set_xlabel(xlabel)

    ax = plt.subplot(gs[1, 0])
    sns.distplot(np.array(distribution).astype(np.double), ax=ax)
    cur_val = np.median(data)
    label = "%g"%cur_val
    plot_vline(cur_val, label, ax=ax)
    ax.set_xlabel(xlabel2)

    png_name = str(os.path.join(os.getcwd(), 'qc_tsnr_plot.png'))
    plt.savefig(png_name, dpi=190, bbox_inches='tight')

    return fig



def plot_3d_overlay(underlay_file, overlay_file, out_filename):
    import nibabel as nb
    from qc_utils import find_cut_coords
    import matplotlib


    underlay = nb.load(underlay_file).get_data()
    overlay = nb.load(overlay_file).get_data()


    coords = find_cut_coords(nb.load(overlay_file))

    # convert zeros to nans for visualization purposes
    overlay[overlay==0]=np.nan

    # plot voxel on anat
    fig =plt.figure()
    fig.set_size_inches(6.5, 6.5)
    fig.subplots_adjust(wspace=0.005)
    import mpl_toolkits.axisartist.floating_axes as floating_axes
    from matplotlib.transforms import Affine2D

    #1
    ax1 = plt.subplot2grid((1,3), (0,0),  colspan = 1, rowspan =1)
    ax1.imshow(underlay[coords[0],:,:])
    ax1.imshow(overlay[coords[0],:,:] , matplotlib.cm.rainbow_r, alpha = 0.5)
    #ax1.set_xlim(23, 157)
    #ax1.set_ylim(101, 230)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_yticklabels(ax1.yaxis.get_majorticklabels(), rotation=180.)

    #2
    ax2 = plt.subplot2grid((1,3), (0,1),  colspan = 1, rowspan =1)
    ax2.imshow(np.rot90(underlay[:,:,coords[2]]))
    ax2.imshow(np.rot90(overlay[:,:,coords[2]]) , matplotlib.cm.rainbow_r, alpha = 0.5 )
    #ax2.set_xlim(230, 20)
    #ax2.set_ylim(207, 4)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    #3
    ax3 = plt.subplot2grid((1,3), (0,2),  colspan = 1, rowspan =1)
    ax3.imshow(underlay[:,coords[1],:])
    ax3.imshow(overlay[:,coords[1],:] , matplotlib.cm.rainbow_r, alpha = 0.5, origin='lower')
    #ax3.set_xlim(38, 140)
    #ax3.set_ylim(160, 60)
    ax3.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    ax1.set_yticklabels(ax1.yaxis.get_majorticklabels(), rotation=90.)
    fig.tight_layout()
    fig.savefig(out_filename, dpi=300, bbox_inches='tight')

    return fig

