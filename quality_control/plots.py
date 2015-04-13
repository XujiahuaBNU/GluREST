__author__ = 'kanaan' 'Dec 19 2014'

def plot_FD(fd1d):
    import numpy as np
    import pylab as plt
    import seaborn as sns
    from numpy import loadtxt
    import os

    threshold = 0.2
    FD_power      = np.genfromtxt(fd1d)
    meanFD        = np.round(np.mean(FD_power), decimals = 2)
    rmsFD         = np.sqrt(np.mean(FD_power))
    count         = np.int(FD_power[FD_power>threshold].size)
    percentFD     = (count*100/(len(FD_power)+1))

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.3)
    fig.set_size_inches(12, 8)

    ax1 = plt.subplot2grid((9,1), (2,0), colspan= 4, rowspan = 7)
    ax1.plot(FD_power)
    ylim = ax1.get_ylim()
    ax1.set_xlim((0, len(FD_power)))
    ax1.set_ylabel("Frame Displacement [mm]")
    ax1.set_xlabel("Frame #")
    ax1.text(20, (ylim[1] - ylim[1]*0.05), 'FD mean = %s'%meanFD, va='center', size = 18, color = 'r')
    ax1.text(20, (ylim[1] - ylim[1]*0.12), '%s%% (%s) > threshold  '%(percentFD, count), va='center', size = 18, color = 'r')
    #ax1.text(20, (ylim[1] - ylim[1]*0.15), ' are above the threshold '%, va='center', size = 18, color = 'r')
    plt.axhline(threshold, linestyle='dashed', linewidth=2)#,color='r')

    ax2 = plt.subplot2grid((9,1), (0,0), colspan= 4, rowspan = 2)
    #bins = np.linspace(0, max_data, max_data + 1)
    ##### seaborn not working for some stupid reason ...... ####
    sns.distplot(FD_power,bins = 70, vertical = False )
    #ax2.set_ylim(ylim)
    #plt.hist(FD_power, 80, histtype="stepfilled", alpha=.7);

    png_name = str(os.path.join(os.getcwd(), 'FD_QC_PLOT.png'))
    plt.savefig(png_name, dpi=190, bbox_inches='tight')
    plt.close()

    return png_name

def plot_nuisance_residuals(mov_params,
                            fd1d,
                            func_preprocessed,
                            func_gm,
                            func_preprocessed_mask,
                            detrend,
                            detrend_mc,
                            detrend_mc_compcor,
                            detrend_mc_compcor_wmcsf_global,):

    from matplotlib import pyplot as plt
    import os
    import matplotlib.cm as cm
    import numpy as np
    from matplotlib.ticker import FixedLocator
    import math

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

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.05)
    fig.set_size_inches(12, 8)

    # open figure
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.05)
    fig.set_size_inches(12, 8)

    # 1 plot translations
    movement = gen_realignment_params(mov_params)
    ax1 = plt.subplot2grid((7,2), (0,0),  colspan = 1, rowspan =1)
    ax1.axes.get_yaxis().set_visible(True)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.plot(movement[0])
    ax1.plot(movement[1])
    ax1.plot(movement[2])
    ax1.set_color_cycle(['red', 'green', 'blue'])
    ax1.axes.get_yaxis().set_visible(True)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_ylabel('trans')
    ax1_ylims = ax1.get_ylim()
    ax1_ymajorlocator = FixedLocator(ax1_ylims)
    ax1.yaxis.set_major_locator(ax1_ymajorlocator)

    # 2 plot rotations
    ax2 = plt.subplot2grid((7,2), (0,1),  colspan = 1, rowspan =1)
    ax2.set_color_cycle(['red', 'green', 'blue'])
    ax2.plot(movement[3])
    ax2.plot(movement[4])
    ax2.plot(movement[5])
    ax2.axes.get_yaxis().set_visible(True)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel('rot')
    ax2_ylims = ax2.get_ylim()
    ax2_ymajorlocator = FixedLocator(ax2_ylims)
    ax2.yaxis.set_major_locator(ax2_ymajorlocator)

    # 3 plot FD
    FD_power = np.loadtxt(fd1d)
    ax3 = plt.subplot2grid((7,2), (1, 0),  colspan = 2, rowspan =1)
    ax3.axes.get_yaxis().set_visible(True)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.plot(FD_power)
    ax3.set_xlim([0, 422])
    ax3.set_ylabel('FD')
    ax3_ylims = ax3.get_ylim()
    ax3_ymajorlocator = FixedLocator(ax3_ylims)
    ax3.yaxis.set_major_locator(ax3_ymajorlocator)

    # 4 plot DVARS
    DVARS = calculate_DVARS(func_preprocessed, func_preprocessed_mask)
    ax4 = plt.subplot2grid((7,2), (2, 0),  colspan = 2, rowspan =1)
    ax4.plot(DVARS, color='red')
    ax4.axes.get_yaxis().set_visible(True)
    ax4.axes.get_xaxis().set_visible(False)
    ax4.set_xlim([0, 422])
    ax4.set_ylabel('BOLD')
    ax4_ylims = ax4.get_ylim()
    ax4_ymajorlocator = FixedLocator(ax4_ylims)
    ax4.yaxis.set_major_locator(ax4_ymajorlocator)

    #plot NUISANCE_DETREND
    n1  = timeseries(detrend, func_gm)
    ax5 = plt.subplot2grid((7,2), (3,0),  colspan = 2, rowspan =1)
    ax5.imshow(n1, interpolation = 'none', aspect = 'auto', cmap=cm.gray, vmin =-100, vmax = 100)
    ax5.set_title('DETREND', fontsize = 8 )
    ax5.axes.get_xaxis().set_visible(False)
    ax5.axes.get_yaxis().set_visible(False)

    # #plot NUISANCE DETREND + motion
    n2  = timeseries(detrend_mc, func_gm)
    ax6 = plt.subplot2grid((7,2), (4,0),  colspan = 2, rowspan =1)
    ax6.imshow(n2, interpolation = 'none', aspect = 'auto', cmap=cm.gray, vmin =-100, vmax = 100)
    ax6.set_title('DETREND + Motion', fontsize = 8)
    ax6.axes.get_xaxis().set_visible(False)
    ax6.axes.get_yaxis().set_visible(False)

    # #plot NUISANCE  DETREND + motion + COMPCOR
    n3  = timeseries(detrend_mc_compcor, func_gm)
    ax7 = plt.subplot2grid((7,2), (5,0),  colspan = 2, rowspan =1)
    ax7.imshow(n3, interpolation = 'none', aspect = 'auto', cmap=cm.gray,  vmin =-100, vmax = 100)
    ax7.set_title('DETREND + Motion + COMPCOR', fontsize = 8)
    ax7.axes.get_xaxis().set_visible(False)
    ax7.axes.get_yaxis().set_visible(False)


    #plot_NUISANCE  DETREND + motion + COMPCOR + global
    n4  = timeseries(detrend_mc_compcor_wmcsf_global, func_gm)
    ax7 = plt.subplot2grid((7,2), (6,0),  colspan = 2, rowspan =1)
    ax7.imshow(n4, interpolation = 'none', aspect = 'auto', cmap=cm.gray, vmin =-100, vmax = 100)
    ax7.set_title('DETREND + Motion + COMPCOR + GLOBAL', fontsize = 8)
    ax7.axes.get_yaxis().set_visible(False)
    ax7.axes.get_xaxis().set_visible(True)

    png_name = str(os.path.join(os.getcwd(), 'qc_nuisance.png'))
    plt.savefig(png_name, dpi=190, bbox_inches='tight')
    plt.close()

    return png_name


# def plot_FD(fd1d):
#     import numpy as np
#     import pylab as plt
#     import seaborn as sns
#     from numpy import loadtxt
#     import os
#     threshold = 0.2
#     FD_power      = loadtxt(fd1d)
#     meanFD        = np.round(np.mean(FD_power), decimals = 2)
#     rmsFD         = np.sqrt(np.mean(FD_power))
#     count         = np.int(FD_power[FD_power>threshold].size)
#     percentFD     = (count*100/(len(FD_power)+1))
#
#     fig = plt.figure()
#     fig.subplots_adjust(wspace=0.3)
#     fig.set_size_inches(12, 8)
#
#     ax1 = plt.subplot2grid((1,7), (0,0), colspan= 5)
#     ax1.plot(FD_power)
#     ylim = ax1.get_ylim()
#     ax1.set_xlim((0, len(FD_power)))
#     ax1.set_ylabel("Frame Displacement [mm]")
#     ax1.set_xlabel("Frame #")
#     ax1.text(20, (ylim[1] - 0.02), 'FD mean = %s'%meanFD, va='center', size = 18, color = 'r')
#     ax1.text(20, (ylim[1] - 0.05), 'FD #  above thresh = %s'%count,     va='center', size = 18, color = 'r')
#     ax1.text(20, (ylim[1] - 0.08), 'FD %% above thresh = %s%%'%percentFD, va='center', size = 18, color = 'r')
#     plt.axhline(threshold, color='r', linestyle='dashed', linewidth=2)
#
#     ax2 = plt.subplot2grid((1,7), (0,5), colspan= 2)
#     #bins = np.linspace(0, max_data, max_data + 1)
#     sns.distplot(FD_power,bins = 10, vertical = True )
#     ax2.set_ylim(ylim)
#
#     png_name = os.path.join(os.getcwd(), 'FD_QC_PLOT.png')
#     plt.savefig(png_name, dpi=190, bbox_inches='tight')
#     plt.close()
#
#     return png_name