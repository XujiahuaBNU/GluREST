__author__ = 'kanaan' 'Dec 23 2014'



#  Forked from CPAC 038

def generate_roi_timeseries(data_file, template, output_type):
    """
    Method to extract mean of voxel across all timepoints for each node in roi mask

    Parameters
    ----------
    datafile : string
        path to input functional data
    template : string
        path to input roi mask in functional native space
    output_type : list
        list of two boolean values suggesting
        the output types - numpy npz file and csv
        format

    Returns
    -------
    out_list : list
        list of 1D file, txt file, csv file and/or npz file containing
        mean timeseries for each scan corresponding
        to each node in roi mask

    Raises
    ------
    Exception

    """
    import nibabel as nib
    import csv
    import numpy as np
    import os
    import shutil

    unit_data = nib.load(template).get_data()
    # Cast as rounded-up integer
    unit_data = np.int64(np.ceil(unit_data))
    datafile = nib.load(data_file)
    img_data = datafile.get_data()
    vol = img_data.shape[3]

    if unit_data.shape != img_data.shape[:3]:
        raise Exception('Invalid Shape Error.'\
                        'Please check the voxel dimensions.'\
                        'Data and roi should have'\
                        'same shape')

    nodes = np.unique(unit_data).tolist()
    sorted_list = []
    node_dict = {}
    out_list = []


    # extracting filename from input template
    tmp_file = os.path.splitext(os.path.basename(template))[0]
    tmp_file = os.path.splitext(tmp_file)[0]
    oneD_file = os.path.abspath('roi_' + tmp_file + '.1D')
    txt_file = os.path.abspath('roi_' + tmp_file + '.txt')
    csv_file = os.path.abspath('roi_' + tmp_file + '.csv')
    numpy_file = os.path.abspath('roi_' + tmp_file + '.npz')

    nodes.sort()
    for n in nodes:
        if n > 0:
            node_array = img_data[unit_data == n]
            node_str = 'node_%s' % (n)
            avg = np.mean(node_array, axis=0)
            avg = np.round(avg, 6)
            list1 = [n] + avg.tolist()
            sorted_list.append(list1)
            node_dict[node_str] = avg.tolist()


    # writing to 1Dfile
    print "writing 1D file.."
    f = open(oneD_file, 'w')
    writer = csv.writer(f, delimiter='\t')

    value_list = []

    new_keys = sorted([int(float(key.split('node_')[1])) for key in node_dict.keys()])

    roi_number_list = [str(n) for n in new_keys]

    roi_number_str = []
    for number in roi_number_list:

        roi_number_str.append("#" + number)


    print "new keys: ", new_keys
    print "roi number str: ", roi_number_str
    for key in new_keys:
        value_list.append(node_dict['node_%s' % key])

    column_list = zip(*value_list)


    writer.writerow(roi_number_str)

    for column in column_list:
        writer.writerow(list(column))
    f.close()
    out_list.append(oneD_file)

    # copy the 1D contents to txt file
    shutil.copy(oneD_file, txt_file)
    out_list.append(txt_file)

    # if csv is required
    if output_type[0]:
        print "writing csv file.."
        f = open(csv_file, 'wt')
        writer = csv.writer(f, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
        headers = ['node/volume'] + np.arange(vol).tolist()
        writer.writerow(headers)
        writer.writerows(sorted_list)
        f.close()
        out_list.append(csv_file)

    # if npz file is required
    if output_type[1]:
        print "writing npz file.."
        np.savez(numpy_file, roi_data=value_list, roi_numbers=roi_number_list)
        out_list.append(numpy_file)

    return out_list