# Import packages
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
# Import functions
from network.get_all import *

# Function to create the network centrality workflow
def create_resting_state_graphs(allocated_memory = None,
                                wf_name = 'resting_state_graph'):
    '''
    Workflow to calculate degree and eigenvector centrality as well as
    local functional connectivity density (lfcd) measures for the
    resting state data.

    Parameters
    ----------
    generate_graph : boolean
        when true the workflow plots the adjacency matrix graph
        and converts the adjacency matrix into compress sparse
        matrix and stores it in a .mat file. By default its False
    wf_name : string
        name of the workflow

    Returns
    -------
    wf : workflow object
        resting state graph workflow object

    Notes
    -----

    `Source <https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/network_centrality/resting_state_centrality.py>`_

    Workflow Inputs::

        inputspec.subject: string (nifti file)
            path to resting state input data for which centrality measure is to be calculated

        inputspec.template : string (existing nifti file)
            path to mask/parcellation unit

        inputspec.method_option: string (int)
            0 for degree centrality, 1 for eigenvector centrality, 2 for lFCD

        inputspec.threshold: string (float)
            pvalue/sparsity_threshold/threshold value

        inputspec.threshold_option: string (int)
            threshold options:  0 for probability p_value, 1 for sparsity threshold, any other for threshold value

        centrality_options.weight_options : string (list of boolean)
            list of two booleans for binarize and weighted options respectively

        centrality_options.method_options : string (list of boolean)
            list of two booleans for Degree and Eigenvector centrality method options respectively

    Workflow Outputs::

        outputspec.centrality_outputs : string (list of nifti files)
            path to list of centrality outputs for binarized or/and weighted and
            degree or/and eigen_vector

        outputspec.threshold_matrix : string (numpy file)
            path to file containing thresholded correlation matrix

        outputspec.correlation_matrix : string (numpy file)
            path to file containing correlation matrix

        outputspec.graph_outputs : string (mat and png files)
            path to matlab compatible sparse adjacency matrix files
            and adjacency graph images

    Order of commands:

    - load the data and template, based on template type (parcellation unit ar mask)
      extract timeseries

    - Calculate the correlation matrix for the image data for each voxel in the mask or node
      in the parcellation unit

    - Based on threshold option (p_value or sparsity_threshold), calculate the threshold value

    - Threshold the correlation matrix

    - Based on weight options for edges in the network (binarize or weighted), calculate Degree
      or Vector Based centrality measures


    High Level Workflow Graph:

    .. image:: ../images/resting_state_centrality.dot.png
       :width: 1000


    Detailed Workflow Graph:

    .. image:: ../images/resting_state_centrality_detailed.dot.png
       :width: 1000

    Examples
    --------

    #>>> import resting_state_centrality as graph
    #>>> wflow = graph.create_resting_state_graphs()
    #>>> wflow.inputs.centrality_options.method_options=[True, True]
    #>>> wflow.inputs.centrality_options.weight_options=[True, True]
    #>>> wflow.inputs.inputspec.subject = '/home/work/data/rest_mc_MNI_TR_3mm.nii.gz'
    #>>> wflow.inputs.inputspec.template = '/home/work/data/mask_3mm.nii.gz'
    ##>> wflow.inputs.inputspec.threshold_option = 1
    ##>> wflow.inputs.inputspec.threshold = 0.0744
    ##>> wflow.base_dir = 'graph_working_directory'
    #>>> wflow.run()

    '''

    # Instantiate workflow with input name
    wf = pe.Workflow(name = wf_name)

    # Instantiate inputspec node
    inputspec = pe.Node(util.IdentityInterface(fields=['subject',
                                                       'template',
                                                       'method_option',
                                                       'threshold_option',
                                                       'threshold',
                                                       'weight_options']),
                        name='inputspec')

    # Instantiate calculate_centrality main function node
    calculate_centrality = pe.Node(util.Function(input_names = ['datafile',
                                                                'template',
                                                                'method_option',
                                                                'threshold_option',
                                                                'threshold',
                                                                'weight_options',
                                                                'allocated_memory'],
                                                 output_names = ['out_list'],
                                                 function = calc_centrality),
                                   name = 'calculate_centrality')
    calculate_centrality.inputs.allocated_memory = 20

    # Connect inputspec node to main function node
    wf.connect(inputspec, 'subject',
               calculate_centrality, 'datafile')
    wf.connect(inputspec, 'template',
               calculate_centrality, 'template')
    wf.connect(inputspec, 'method_option',
               calculate_centrality, 'method_option')
    wf.connect(inputspec, 'threshold_option',
               calculate_centrality, 'threshold_option')
    wf.connect(inputspec, 'threshold',
               calculate_centrality, 'threshold')
    wf.connect(inputspec,'weight_options',
               calculate_centrality,'weight_options')



    # Instantiate outputspec node
    outputspec = pe.Node(util.IdentityInterface(fields=['centrality_outputs',
                                                        'threshold_matrix',
                                                        'correlation_matrix',
                                                        'graph_outputs']),
                         name = 'outputspec')

    # Connect function node output list to outputspec node
    wf.connect(calculate_centrality, 'out_list',
               outputspec, 'centrality_outputs')

    # Return the connected workflow
    return wf


# Main centrality function utilized by the centrality workflow
def calc_centrality(datafile,
                    template,
                    method_option,
                    threshold_option,
                    threshold,
                    weight_options,
                    allocated_memory):
    '''
    Method to calculate centrality and map them to a nifti file

    Parameters
    ----------
    datafile : string (nifti file)
        path to subject data file
    template : string (nifti file)
        path to mask/parcellation unit
    method_option : integer
        0 - degree centrality calculation, 1 - eigenvector centrality calculation, 2 - lFCD calculation
    threshold_option : an integer
        0 for probability p_value, 1 for sparsity threshold,
        2 for actual threshold value, and 3 for no threshold and fast approach
    threshold : a float
        pvalue/sparsity_threshold/threshold value
    weight_options : list (boolean)
        list of booleans, where, weight_options[0] corresponds to binary counting
        and weight_options[1] corresponds to weighted counting (e.g. [True,False])
    allocated_memory : string
        amount of memory allocated to degree centrality

    Returns
    -------
    out_list : list
        list containing out mapped centrality images
    '''

    # Import packages
    from network.get_all import load, get_centrality_by_rvalue, get_centrality_by_sparsity, get_centrality_fast
    from network.utils   import calc_blocksize, norm_cols, convert_pvalue_to_r, map_centrality_matrix

    # Check for input errors
    if weight_options.count(True) == 0:
        raise Exception("Invalid values in weight options" \
                        "At least one True value is required")
    # If it's sparsity thresholding, check for (0,1]
    if threshold_option == 1:
        if threshold <= 0 or threshold > 1:
            raise Exception('Threshold value must be a positive number'\
                            'greater than 0 and less than or equal to 1.'\
                            '\nCurrently it is set at %d' % threshold)
    if method_option == 2 and threshold_option != 2:
        raise Exception('lFCD must use correlation-type thresholding.'\
                         'Check the pipline configuration has this setting')
    import time
    start = time.clock()

    # Init variables
    out_list = []
    ts, aff, mask, t_type, scans = load(datafile, template)

    # If we're doing eigenvectory centrality, need entire correlation matrix
    if method_option == 0 and threshold_option == 1:
        block_size = calc_blocksize(ts, memory_allocated=allocated_memory,
                                    sparsity_thresh=threshold)
    elif method_option == 1:
        block_size = calc_blocksize(ts, memory_allocated=allocated_memory,
                                    include_full_matrix=True)
    # Otherwise, compute blocksize with regards to available memory
    else:
        block_size = calc_blocksize(ts, memory_allocated=allocated_memory,
                                    include_full_matrix=False)
    # Normalize the timeseries for easy dot-product correlation calc.
    ts_normd = norm_cols(ts.T)

    # P-value threshold centrality
    if threshold_option == 0:
        r_value = convert_pvalue_to_r(scans, threshold)
        centrality_matrix = get_centrality_by_rvalue(ts_normd,
                                                     mask,
                                                     method_option,
                                                     weight_options,
                                                     r_value,
                                                     block_size)
    # Sparsity threshold
    elif threshold_option == 1:
        centrality_matrix = get_centrality_by_sparsity(ts_normd,
                                                       method_option,
                                                       weight_options,
                                                       threshold,
                                                       block_size)
    # R-value threshold centrality
    elif threshold_option == 2:
        centrality_matrix = get_centrality_by_rvalue(ts_normd,
                                                     mask,
                                                     method_option,
                                                     weight_options,
                                                     threshold,
                                                     block_size)
    # For fast approach (no thresholding)
    elif threshold_option == 3:
        centrality_matrix = get_centrality_fast(ts, method_option)
    # Otherwise, incorrect input for threshold_option
    else:
        raise Exception('Option must be between 0-3 and not %s, check your '\
                        'pipeline config file' % str(threshold_option))

    # Print timing info
    print 'Timing:', time.clock() - start

    # Map the arrays back to images
    for mat in centrality_matrix:
        centrality_image = map_centrality_matrix(mat, aff, mask, t_type)
        out_list.append(centrality_image)

    # Finally return
    return out_list
