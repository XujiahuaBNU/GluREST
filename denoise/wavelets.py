__author__ = 'kanaan'

from nipype.pipeline.engine import Workflow, Node
import nipype.interfaces.utility as util
from nipype.interfaces.base import CommandLine
import os


#
#
# ============================================
#          BrainWavelet Toolbox v1.1
# ============================================
#
# Author: Ameera Patel, 2014
#
# Reference:
# Patel, et.al (2014) A wavelet method for modeling and despiking motion
# artifacts from resting- state fmri time series. Neuroimage, 95, 287-304
# doi:10.1016/ Neuroimage.2014.03.012
# www.brainwavelet.org.
#
#
__author__ = 'kanaan'

from nipype.pipeline.engine import Workflow, Node
import nipype.interfaces.utility as util
from nipype.interfaces.base import CommandLine


func = '/SCR2/tmp/wavelet_via_py/func2mni_preproc.nii'


def matlab_despike_command(func):
    import os
    import subprocess
    # make sure your nifti is unzipped


    cur_dir = os.getcwd()

    matlab_command = ['matlab',
                      '-nodesktop' ,
                      '-nosplash',
          '-r "WaveletDespike(\'%s\',\'%s/rest_dn\', \'wavelet\', \'d4\', \'LimitRAM\', 10) ; quit;"' %(func, cur_dir)]

    print ''
    print 'Running matlab through python...........Bitch Please....'
    print ''
    print  subprocess.list2cmdline(matlab_command)
    print ''

    subprocess.call(matlab_command)

    spike_percent   = [os.path.join(cur_dir,i) for i in os.listdir(cur_dir) if 'SP' in i][0]
    noise_img       = [os.path.join(cur_dir,i) for i in os.listdir(cur_dir) if 'noise' in i][0]
    despiked_img    = [os.path.join(cur_dir,i) for i in os.listdir(cur_dir) if 'wds' in i][0]

    return despiked_img, noise_img, spike_percent

def WaveletDespike():

    flow  = Workflow('denoise_wavelet_despike')

    inputnode  = Node(util.IdentityInterface(fields=['func_mni']),
                     name = 'inputnode')

    outputnode = Node(util.IdentityInterface(fields=['despiked_img',
                                                     'noise_img',
                                                     'spike_percent']),
                     name = 'outputnode')

    wavelet_denoise = Node(util.Function(input_names   = ['func'],
                                         output_names  =['despiked_img', 'noise_img', 'spike_percent'],
                                         function      =matlab_despike_command),
                           name ='wavelet_despike')

    flow.connect(inputnode,        'func_mni',        wavelet_denoise,   'func'          )
    flow.connect(wavelet_denoise,  'noise_img',       outputnode, 	 'despiked_img'  )
    flow.connect(wavelet_denoise,  'despiked_img',    outputnode,        'noise_img'     )
    flow.connect(wavelet_denoise,  'spike_percent',   outputnode,        'spike_percent' )

    return flow
