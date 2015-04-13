__author__ = 'kanaan' 'Dec 18 2014'
# -*- coding: utf-8 -*-

#anatomical
#okay ### IMAGE: visual of skull strip
#okay ### IMAGE: GM-WM-CSF on T1
### IMAGE: MNI edge on anatomical

#functional
# IMAGE:T1 Edge on Mean Functional image
# PLOT: func-T1 similarity
# IMAGE: MNI Edge on Mean Functional image


import nipype.interfaces.utility as util
from nipype.pipeline.engine import Node, Workflow

from qc_utils     import make_edge
from plots        import plot_FD, plot_nuisance_residuals
from make_montage import make_montage_axial, make_montage_sagittal, montage_tissues_axial, montage_tissues_sagittal


def create_qc_report():
    flow = Workflow('quality_control')

    inputnode        = Node(util.IdentityInterface(fields=['subject_id',
                                                           'anat_native',
                                                           'anat2func',
                                                           'anat_wm',
                                                           'anat_gm',
                                                           'anat_csf',
                                                           'func_wm',
                                                           'func_gm',
                                                           'func_csf',
                                                           'mov_params',
                                                           'fd1d',
                                                           'func_preprocessed',
                                                           'func_preprocessed_mask',
                                                           'detrend',
                                                           'detrend_mc',
                                                           'detrend_mc_compcor',
                                                           'detrend_mc_compcor_wmcsf_global']),

                                                    name  = 'inputnode'  )
    outputnode       = Node(util.IdentityInterface(fields = ['report']),
                                                    name  = 'outputnode' )

    '######################################################################'
    '''                     Anatomical Skullstripping                    '''
    '######################################################################'

    anat_edge = Node(util.Function(   input_names  = ['file_',],
                                      output_names = ['new_fname'],
                                      function     = make_edge),
                                      name         = 'anat_edge')

    anat_skull_ax =Node(util.Function(input_names  = [ 'overlay',
                                                       'underlay',
                                                       'png_name',
                                                       'cbar_name'],
                                      output_names = ['png_name'],
                                      function     = make_montage_axial),
                                      name         = 'anat_skullstrip_axial')
    anat_skull_ax.inputs.png_name  = 'anat_skull_vis_axial.png'
    anat_skull_ax.inputs.cbar_name = 'red'

    flow.connect(inputnode, 'anat_native',   anat_edge,         'file_'    )
    flow.connect(inputnode, 'anat_native',   anat_skull_ax,     'underlay' )
    flow.connect(anat_edge, 'new_fname',     anat_skull_ax,     'overlay' )



    anat_skull_sag =Node(util.Function(input_names  = [ 'overlay',
                                                          'underlay',
                                                          'png_name',
                                                          'cbar_name'],
                                      output_names = ['png_name'],
                                      function     = make_montage_sagittal),
                                      name         = 'anat_skullstrip_saggital')
    anat_skull_sag.inputs.png_name  = 'anat_skull_vis_saggital.png'
    anat_skull_sag.inputs.cbar_name = 'red'


    flow.connect(inputnode, 'anat_native',   anat_skull_sag, 'underlay' )
    flow.connect(anat_edge, 'new_fname',     anat_skull_sag, 'overlay' )


    '######################################################################'
    '''                         Anatomical Segmentation                  '''
    '######################################################################'

    anat_tissue_ax   = Node(util.Function(input_names  = ['overlay_csf',
                                                          'overlay_wm',
                                                          'overlay_gm',
                                                          'underlay',
                                                          'png_name'],
                                      output_names = ['png_name'],
                                      function     = montage_tissues_axial),
                                      name         = 'anat_tissue_axial')

    anat_tissue_ax.inputs.png_name = 'anat_tissue_axial.png'


    anat_tissue_sagg = Node(util.Function(input_names  = ['overlay_csf',
                                                          'overlay_wm',
                                                          'overlay_gm',
                                                          'underlay',
                                                          'png_name'],
                                      output_names = ['png_name'],
                                      function     = montage_tissues_sagittal),
                                      name         = 'anat_tissue_saggital')

    anat_tissue_sagg.inputs.png_name = 'anat_tissue_saggital.png'


    flow.connect(inputnode, 'anat_native',   anat_tissue_ax,   'underlay'    )
    flow.connect(inputnode, 'anat_wm',       anat_tissue_ax,   'overlay_wm'  )
    flow.connect(inputnode, 'anat_gm',       anat_tissue_ax,   'overlay_gm' )
    flow.connect(inputnode, 'anat_csf',      anat_tissue_ax,   'overlay_csf'  )

    flow.connect(inputnode, 'anat_native',   anat_tissue_sagg,  'underlay'    )
    flow.connect(inputnode, 'anat_wm',       anat_tissue_sagg,  'overlay_wm'  )
    flow.connect(inputnode, 'anat_gm',       anat_tissue_sagg,  'overlay_gm' )
    flow.connect(inputnode, 'anat_csf',      anat_tissue_sagg,  'overlay_csf'  )


    ''' Anatomical ---> Functional'''
    ''' Anatomical ---> MNI'''

    ''' Functional ---> MNI'''
    ''' Functional temporal SNR'''
    ''' Functional Tissue masks '''

    '######################################################################'
    '''                 Functional: Framewise Displacement               '''
    '######################################################################'

    fd      =  Node(util.Function(input_names  = ['fd1d',],
                                  output_names = ['png_name'],
                                  function     = plot_FD),
                                  name         = 'plot_fd')
    flow.connect(inputnode, 'fd1d',  fd,  'fd1d'    )


    '######################################################################'
    '''                    Functional: Nuisance plot                     '''
    '######################################################################'

    nuisance =  Node(util.Function(input_names  = [  'mov_params',
                                                     'fd1d',
                                                     'func_preprocessed',
                                                     'func_gm',
                                                     'func_preprocessed_mask',
                                                     'detrend',
                                                     'detrend_mc',
                                                     'detrend_mc_compcor',
                                                     'detrend_mc_compcor_wmcsf_global'],
                                  output_names = ['png_name'],
                                  function     = plot_nuisance_residuals),
                                  name         = 'visualize_nuisance_residuals')
    # denoise.inputs.ignore_exception = True

    flow.connect(inputnode, 'fd1d',                             nuisance,         'fd1d'                               )
    flow.connect(inputnode, 'mov_params',                       nuisance,         'mov_params'                         )
    flow.connect(inputnode, 'func_gm',                          nuisance,         'func_gm'                            )
    flow.connect(inputnode, 'func_preprocessed',                nuisance,         'func_preprocessed'                  )
    flow.connect(inputnode, 'func_preprocessed_mask',           nuisance,         'func_preprocessed_mask'             )
    flow.connect(inputnode, 'detrend',                          nuisance,         'detrend'                            )
    flow.connect(inputnode, 'detrend_mc',                       nuisance,         'detrend_mc'                         )
    flow.connect(inputnode, 'detrend_mc_compcor',               nuisance,         'detrend_mc_compcor'                 )
    flow.connect(inputnode, 'detrend_mc_compcor_wmcsf_global',  nuisance,         'detrend_mc_compcor_wmcsf_global'    )

    return flow

    # ################CRASH##################
    #
    # ''' generate pdf '''
    # canvas  =  Node(util.Function(input_names  = ['framewise_displacement'],
    #                               output_names = ['plot'],
    #                               function     = create_canvas),
    #                               name         = 'pdf_report')
    #
    # flow.connect(inputnode,          'subject_id',       canvas,         'subject_id'             )
    # flow.connect(anat_skull_ax,      'png_name',         canvas,         'anat_skull_axial'       )
    # flow.connect(anat_skull_sag,     'png_name',         canvas,         'anat_skull_saggital'    )
    # flow.connect(anat_tissue_ax,     'png_name',         canvas,         'anat_tissue_axial'      )
    # flow.connect(anat_tissue_sagg,   'png_name',         canvas,         'anat_tissue_saggital'   )
    # flow.connect(fd,                 'png_name',         canvas,         'framewise_displacement'   )


#
#
# def create_canvas(framewise_displacement,
#                  subject_id,
#                  anat_skull_axial,
#                  anat_skull_saggital,
#                  anat_tissue_axial,
#                  anat_tissue_saggital):
#
#     from reportlab.pdfgen import canvas
#     from reportlab.lib.units import inch
#
#     #create canvas
#     plot =  canvas.Canvas('report_%s.pdf' %subject_id, pagesize=(1918, 2220))
#
#     #page 1
#     plot.setFont("Helvetica", 50)
#     plot.drawString(inch*11, inch*29, 'SUBJECT RB1T' )
#     plot.drawImage(anat_skull_axial, 1, inch*15)
#     plot.drawImage(anat_skull_saggital, 1, inch*1)
#     plot.drawString(inch*10, inch*0.5, 'Anatomical Skullstriping' )
#     plot.showPage()
#
#     #page 2
#     plot.setFont("Helvetica", 50)
#     plot.drawImage(anat_tissue_axial, 1, inch*15)
#     plot.drawImage(anat_tissue_saggital, 1, inch*1)
#     plot.drawString(inch*9.4, inch*0.5, 'Anatomical Segmentation' )
#     plot.showPage()
#
#     #page 3
#     plot.drawImage(framewise_displacement, 1, inch*1)
#     plot.showPage()
#
#     plot.save()
#
#     return plot
