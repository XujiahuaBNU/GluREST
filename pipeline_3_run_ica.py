__author__ = 'kanaan'
'''

## preprocessing
1. First 4 volumes were removed
2. Motion correction,
3. brain extraction,
4. spatial smoothing with a 5 FWHM
5. High-pass temporal filtering at 100 s.


##T1 structural images were segmented using FSL FIRST
Bilateral regions included in these masks were the entire striatum (comprising regions of caudate,
putamen, and ventral striatum), globus pallidus, amygdala, hippocampus, and thalamus and midbrain.
The unthresholded versions of these segmented structures were combined into a single, liberal mask image for each subject
To include mid- brain voxels within our masks, we carried out nonlinear warp trans- formation (as implemented in FSL FNIRT)
of 6 binary, bilateral volumes from the Talairach Daemon atlas (Lancaster et al. 2000; labels = midbrain, substantia nigra,
subthalamic nucleus, red nucleus, mammillary body, and medial geniculate body) to the high-resolution space of each subject
This midbrain information was then added to the mask containing subjects other subcortical regions.
These subject-specific combined masks were then affine-registered to EPI space using FSL FLIRT and used in
subsequent subject-wise SBCA, to quantify subcortical functional connectivity with neocortical RSNs.

##ICA
-probabilistic group ICA with temporal concatenation
-Template RSN maps were thresholded (at z > 3), then binarized, and transformed from MNI152 standard space to native space
-Voxels with less than 20% probability of containing gray matter in the equivalent T1 structural (as calculated using FSL FAST)
were removed from all subject-specific RSN spatial maps.

SBCA
SBCA, the subcortical seed masks from each subject were examined individually, in EPI space, for their voxel-wise spatial
distributions of functional connectivity strength with the characteristic activity of each of the 20 RSNs (7 subjected to higher-level
analysis and 13 nuisance).

20 non-artifactual components from group ICA were included in this first-level analysis to ensure that potential extraneous
interactions, or temporally overlapping relationships, between any of the 7 RSNs of interest and any of the 13 nuisance RSNs
could be factored out of the analysis, in effect treating the latter as confound regressors.

Voxel-wise connectivity strengths were quantified by calculating partial correlation coefficients between the BOLD signal time
series at each mask voxel and that of the weighted principal eigenvariate associated with each RSN.

Weighted principal eigenvariate associated with each RSN was calculated via subject wise principal component analyses

Voxelwise coefficients are termed partial because the analysis associated with a given target RSN controlled,
in turn, for the seed voxels activity relationship with each of the other 19 RSNs examined as targets in separate correlation analyses.

In these analyses, we also controlled for the confounding influences of structured noise from WM and CSF and residual motion artifacts.
To this end, binary T1-segmented maps of WM and CSF  were registered to EPI space using FLIRT and, for each session,
used as masks against the associated, preprocessed functional data sets, in order to extract confound time series that were calculated
as the mean BOLD signal within these tissue masks. In addition to the WM and CSF confounds, 6 time series resulting from the motion
correction procedure describing individual subject head motion parameters were also regressed out of the SBCA.

RSNs
(1) anterocentric DMN
(2) posterocentric DMN
(3) left-lateralized FPNs
(4) right-lateralized FPNs
(5) fronto-insular and
(6) dorsal media- lateral frontal salience executive networks SENs
(7) the hippocampal- parietal/ventral DMN.


To test the correlation maps resulting from SBCA for between group differences (7 RSNs x 49 subjects = 343 maps), maps were
transformed to MNI space,
Correlation maps were then arranged in a single 4-D file per RSN, containing, for each subject, a subcortical map of
connectivity with said RSN (thus 49 per RSN). These RSN-specific subcortical connectivity maps were then analyzed within the
framework of the general linear model, using nonparametric permutation testing (5000 permutations; as implemented in FSL randomise)
to identify subcortical regions in which functional connectivity with a given RSN of interest differed between dopamine drug treatment groups,
in terms of being more strongly or weakly positive or negative.

#######################################################################################################
Cerliani L, Mennes M, Thomas RM, Di Martino A, Thioux M, Keysers C. Increased functional connectivity between subcortical
and cortical resting-state networks in autism spectrum disorder. JAMA Psychiatry. Published online June 10, 2015. doi:10.1001/jamapsychiatry.2015.0101

Preprocessing
1. Structural (T1-weighted) images were skull stripped (and the results visually checked) after inhomogeneity correction.
Resting-state fMRI
- limited to 180 time points
- motion correction
- slice-timing correction;
- spatial smoothing (Gaussian kernel, FWHM=6mm);
- highpass temporal filtering (100 sec). no low pass
- Transformation to MNI
- resampling to 4x4x4xmm


ICA
masking out non-brain voxels,
voxel-wise demeaning
variance normalization of the RSfmri data.
- melodic
- dual regression:Spatial Regression: network-specific summary time courses are estimated using all the spatial modes extracted by the ICA.
                  In this way, artefactual signal from confounding components is regressed out from the estimation of the
                  summary time course for a specific functionally relevant network.
- limit frequencies to 0.009 and 0.08 Hz.


'''














def ica_aroma_denoise(fslDir, inFile, mask, dim, TR, mc, denType):
	import os


