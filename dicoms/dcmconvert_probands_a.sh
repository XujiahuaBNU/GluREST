#!/bin/bash

####### SCRIPT TO PREPARE TOURETE PROJECT DATA FOR ANALYSIS
####### COPIES ALL DICOMS TO A LOCAL FOLDER
####### CONVERTS ALL DICOMS TO NIFTIs WITH PROPER NAMES
####### OUTPUT DATA CAN BE FOUND at /scr/sambesi1/projects/MRS/study_a/data/NIFTI/Controls/${subject}/sequence



while read subject
do

echo '########################################################################'
echo                        'Starting for subject' ${subject}
echo '########################################################################'
echo    RUNNNING ISISCONV For Subject_${subject}
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/
mkdir /a/projects/nmr093a/probands/${subject}/NIFTI/
isisconv -in /a/projects/nmr093a/probands/${subject}/DICOM -out /a/projects/nmr093a/probands/${subject}/NIFTI/${subject}_S{sequenceNumber}_{sequenceDescription}_{echoTime}.nii -rf dcm -wdialect fsl

mv /a/projects/nmr093a/probands/${subject}/NIFTI/*mp2rage_p3_602B_INV1_2.98* /a/projects/nmr093a/probands/${subject}/NIFTI/MP2RAGE_INV1.nii
mv /a/projects/nmr093a/probands/${subject}/NIFTI/*mp2rage_p3_602B_INV2_2.98* /a/projects/nmr093a/probands/${subject}/NIFTI/MP2RAGE_INV2.nii

mv /a/projects/nmr093a/probands/${subject}/NIFTI/*mp2rage_p3_602B_DIV_Images_2.98* /a/projects/nmr093a/probands/${subject}/NIFTI/MP2RAGE_DIV.nii
mv /a/projects/nmr093a/probands/${subject}/NIFTI/*mp2rage_p3_602B_T1_Images_2.98* /a/projects/nmr093a/probands/${subject}/NIFTI/MP2RAGE_T1MAPS.nii
mv /a/projects/nmr093a/probands/${subject}/NIFTI/*mp2rage_p3_602B_UNI_Images_2.98* /a/projects/nmr093a/probands/${subject}/NIFTI/MP2RAGE_UNI.nii

mv /a/projects/nmr093a/probands/${subject}/NIFTI/*resting* /a/projects/nmr093a/probands/${subject}/NIFTI/REST.nii
mv /a/projects/nmr093a/probands/${subject}/NIFTI/*mbep2d_se_52* /a/projects/nmr093a/probands/${subject}/NIFTI/REST_SE.nii
mv /a/projects/nmr093a/probands/${subject}/NIFTI/*se_invpol_52* /a/projects/nmr093a/probands/${subject}/NIFTI/REST_SE_INVPOL.nii

mv /a/projects/nmr093a/probands/${subject}/NIFTI/*bval /a/projects/nmr093a/probands/${subject}/NIFTI/DWI_BVAL.bval
mv /a/projects/nmr093a/probands/${subject}/NIFTI/*bvec /a/projects/nmr093a/probands/${subject}/NIFTI/DWI_BVEC.bvec
mv /a/projects/nmr093a/probands/${subject}/NIFTI/*AP_unwarp_diff* /a/projects/nmr093a/probands/${subject}/NIFTI/DWI_AP.nii
mv /a/projects/nmr093a/probands/${subject}/NIFTI/*PA_unwarp_diff* /a/projects/nmr093a/probands/${subject}/NIFTI/DWI_PA.nii
mv /a/projects/nmr093a/probands/${subject}/NIFTI/*mbep2d_diff_80* /a/projects/nmr093a/probands/${subject}/NIFTI/DWI.nii

rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*AAH*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*AX*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*ax*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*Ax*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*Cor*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*COR*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*cor*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*hip*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*slab*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*Modus*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*acpc*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*DUMMY*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*dummy*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*MPR-Modus*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*short*
rm -rf /a/projects/nmr093a/probands/${subject}/NIFTI/*SLAB*
echo '########################################################################'
echo                        'Complete conversion for this' ${subject}
echo '########################################################################'
done<${1}