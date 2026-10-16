# ASD-Cerebellar-Functional-Gradients
Analysis code for investigating cerebellar functional gradients in autism spectrum disorder (ASD).

## Overview
Content includes source code for cerebellar functional gradient analysis. Due to the large size of the analyzed fMRI data, raw ABIDE data are not deposited here. The project is structured into six parts corresponding to the major analyses in the article, including gradient calculation, between-group statistics, individual gradient mapping, meta-analysis, cognitive terms decoding, and gene expression association analysis.

## Toolboxes
SPM12, https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
BrainSpace for gradient embedding, https://github.com/MICA-MNI/BrainSpace
ComBatHarmonization, ver. 20180620, https://github.com/Jfortin1/ComBatHarmonization
AHBAprocessing, ver. 20181025, https://github.com/BMHLab/AHBAprocessing
Scripts from Whitaker et al. PNAS 2016, https://github.com/KirstieJane/NSPN_WhitakerVertes_PNAS2016

Other software and web-based tools include:
Nibabel (Python), https://nibabel.readthedocs.io/
NeuroSynth, web-based decoding functions were used, https://neurosynth.org/
Gorilla, http://cbl-gorilla.cs.technion.ac.il/

We thank the authors and developers for providing these wonderful tools for data analysis.

## Gradient analysis
This part was primarily carried out using custom MATLAB scripts with gradient embedding implemented by BrainSpace and site effect correction using ComBatHarmonization (https://github.com/Jfortin1/ComBatHarmonization). Please see the comments in the scripts for details. The analysis details include:
Preprocess ABIDE CIFTI resting-state fMRI data.
Generate cerebellar voxel-wise connectivity matrix and calculate cerebellar functional gradients.
Generate group-averaged cerebellar gradient maps for ASD and HC groups.
Calculate the explained variance of each gradient for each individual.
Correct site effects of gradient scores by using ComBatHarmonization.
Calculate gradient spatial dispersion and similarity.
Export gradient-related t-maps to NIfTI files for subsequent meta-analysis.

## Cognitive terms
We used Neurosynth (https://neurosynth.org/) to assess the topic terms associated with alterations in cerebellar functional gradients in ASD.
The thresholded Z-maps derived from the between-group comparisons were submitted to the Neurosynth decoder.
For each of the maps, noncognitive terms (e.g., anatomical and demographic terms) were removed.
The cognitive terms were visualized on a word-cloud plot with font size scaled according to correlation with Neurosynth meta-analytic maps.

## Classification validation
We validated the discriminative capacity of cerebellar gradient features using leave-one-site-out (LOSO) SVM classification with permutation testing.
The LOSO cross-validation strategy was used to avoid site bias: exclude one acquisition site at each fold and train the SVM classifier on remaining sites.
Permutation test was performed to evaluate statistical significance of classification accuracy.

## Behavioral CCA
Canonical correlation analysis (CCA) was performed to identify multivariate association between cerebellar functional gradient patterns and ASD-related behavioral metrics.
Permutation-based inference was used to assess the significance of CCA modes.

## Gene expression association
In this analysis, we adapted revised scripts from Whitaker et al. PNAS 2016, (https://github.com/KirstieJane/NSPN_WhitakerVertes_PNAS2016). The main script is named gene_pls_cerebellar_gradient.m, see the code in this file for more details.
Gene expression data from the Allen Institute for Brain Science were preprocessed using AHBAprocessing (https://github.com/BMHLab/AHBAprocessing), obtaining gene expression profiles for cerebellar ROIs.
Perform PLS analysis to examine the association between between-group gradient Z-maps and gene expression profiles.
Significance of the PLS component was determined by permutation test.
The weights of PLS components were determined by the bootstrap method.
Both the descending and ascending order of PLS weighted genes were submitted to Metascape (https://metascape.org/) for enrichment analysis. See the Metascape website for instructions.
