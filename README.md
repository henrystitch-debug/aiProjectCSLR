# CorrNet+ MultiFrame Extension

## Overview

This repository is part of a university machine learning project focused on reproducing and improving the CorrNet+ Sign Language Recognition model. (Paper: CorrNet+: Sign Language Recognition and Translation via Spatial-Temporal Correlation)

The original CorrNet+ implementation was replicated as a baseline for academic study and experimentation. Using the same datasets, training pipeline, and evaluation methodology as the original work, we investigated several potential modifications to the architecture to evaluate whether they could improve recognition performance.

This branch contains the **MS-TCN-** approach.

## MS-TCN Approach
CorrNet+'s temporal module applies a single-scale 1D convolution over the frame-feature sequence before the BiLSTM. This design captures only one temporal resolution at each layer.

This motivated us to investigate a multi-scale temporal convolution network (MS-TCN) as a replacement for CorrNet+'s single-scale temporal convolution block. The inspiration comes from the USTM framework (Hasanaath et al., 2025), which introduces an MS-TCN component as part of a broader unified spatio-temporal model built on a Swin Transformer backbone. In USTM, the MS-TCN captures local temporal dynamics at multiple dilation rates, operating in parallel with a BiLSTM that handles longer-range dependencies. 

This branch is built on the hypothesis, that this multi-scale temporal front-end, isolated from the rest of USTM's architecture, could be grafted onto CorrNet+'s pipeline in place of the existing single-scale 1D convolution, enriching the temporal representation fed to the BiLSTM without altering the spatial or trajectory-modelling components. 

## Experimental Setup

To ensure fair comparison with the baseline implementation:

* The same datasets were used as in the original CorrNet+ paper.
* Training and evaluation procedures remained unchanged wherever possible.
* Training was performed for **15 epochs**.

## Results

| Model             | Epochs | Best WER         |
| ----------------- | ------ | ---------------- |
| MS-TCN             | 15     | **92**           |

The MSTCN approach did not outperform the reproduced baseline model.

## Discussion

Several factors may explain the observed results:

* The MS-TCN was designed and validated as part of a tightly coupled system; its contribution in USTM is attributable to the combination of all components rather than to the MS-TCN block in isolation.
* The upstream backbone here is CorrNet+'s ResNet-18 with correlation stages, which produces rich trajectory features but in a fundamentally different representational space from hierarchical Swin features. 
* Transplanting only the MS-TCN into a different backbone — without the TAPE adapter and without the Swin Transformer's hierarchical spatial representations — removes the context in which the multi-scale dilated branches are most beneficial.

## Acknowledgments

This project builds upon the original CorrNet+ research and implementation. Full credit for the original model architecture, methodology, and published results belongs to the original authors. This repository is maintained solely for educational and research purposes as part of a university machine learning course.
