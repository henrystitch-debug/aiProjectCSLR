# CorrNet+ MultiFrame Extension

## Overview

This repository is part of a university machine learning project focused on reproducing and improving the CorrNet+ Sign Language Recognition model. (Paper: CorrNet+: Sign Language Recognition and Translation via Spatial-Temporal Correlation)

The original CorrNet+ implementation was replicated as a baseline for academic study and experimentation. Using the same datasets, training pipeline, and evaluation methodology as the original work, we investigated several potential modifications to the architecture to evaluate whether they could improve recognition performance.

This branch contains the **Multiframe + MS-TCN** approach.

## Multiframe + MS-TCN Approach
This branch contains the code for the mixed approach of Multiframe and MS-TCN approach. The more detailed descriptions of both can be found on the separate branches.

## Experimental Setup

To ensure fair comparison with the baseline implementation:

* The same datasets were used as in the original CorrNet+ paper.
* Training and evaluation procedures remained unchanged wherever possible.
* Training was performed for **5 epochs**.

## Results

| Model               | Epochs | Best WER         |
| ------------------- | ------ | ---------------- |
| Multiframe + MS-TCN |   5    |    **91.6**      |

The Multiframe + MS-TCN approach did not outperform the reproduced baseline model.

## Discussion

Several factors may explain the observed results:

* The MS-TCN was designed and validated as part of a tightly coupled system; its contribution in USTM is attributable to the combination of all components rather than to the MS-TCN block in isolation.
* The upstream backbone here is CorrNet+'s ResNet-18 with correlation stages, which produces rich trajectory features but in a fundamentally different representational space from hierarchical Swin features. 
* Transplanting only the MS-TCN into a different backbone — without the TAPE adapter and without the Swin Transformer's hierarchical spatial representations — removes the context in which the multi-scale dilated branches are most beneficial.

## Acknowledgments

This project builds upon the original CorrNet+ research and implementation. Full credit for the original model architecture, methodology, and published results belongs to the original authors. This repository is maintained solely for educational and research purposes as part of a university machine learning course.
