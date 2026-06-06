# CorrNet+ MultiFrame Extension

## Overview

This repository is part of a university machine learning project focused on reproducing and improving the CorrNet+ Sign Language Recognition model. (Paper: CorrNet+: Sign Language Recognition and Translation via Spatial-Temporal Correlation)

The original CorrNet+ implementation was replicated as a baseline for academic study and experimentation. Using the same datasets, training pipeline, and evaluation methodology as the original work, we investigated several potential modifications to the architecture to evaluate whether they could improve recognition performance.

This branch contains the **MultiFrame** approach.

## MultiFrame Approach

The original CorrNet+ model primarily models spatial-temporal correlations between neighboring frames. The motivation behind the MultiFrame approach was to provide the model with a broader temporal context by allowing it to consider information from multiple surrounding frames rather than relying only on immediately adjacent frames.

Sign language recognition depends heavily on movement patterns that often span several frames. By extending the temporal window, we hypothesized that the model could better capture longer-term motion dynamics and improve its understanding of complex gestures.

The key idea of this branch is therefore to increase the temporal context available during correlation modeling, enabling the network to learn relationships across a larger sequence of frames.

## Experimental Setup

To ensure fair comparison with the baseline implementation:

* The same datasets were used as in the original CorrNet+ paper.
* The same preprocessing pipeline was maintained.
* Training and evaluation procedures remained unchanged wherever possible.
* Only the temporal correlation mechanism was modified to incorporate information from multiple frames.

Training was performed for **15 epochs**.

## Results

| Model             | Epochs | Best WER         |
| ----------------- | ------ | ---------------- |
| MultiFrame        | 15     | **34.8**         |

The MultiFrame approach did not outperform the reproduced baseline model. Although the model successfully trained and learned meaningful representations, the final Word Error Rate (WER) remained above the baseline performance.

## Discussion

Several factors may explain the observed results:

* Increasing the temporal window may introduce additional noise from less relevant frames.
* The model may require more training epochs to fully benefit from the larger temporal context.
* Additional hyperparameter tuning could be necessary to effectively utilize information from multiple frames.
* The original CorrNet+ architecture may already capture sufficient temporal information through its existing correlation mechanisms.

While the MultiFrame approach did not improve performance in this experiment, it provided valuable insight into how temporal context affects sign language recognition models and serves as a useful comparison point for future modifications.

## Acknowledgments

This project builds upon the original CorrNet+ research and implementation. Full credit for the original model architecture, methodology, and published results belongs to the original authors. This repository is maintained solely for educational and research purposes as part of a university machine learning course.
