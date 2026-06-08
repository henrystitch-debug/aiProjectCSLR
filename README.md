# CorrNet+ MultiFrame Extension

## Overview

This repository is part of a university machine learning project focused on reproducing and improving the CorrNet+ Sign Language Recognition model. (Paper: CorrNet+: Sign Language Recognition and Translation via Spatial-Temporal Correlation)

The original CorrNet+ implementation was replicated as a baseline for academic study and experimentation. Using the same datasets, training pipeline, and evaluation methodology as the original work, we investigated several potential modifications to the architecture to evaluate whether they could improve recognition performance.

This branch contains the **Self-Correlation** approach.

## Self-Correlation Approach

CorrNet+'s correlation module is fundamentally a cross-frame operation: for each frame t, it computes a compact representation and then correlates it against the spatial feature maps of neighbouring frames. This produces an affinity-weighted sum that highlights spatial regions in adjacent frames that are consistent with, or in motion relative to, the current frame — the mechanism by which the model tracks trajectories across time. 

As a single 1x1 summary vector, it discards the spatial layout of x(t) before the cross-frame correlation step. Any within-frame relationship between the compact representation and the full spatial map is never explicitly modelled. 

This motivated our second experimental branch: augmenting the existing cross-frame correlation with a self-correlation term that relates the representation back to the spatial map of its own frame. The intuition is that such a self-correlation acts as a form of within-frame spatial re-weighting: Positions that strongly match the summary — likely the most semantically active regions — receive higher weight, while uninformative background positions are suppressed.

## Experimental Setup

To ensure fair comparison with the baseline implementation:

* The same datasets were used as in the original CorrNet+ paper.
* The same preprocessing pipeline was maintained.
* Training and evaluation procedures remained unchanged wherever possible.
* Only the correlation module was modified to incorporate self-correlation.
Training was performed for **15 epochs**.

## Results

| Model             | Epochs | Best WER         |
| ----------------- | ------ | ---------------- |
| Self-Correlation  | 37     | **31.4**         |

The Self-Correlation approach did not outperform the reproduced baseline model. Although the model successfully trained and learned meaningful representations, the final Word Error Rate (WER) remained above the baseline performance.

## Discussion

Several factors may explain the observed results:

* It operates at the same location in the pipeline as the existing cross-frame correlation, so self-correlation is architecturally native to CorrNet+.
* The best checkpoint occurred relatively early (epoch 12), after which performance degraded — a sign of overfitting. The single additional parameter (alpha_self) likely reached a useful value quickly, but the rest of the network may have begun to overfit the self-correlation signal in later epochs.
* Regularisation strategies such as dropout on the affinity map, weight decay tuning, or early stopping based on a held-out validation schedule could potentially improve the final result. 

 
While the Self-Correlation approach did not improve performance in this experiment, it provided valuable insight into how temporal context affects sign language recognition models and serves as a useful comparison point for future modifications.

## Acknowledgments

This project builds upon the original CorrNet+ research and implementation. Full credit for the original model architecture, methodology, and published results belongs to the original authors. This repository is maintained solely for educational and research purposes as part of a university machine learning course.
