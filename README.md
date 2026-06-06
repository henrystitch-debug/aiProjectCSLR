# CorrNet+ Research Extension Project

## Overview

This repository is part of a university machine learning project focused on understanding, reproducing, and improving state-of-the-art Sign Language Recognition (SLR) models.

Our work is based on the original **CorrNet+** implementation presented in the paper:

> *CorrNet+: Sign Language Recognition and Translation via Spatial-Temporal Correlation* (2024)

The original repository was replicated as a baseline for academic study and experimentation. The goal of this project is to analyze the architecture, training process, and performance of CorrNet+, and investigate potential improvements through a series of independent experimental approaches.

This repository is **not the original CorrNet+ implementation**. Instead, it serves as a research environment for reproducing the published results and evaluating modifications developed as part of our coursework.

## Original Work

CorrNet+ is a deep learning framework for Continuous Sign Language Recognition (CSLR) and Sign Language Translation (SLT). The model introduces spatial-temporal correlation modeling to better capture relationships between body movements across video frames, allowing it to learn meaningful motion patterns and improve recognition performance.

The original authors demonstrated state-of-the-art results on several benchmark datasets, including:

* PHOENIX14
* PHOENIX14-T
* CSL-Daily
* CSL

Their approach focuses on modeling human body trajectories and motion dynamics across adjacent frames, enabling more accurate sign language understanding.

## Project Goals

The objectives of this project are:

1. Reproduce the baseline CorrNet+ training pipeline.
2. Understand the architecture and training methodology used in the original paper.
3. Evaluate the model using the same datasets and experimental setup described by the authors.
4. Develop and test multiple modifications aimed at improving performance, efficiency, robustness, or training behavior.
5. Compare experimental results against the reproduced baseline.

## Experimental Branches

The `main` branch contains the replicated baseline implementation used as the reference model.

Additional branches contain independent experimental modifications. Each branch explores a different strategy for improving the original model architecture, training process, or feature extraction pipeline.

Experimental results will be compared against the baseline to determine whether the proposed changes provide measurable improvements.

## Datasets

To ensure fair comparison with the original work, all experiments use the same datasets, preprocessing procedures, and evaluation metrics described in the CorrNet+ paper whenever possible.

The primary datasets used include:

* PHOENIX14
* PHOENIX14-T


## Acknowledgments

This project builds upon the original CorrNet+ research and implementation. Full credit for the original model architecture, methodology, and published results belongs to the original authors.
