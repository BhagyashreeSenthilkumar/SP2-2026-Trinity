# SP2-2026-Trinity
# Sona Power Predict – 2026

## College Name

Sona College of Technology

## Team Name

Trinity

# Team Members

* Akshaya N – Year 3, Computer Science and Engineering (CSE)
* Bhagyashree S – Year 3, Computer Science and Engineering (CSE)
* Bhavadharini K – Year 3, Computer Science and Engineering (CSE)

## Overview

This project predicts IPL PowerPlay (Overs 1–6) scores using a hybrid machine learning model built with heuristic weighted blending and Ridge Regression.

The model is trained using historical IPL ball-by-ball data and incorporates:

* Batting team strength
* Bowling team defensive performance
* Venue scoring trends
* Head-to-head statistics
* Seasonal scoring inflation trends
* Player-level adjustments

The approach combines statistical modelling with recency-based feature engineering to improve prediction accuracy.

# Libraries Used in the Model

Based on the `mymodelfile.py` submission, the following Python libraries are utilized:

* pandas
  Used for DataFrame manipulation, grouping, preprocessing, and handling historical IPL datasets.

* numpy
  Used for numerical operations, vectorized computations, and array handling.

* scikit-learn
  Used for implementing the Ridge Regression model.

* re
  Used for text cleaning and venue normalization.

* datetime
  Used for year extraction and trend calculations.

# Model Highlights
* Four-tier recency weighted feature blending
* Exponential Weighted Moving Average (EWMA)
* Ridge Regression hybrid prediction
* Venue normalization and team mapping
* Player-level batting and bowling adjustments
* Multi-layer fallback prediction pipeline
* Fully vectorized prediction architecture
  
# License

This project is licensed under the MIT License.

# Competition

SONA POWER PREDICT 2026
