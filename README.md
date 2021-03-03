# Macro-forecasting

This repository contains implementation of several forecasting models following Coulombe et al (2019) "How is Machine Learning Useful for Macroeconomic Forecasting?". The files included implement autoregressive models augmented with various machine learning features and cross-validation to choose model hyperparameters. The models considered here are 

1. Benchmark Autoregressive model
2. AR with Ridge regularization
3. Random Forest AR model
4. AR with Kernel Ridge regression
5. Support Vector Regression AR

For each model, hyperparameters are chosen via two types of cross validation techniques: K-fold and Psuedo-out-of sample. Each Python file is a function that implements one of the above models, where the user can specify the cross-validation technique of their choosing. The function chooses the optimal hyperparameters using cross-validation and the optimal number of lags for the AR process. The output contains predictions and the out-of-sample root mean squared prediction error. 

The dataset used is FRED-MD maintained by the Federal Reserve Bank of St. Louis. I follow McCracken and Ng (2016) while preprocessing the data to get stationary outcomes through transformation of the variables. The Jupyter notebook provides more details of data preprocessing, implementation of each model and the results of the forecasting exercise. 
