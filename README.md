# Philithropic Grant Analysis

## Overview

Philanthropic organizations apply for funding and receive monetary grants to create and manage charitable projects that fulfill necessary community needs. These projects range from land and nature preservation to healthcare initiatives that improve community health. The type of organization that apply for these grants to fund their philanthropic project range from trusts to cooperatives. 
The goal of this project is to create a neural network based, machine learning model to predict the outcome of future projects funded by charitable grants. This model with be based on a data set that contains information on philanthropic organization that have managed charitable projects, the types of projects that they managed, the size of the grant issued to fund these projects, and the outcome of these projects. This model will use features that describe the type of philanthropic organization applying for a grant and features that describe the type of project that grant will fund to predict the probable outcome of the project.

## Modular Design Approach

Three Jupyter Notebooks were created

1. Performs feature analysis and feature elimination
2. Encodes the static features that would remain constant over all training runs
3. Perform flexible model training that uses parameters to vary
    * The binning threshold of the feature APPLICATION_TYPE and CLASSIFICATION
    * The threshold for eliminating upper end outliers for the target variable ASK_AMT
    * The number of neurons used by each hidden layers
    * The activation function for each hidden layer
    
 The advantage of this modular approach is that feature analysis and static feature encoding process do not have to be run each time a training run is executed which allows for faster training run and more runs to be executed. The feature analysis and elimination process and the static feature encoding process only has to be executed once because the output data of those processed remain static over all model training executions.  In addition, having the static features encoded in a standalone notebook allows the for different feature encoding algorithms to be used to encode feature data, such as label encoding, to test if the model using data encoded by a different encoding algorithms improved the accuracy of the model.  





