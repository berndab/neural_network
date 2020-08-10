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

## Analysis

### Feature Elimination
The EIN and name fields were eliminated because they are identity field and are not consider model features. The STATUS and  SPECIAL_CONSIDERATIONS because they have no predictive value. These are  dichotomous feature where 99.99% of the data rows have a STATUS value of 1 and 99.92% of the data rows have a SPECIAL_CONSIDERATIONS	value of N. These features have no predictive value because almost 100% of the data rows have the same value. 


### Static Features

The following feature have a small number of unique categorical values and therefore, do not require binning. These features’ unique categorical value will remain constant when training and testing the neural network. Therefore, they can be encoded once using one-hot encoding and stored in a file.

<table>
   <thead>
      <tr>
         <th>Feature Name</th>
         <th>Unique Catigorical Values</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>AFFILIATION</td>
         <td>6</td>
     </tr>
     <tr>
         <td>USE_CASE</td>
         <td>5</td>
      </tr>
      <tr>
         <td>ORGANIZATION</td>
         <td>4</td>
      </tr>
      <tr>
         <td>INCOME_AMT</td>
         <td>9</td>
      </tr>
   </tbody>
</table>

### Binnable Features

The following features have a significant amount of unique catigorical values and can be binned with differnt binning row count thresholds in order to determine how varying binning thresholds influence the accuracy of the neural network, machine learning model

#### Feature APPLICATION_TYPE
<table>
   <thead>
      <tr>
         <th>Unique Categorial Value</th>
         <th>Data Set Row Count</th>
      </tr>
   </thead>
   <tbody>
	  <tr><td>T3</td><td>27037</td></tr>
	  <tr><td>T4</td><td>1542</td></tr>
	  <tr><td>T6</td><td>1216</td></tr>
	  <tr><td>T5</td><td>1173</td></tr>
	  <tr><td>T19</td><td>1065</td></tr>
	  <tr><td>T8</td><td>737</td></tr>
	  <tr><td>T7</td><td>725</td></tr>
	  <tr><td>T10</td><td>528</td></tr>
	  <tr><td>T9</td><td>156</td></tr>
	  <tr><td>T13</td><td>66</td></tr>
	  <tr><td>T12</td><td>27</td></tr>
	  <tr><td>T2</td><td>16</td></tr>
	  <tr><td>T25</td><td>3</td></tr>
	  <tr><td>T14</td><td>3</td></tr>
	  <tr><td>T15</td><td>2</td></tr>
	  <tr><td>T29</td><td>2</td></tr>
	  <tr><td>17</td><td>1</td></tr>
   </tbody>
</table>

