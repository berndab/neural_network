# Philanthropic Grant Analysis

## Overview

Philanthropic organizations apply for funding and receive monetary grants to create and manage charitable projects that fulfill necessary community needs. These projects range from land preservation to healthcare initiatives . The type of organizations that apply for these grants range from trusts to cooperatives. In order to decide whether to issue a grant, a organization that gives grants must evaluate the type of organization that is applying for a grant, the types of projects that this organization has manages in the past, and the outcomes of these projects. The goal of this project is to create a neural network, machine learning model to assist in this evaluation process to make it more efficient and accurate.


This model will be trained and tested on a data set that contains information about past charitable projects that were funded by grants. The data set will contain entries for each project that have features which categorize the type of organization that managed a specific project, the type of project managed, the dollar value of the grant issued, and the outcome of the project. New grant application will contain values for these features which will be inputted into the neural network model to produce an outcome prediction for the project proposed in the application.

 
## Model Development Modular Design Approach

The model development process was designed using a modular approach. The machine learning model development process contains process steps that only need to be executed once during model development, like the feature analysis and the feature elimination steps. If the code for all model development process steps is contained in one Jupyter Notebook then all the process steps must be executed for each model training and testing cycle, even steps that only need to be run once. By segregating, the programing code for process development steps that only run once into their own standalone notebook modules and the data that they generate is stored in a csv file, then only the training and testing notebook module needs to be executed to run a training and testing cycle. Each model training and testing cycle will execute faster as a standalone notebook module without the code for preprocessing steps. Faster model training and testing cycles allow for more cycles can be run  making the process of determining the configuration that produces the most accurate model more efficient.

Furthermore, by using a modular approach, different standalone modules for static features encoding can be develop using different feature encoding algorithms to determine if these other encoding algorithms create a more accurate model. Finally, the code in the training and testing standalone notebook module can be exported to create a python file. This file then can be used to to do automated model testing shorting the overall model development life cycle.  
 

### Model Development Modules

* Notebook Module 1: **grant_analysis.1.preprocess.feature_elimination.ipynb**
   * Performs feature analysis and feature elimination based on the results of the feature analysis step. The output of this process step is stored in a csv file
* Notebook Module 2: **grant_analysis.2.preprocess.one_hot_encoder.ipynb**
   * Encodes the features, using one-hot encoding, that do not need to be bucketed and will remain constant for all model training and testing cycles
* Notebook Module 3: **grant_analysis.3.model_training.ipynb**
   * Uses bucketing threshold variables for these features that enables executing model training and testing cycles with different bucketing configurations 
      * APPLICATION_TYPE
      * CLASSIFICATION
   * Uses a upper range, outlier filter threshold variable for this feature that enables executing model training and testing cycles with different outlier filtering of upper range values
      * ASK_AMT
   * Uses variables to set the neuron count for each hidden layer, If the variable is set to 0 the hidden layer is excluded from the model
   * Uses variables to set the activation function for the input, hidden, and the output neuron layers
   * Scales the ASK_AMT feature using the StandardScaler


## Analysis

### Feature Elimination

The EIN and NAME features were eliminated because they are identity field and are not considered predictive model features. 

The value distributions of the STATUS and SPECIAL_CONSIDERATIONS features were examined. These features have dichotomous values of either 0 or 1 or T and F. For the STATUS feature, 99.99% of the rows in the dataset have a value of 1. For the SPECIAL_CONSIDERATIONS feature, 99.92% of the rows have a value N. Because almost 100% of the rows for these features have the same value, they offer no predictive value to the model and were eliminated from the data set.

### Static Features

The following feature have a small number of unique categorical values which do not require bucketing and can be encoded as is using one-hot encoding.

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

### Bucketing  Features

The features APPLICATION_TYPE  has 17 unique categorical values while the CLASSIFICATION feature has 71 unique categorical values. Both features have unique categorical values with low row count in the data set. Therefore, these features are candidates for bucketing. The model training and testing notebook modules has bucketing row count thresholds variables defined for each feature to enable testing the model with different bucketing configuration to determine how these configurations affect the accuracy of the neural network, machine learning model.

#### Feature APPLICATION_TYPE Catigorical Values Distribution (17)

<table>
   <thead>
      <tr>
         <th>Categorial Value</th>
         <th>Data Set Row Count</th>
         <th>Categorial Value</th>
         <th>Data Set Row Count</th>
      </tr>
   </thead>
   <tbody>
	  <tr><td>T3</td><td>27037</td><td>T13</td><td>66</td></tr>
	  <tr><td>T4</td><td>1542</td><td>T12</td><td>27</td></tr>
	  <tr><td>T6</td><td>1216</td><td>T2</td><td>16</td></tr>
	  <tr><td>T5</td><td>1173</td><td>T25</td><td>3</td></tr>
	  <tr><td>T19</td><td>1065</td><td>T14</td><td>3</td></tr>
	  <tr><td>T8</td><td>737</td><td>T15</td><td>2</td></tr>
	  <tr><td>T7</td><td>725</td><td>T29</td><td>2</td></tr>
	  <tr><td>T10</td><td>528</td><td>T17</td><td>1</td></tr>
	  <tr><td>T9</td><td>156</td><td></td><td><td></td>
   </tbody>
</table>

#### Feature CLASSIFICATION Catigorical Values Distribution (71)

<table>
   <thead>
      <tr>
         <th>Categorial Value</th>
         <th>Row Count</th>
         <th>Categorial Value</th>
         <th>Row Count</th>
     </tr>
   </thead>
   <tbody>
	  <tr><td>C1000</td><td>17326</td><td>C1257</td><td>5</td></tr>
	  <tr><td>C2000</td><td>6074</td><td>C0   </td><td>3</td></tr>
	  <tr><td>C1200</td><td>4837</td><td>C2710</td><td>3</td></tr>
	  <tr><td>C3000</td><td>1918</td><td>C1260</td><td>3</td></tr>
	  <tr><td>C2100</td><td>1883</td><td>C1256</td><td>2</td></tr>
	  <tr><td>C7000</td><td>777</td><td>C1234</td><td>2</td></tr>
	  <tr><td>C1700</td><td>287</td><td>C1246</td><td>2</td></tr>
	  <tr><td>C4000</td><td>194</td><td>C1267</td><td>2</td></tr>
	  <tr><td>C5000</td><td>116</td><td>C3200</td><td>2</td></tr>
	  <tr><td>C1270</td><td>114</td><td>C2570</td><td>1</td></tr>
	  <tr><td>C2700</td><td>104</td><td>C1900</td><td>1</td></tr>
	  <tr><td>C2800</td><td>95</td><td>C3700</td><td>1</td></tr>
	  <tr><td>C7100</td><td>75</td><td>C8210</td><td>1</td></tr>
	  <tr><td>C1300</td><td>58</td><td>C6100</td><td>1</td></tr>
	  <tr><td>C1280</td><td>50</td><td>C2150</td><td>1</td></tr>
	  <tr><td>C1230</td><td>36</td><td>C4200</td><td>1</td></tr>
	  <tr><td>C1400</td><td>34</td><td>C2170</td><td>1</td></tr>
	  <tr><td>C2300</td><td>32</td><td>C1236</td><td>1</td></tr>
	  <tr><td>C7200</td><td>32</td><td>C4120</td><td>1</td></tr>
	  <tr><td>C1240</td><td>30</td><td>C2561</td><td>1</td></tr>
	  <tr><td>C8000</td><td>20</td><td>C1820</td><td>1</td></tr>
	  <tr><td>C7120</td><td>18</td><td>C1728</td><td>1</td></tr>
	  <tr><td>C1500</td><td>16</td><td>C2600</td><td>1</td></tr>
	  <tr><td>C6000</td><td>15</td><td>C4500</td><td>1</td></tr>
	  <tr><td>C1800</td><td>15</td><td>C1283</td><td>1</td></tr>
	  <tr><td>C1250</td><td>14</td><td>C1248</td><td>1</td></tr>
	  <tr><td>C8200</td><td>11</td><td>C5200</td><td>1</td</tr>
	  <tr><td>C1238</td><td>10</td><td>C2190</td><td>1</td</tr>
	  <tr><td>C1278</td><td>10</td><td>C2380</td><td>1</td</tr>
	  <tr><td>C1237</td><td>9</td><td>C1580</td><td>1</td></tr>
	  <tr><td>C1235</td><td>9</td><td>C1370</td><td>1</td></tr>
	  <tr><td>C7210</td><td>7</td><td>C1570</td><td>1</td></tr>
	  <tr><td>C1720</td><td>6</td><td>C1245</td><td>1</td></tr>
	  <tr><td>C2400</td><td>6</td><td>C2500</td><td>1</td></tr>
	  <tr><td>C4100</td><td>6</td><td>C1732</td><td>1</td></tr>
	  <tr><td>C1600</td><td>5</td><td></td><td></td></tr>
   </tbody>
</table>

### Feature with Potential Outliers

The feature ASK_AMT has a very wide distribution of values, 75% of all values are exactly $5000 dollars. Yet the mean is $2,769,198.68, the standard deviation is $87,130,452.44, and the maximum value is $8,597,806,340.00. This indicate that the 25% of values in the top quartile are values exceeding higher than $50000 which are skewing the distribution of values for this feature. The model training and testing notebook module have an outlier exclusion variable that filters out any ASK_AMT value equal to or grater then the variable value. This enables the execution of training and testing cycles with ASK_AMT outliers filtered out of the training and testing data set to determine how removing outliers  affects accuracy of the model. 

<table>
   <thead>
      <tr>
         <th>Statistical Measure</th>
         <th>Values</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>Mean</td>
         <td>$2,769,198.68</td>
     </tr>
     <tr>
         <td>Standard Deviation</td>
         <td>$87,130,452.44</td>
      </tr>
      <tr>
         <td>Min Value</td>
         <td>$5,000.00</td>
      </tr>
      <tr>
         <td>25%</td>
         <td>$5,000.00</td>
      </tr>
      <tr>
         <td>50%</td>
         <td>$5,000.00</td>
      </tr>
      <tr>
         <td>75%</td>
         <td>$5,000.00</td>
      </tr>
      <tr>
         <td>Max</td>
         <td>$8,597,806,340.00</td>
      </tr>
   </tbody>
</table>


## Neural Network Model Predictive Performance Analysis

The model training testing program used configurable variables to set the neural network model training parameters, to set the threshold row count value for features bucketing , and to to set the outlier filter threshold for ASK_AMT feature values

These variable were used to set the following model testing parameter
* Model Parameters
  * Hidden layer 1 neuron count
  * Hidden layer 1 activation function
  * Hidden layer 2 neuron count
  * Hidden layer 2 activation function
  * Hidden layer 3 neuron count
  * Hidden layer 3 activation function
  * Model training epochs
* Feature Bucketing Parameters
  * Bucketing row count threshold for feature APPLICATION_TYPE
  * Bucketing row count threshold for feature CLASSIFICATION
* Outlier filter parameters for feature ASK_AMT

### Feature Bucketing

The model performed best when with a low bucketing threshold for the two features APPLICATION_TYPE and CLASSIFICATION where the value only eliminated categorical values for these features that had row count in the single digits. High bucketing threshold amount that removed categorical values with row count above 100 significantly reduced the model’s accuracy.

### ASK_AMOUNT Feature Outlier Filtering

Any attempt to filter even the most extream outlier values lowered the accuracy performance of the machine learning model. This may be because these upper 25% or the ask amout values offer significant predictive value to the model do to their high dollar amounts

### Neural Network Layer Activation Functions

The most accurate model was the version that used the ReLU activation for the import and all hidden layers with the output layer using the sigmoid activation function. The tabh activation function used on in the input and hidden layers produced almost the same accuracy as using the ReLU function. All other activation function used produced a trained model with less accuracy. 

### Adding Hidden Layers and Hidden Layer Neurons

Adding more neurons to the hidden layer and more hidden layers continued to increase the models accuracy until it reached a pick trained accuracy of about 0.445 - 0.4470 using
* hidden layer 1:  75 neurons
* hidden layer 2: 150 neurons
* hidden layr  3:  10 neurons

Any additional neurons cause the model to become overfitted which was indicated by a higher training accuracy and a lower testing accuracy.

### Epochs

The training accuracy and testing accuracy of the model peaks when using 200 epoch to train the model. The model started to become overfit when using more than 200 epoch which was indicated by a higher training accuracy and a lower testing accuracy. 

### Conclusion

The peak model training accuracy that was achieved was 0.75 in some epoch executions while the peak testing accuracy that was achieved by varying the model training parameters of the model was 0.733. The accuracy of the model seems to be limited because the available features to train the model had weak predictive value. 

Because of the week predictive value of the data features,  a random forest model may produce better predictive accuracy since random forest machine learning model is a weak learning model that creates many decision trees based on random subsets of the training data. The collection of randomly generated decision trees when combined may produce predictive model with a higher accuracy than the neural network model that was tested. 





