# Philanthropic Grant Analysis

## Overview

Philanthropic organizations apply for funding and receive monetary grants to create and manage charitable projects that fulfill necessary community needs. These projects range from land preservation to healthcare initiatives. The type of organizations that apply for these grants range from trusts to cooperatives. In order to decide whether to issue a grant, a grant issuing organization must evaluate the type of organization that is applying for a grant, the types of projects that this organization has manages in the past, the outcomes of these projects, and the type of the current project that they are seeking a grant for. In order to create a more efficient and more accurate grant evaluation process, this project will develop a neural network, machine learning model to assist in the evaluation process.


This model will be trained and tested on a data set that contains information about past charitable projects. The data set contains entries for individual projects that include features that categorize the type of organization that managed the project, the type of project managed, the grant value received for the project, and the outcome of the project. Then, when new grant application are received, the features that the model uses will be provided by the grant application and will be inputted into the neural network model to produce an outcome prediction for the proposed project.

 
## Model Development Modular Design Approach

The model development process was designed using a modular approach. The machine learning model development process has several process steps. Some of these process steps only need to be executed once during model development, such as the feature analysis and feature elimination steps. The typical approach for machine learning model development is to create on Jupyter Notebook that contains the code for all model development process steps. However, in order to execute a model training and testing cycle, all the process steps contained in the one notebook must be executed, even steps that only needed to be run once. Segregating the programing code for single-execution, process development steps into their own standalone notebook modules, enables the segregation of the model training and testing code into its own standalone notebook module. To run training and testing cycles with different model parameters, only the standalone training and testing notebook needs to be executed. This enables faster execution of training and testing cycles which intern allow for more cycles to be executed in the limited time available for training and testing phase of the model development lifecycle. Having the capability to execute more training and testing cycles decreases the time needed to determine the configuration parameters that produce the most accurate model.

Furthermore, by using this modular approach, multiple modules for the static features encoding process step can be created that use different encoding algorithms, such as the LabelEncoder algorithm. The accuracy of the model using the different encoding methods can be compared to determine which encoding algorithm produces the most accurate model. Finally, the code in the training and testing standalone notebook module can be exported to a python file which intern can be modified to automated the model testing process which would make the process task of  determining the most accurate model configuration more efficient.  

## Technical Summary

### Python Modules and Classes

|Module                             |Class                              |Module                             |Class                              |
|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|
|**Environment**                    |                                   |**Feature Encoding Algorithms**    |                                   |
|Anaconda                           |                                   |sklearn.preprocessing              |OneHotEncoder                      |
|**Application**                    |                                   |**Feature Scaling Algorithms**     |                                   |
|Jupyter Notebook                   |                                   |sklearn.preprocessing              |StandardScaler                     |
|**Data Analysis**                  |                                   |sklearn.preprocessing              |MinMaxScaler                       |
|pandas                             |                                   |**Utils**                          |                                   |
|numpy                              |                                   |sklearn.model_selection            |train_test_split                   |
|**Model Components**               |                                   |                                   |                                   |
|tensorflow.keras.models            |Sequential                         |                                   |                                   |
|tensorflow.keras.                  |Dense                              |                                   |                                   |
|tensorflow.keras.callbacks         |ModelCheckpoint                    |                                   |                                   |

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

The EIN and NAME data features were eliminated from the data set because these fields contain identity information and are not considered predictive model features. 

The value distributions of the STATUS and SPECIAL_CONSIDERATIONS features were examined. These features have dichotomous values of either 0 and 1 or T and F. For the STATUS feature, 99.99% of the rows in the dataset have a value of 1. For the SPECIAL_CONSIDERATIONS feature, 99.92% of the rows have a value N. Because almost 100% of the rows for these features have the same value, they offer no predictive value to the model and were eliminated from the data set.

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

The APPLICATION_TYPE feature has 17 unique categorical values and the CLASSIFICATION feature has 71 unique categorical values. Both features have categorical values with low row count in the data set. Therefore, these features are candidates for bucketing. The model training and testing notebook module has a variable for each feature which sets the row count threshold for bucketing. Any category with a row count less than of equal to the threshold variable value will be bucketed into the "Other" category.  

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

The ASK_AMT feature has a very wide distribution of values. 75% of all values are exactly $5000 dollars. Yet the mean is $2,769,198.68, the standard deviation is $87,130,452.44, and the maximum value is $8,597,806,340.00. This indicate that values in the upper quartile far exceed $50000 and skew the distribution of values for this feature. The model training and testing notebook module have an outlier filters threshold variable for this feature. Any value greater than or equal to the outlier filter threshold value will be removed from the data set. The allows the model to be tested different level out outlier filtering to see if removing outliers from the data set enable the model to be more accurate. 

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


## Model Predictive Performance Analysis

The model training and testing notebook module uses variables to configuration the data used to train and test the neural network model. It also uses variables to set neural network model configuration parameters.

The following variables are used to configure the data set used for training and testing the neural network model
* Feature Bucketing Parameters
  * Bucketing row count threshold for feature APPLICATION_TYPE
  * Bucketing row count threshold for feature CLASSIFICATION
* Outlier filter parameters for feature ASK_AMT

The following variables are used to set the various neural network model configuration parameters
* Model Parameters
  * Hidden layer 1 neuron count
  * Hidden layer 1 activation function
  * Hidden layer 2 neuron count
  * Hidden layer 2 activation function
  * Hidden layer 3 neuron count
  * Hidden layer 3 activation function
  * Model training epochs

### Feature Bucketing

The model performed best when using a low bucketing threshold for the features APPLICATION_TYPE and CLASSIFICATION which bucket categories that had row counts less than or equal to 10. Using larger threshold values to bucket categorical values significantly reduced the model’s accuracy.

### ASK_AMOUNT Feature Outlier Filtering

Any attempt to filter even the most extreme outlier values lowered the accuracy of the machine learning model. This may be because these upper quartile values have significant predictive value to the model do to their high dollar amounts.

### Neural Network Layer Activation Functions

The most accurate model used the ReLU activation for the import and hidden layers configurations and the sigmoid activation function for the output layer. The model accuracy when using the tanh activation function was used in the input and hidden layers configuration was slightly less accurate then when the ReLU function . All other activation function used for the input and hidden layers produced a model that was much less accurate then the model using ReLu and tanh. 

### Adding Hidden Layers and Hidden Layer Neurons

Adding more neurons to the hidden layer and more hidden layers continued to increase the models accuracy until it reached a peak trained accuracy of about 0.7461 using
* hidden layer 1:  75 neurons
* hidden layer 2: 150 neurons
* hidden layer 3:  10 neurons

Model testing accuracy varied between 0.72 and 0.73

Adding additional neurons cause the model to become overfitted which was indicated by a increasing training accuracy trend and a decreasing testing trend.

### Epochs

The training and testing accuracy of the model peaked at about 200 epochs used to train the model. As epochs were increased, the model became overfitted which was indicated by increasing training accuracy trend and a decreasing testing accuracy trend.


### Conclusion

The peak model training accuracy that was achieved was 0.7461 while the peak testing accuracy that was achieved was 0.733. The accuracy of the model seems to be limited because the available features to train the model have weak predictive value. 

Because of the week predictive value of the data features,  a random forest model may produce better predictive accuracy since random forest machine learning model is a weak learning model that creates many decision trees based on random subsets of the training data. The collection of randomly generated decision trees, when combined, may produce a model with more accuracy.





