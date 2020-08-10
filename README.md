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

The features APPLICATION_TYPE and CLASSIFICATION have 17 and 71 unique categorical values respectively and have categorical values with low row counts. Therefore, these features are candidates for binning. Binning row count thresholds variables are created in the model training and testing workbook to enable vary the number of feature categories that are updated to the “Other” and to determine how the varying the binning threshold influences the accuracy of the neural network, machine learning model.

#### Feature APPLICATION_TYPE
<table>
   <thead>
      <tr>
         <th>Unique Categorial Value (17)</th>
         <th>Data Set Row Count</th>
         <th>Unique Categorial Value (17)</th>
         <th>Data Set Row Count</th>
      </tr>
   </thead>
   <tbody>
	  <tr><td>T3</td><td>27037</td><td>T9</td><td>156</td></tr>
	  <tr><td>T4</td><td>1542</td><td>T13</td><td>66</td</tr>
	  <tr><td>T6</td><td>1216</td><td>T12</td><td>27</td></tr>
	  <tr><td>T5</td><td>1173</td><td>T2</td><td>16</td></tr>
	  <tr><td>T19</td><td>1065</td><td>T25</td><td>3</td></tr>
	  <tr><td>T8</td><td>737</td><td>T14</td><td>3</td></tr>
	  <tr><td>T7</td><td>725</td><td>T15</td><td>2</td></tr>
	  <tr><td>T10</td><td>528</td><td>T29</td><td>2</td></tr>
	  <tr><td></td><td><td>17</td><td>1</td></td>
   </tbody>
</table>

#### Feature CLASSIFICATION


<table>
   <thead>
      <tr>
         <th>Unique Classifier</th>
         <th>Row Count</th>
      </tr>
   </thead>
   <tbody>
	  <tr><td>C1000</td><td>17326</td></tr>
	  <tr><td>C2000</td><td>6074</td></tr>
	  <tr><td>C1200</td><td>4837</td></tr>
	  <tr><td>C3000</td><td>1918</td></tr>
	  <tr><td>C2100</td><td>1883</td></tr>
	  <tr><td>C7000</td><td>777</td></tr>
	  <tr><td>C1700</td><td>287</td></tr>
	  <tr><td>C4000</td><td>194</td></tr>
	  <tr><td>C5000</td><td>116</td></tr>
	  <tr><td>C1270</td><td>114</td></tr>
	  <tr><td>C2700</td><td>104</td></tr>
	  <tr><td>C2800</td><td>95</td></tr>
	  <tr><td>C7100</td><td>75</td></tr>
	  <tr><td>C1300</td><td>58</td></tr>
	  <tr><td>C1280</td><td>50</td></tr>
	  <tr><td>C1230</td><td>36</td></tr>
	  <tr><td>C1400</td><td>34</td></tr>
	  <tr><td>C2300</td><td>32</td></tr>
	  <tr><td>C7200</td><td>32</td></tr>
	  <tr><td>C1240</td><td>30</td></tr>
	  <tr><td>C8000</td><td>20</td></tr>
	  <tr><td>C7120</td><td>18</td></tr>
	  <tr><td>C1500</td><td>16</td></tr>
	  <tr><td>C6000</td><td>15</td></tr>
	  <tr><td>C1800</td><td>15</td></tr>
	  <tr><td>C1250</td><td>14</td></tr>
	  <tr><td>C8200</td><td>11</td></tr>
	  <tr><td>C1238</td><td>10</td></tr>
	  <tr><td>C1278</td><td>10</td></tr>
	  <tr><td>C1237</td><td>9</td></tr>
	  <tr><td>C1235</td><td>9</td></tr>
	  <tr><td>C7210</td><td>7</td></tr>
	  <tr><td>C1720</td><td>6</td></tr>
	  <tr><td>C2400</td><td>6</td></tr>
	  <tr><td>C4100</td><td>6</td></tr>
	  <tr><td>C1600</td><td>5</td></tr>
	  <tr><td>C1257</td><td>5</td></tr>
	  <tr><td>C0   </td><td>3</td></tr>
	  <tr><td>C2710</td><td>3</td></tr>
	  <tr><td>C1260</td><td>3</td></tr>
	  <tr><td>C1256</td><td>2</td></tr>
	  <tr><td>C1234</td><td>2</td></tr>
	  <tr><td>C1246</td><td>2</td></tr>
	  <tr><td>C1267</td><td>2</td></tr>
	  <tr><td>C3200</td><td>2</td></tr>
	  <tr><td>C2570</td><td>1</td></tr>
	  <tr><td>C1900</td><td>1</td></tr>
	  <tr><td>C3700</td><td>1</td></tr>
	  <tr><td>C8210</td><td>1</td></tr>
	  <tr><td>C6100</td><td>1</td></tr>
	  <tr><td>C2150</td><td>1</td></tr>
	  <tr><td>C4200</td><td>1</td></tr>
	  <tr><td>C2170</td><td>1</td></tr>
	  <tr><td>C1236</td><td>1</td></tr>
	  <tr><td>C4120</td><td>1</td></tr>
	  <tr><td>C2561</td><td>1</td></tr>
	  <tr><td>C1820</td><td>1</td></tr>
	  <tr><td>C1728</td><td>1</td></tr>
	  <tr><td>C2600</td><td>1</td></tr>
	  <tr><td>C4500</td><td>1</td></tr>
	  <tr><td>C1283</td><td>1</td></tr>
	  <tr><td>C1248</td><td>1</td></tr>
	  <tr><td>C5200</td><td>1</td></tr>
	  <tr><td>C2190</td><td>1</td></tr>
	  <tr><td>C2380</td><td>1</td></tr>
	  <tr><td>C1580</td><td>1</td></tr>
	  <tr><td>C1370</td><td>1</td></tr>
	  <tr><td>C1570</td><td>1</td></tr>
	  <tr><td>C1245</td><td>1</td></tr>
	  <tr><td>C2500</td><td>1</td></tr>
	  <tr><td>C1732</td><td>1</td></tr>
   </tbody>
</table>

## Model Training Analysis
