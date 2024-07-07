# Boston house price prediction

## Installation Guide
1. Clone or Fork the Project
2. Create a Virtual Enviroment
3. go to same virtual enviroment and write below cmd
4. pip install -r requirements.txt


### 1. Project Description
#### A. Problem Statement

Thousands of houses are sold everyday. There are some questions every buyer asks himself like: What is the actual price that this house deserves? Am I paying a fair price?

#### B. Tools and Libraries
Tools<br><br>
a.Python<br>
b.Jupyter Notebook<br>
c. Flask<br>
d. HTML<br>
e. Render<br>
f. GitHub

Libraries<br><br>
a.Pandas<br>
b.Scikit Learn<br>
c.Numpy<br>
d.Seaborn<br>
e.Matpoltlib<br>

### 2. Data Collection
For this project we used the data that is available on sklearn itself(from sklearn.datasets import load_boston)
There are 13 columns and 506 Rows. These are the major point about the data set.<br>
CRIM: per capita crime rate by town<br>
ZN: proportion of residential land zoned for lots over 25,000 sq.ft.<br>
INDUS: proportion of non-retail business acres per town<br>
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)<br>
NOX: nitric oxides concentration (parts per 10 million)<br>
RM: average number of rooms per dwelling<br>
AGE: proportion of owner-occupied units built prior to 1940<br>
DIS: weighted distances to five Boston employment centres<br>
RAD: index of accessibility to radial highways<br>
TAX: full-value property-tax rate per 10,000usd<br>
PTRATIO: pupil-teacher ratio by town<br>
B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town<br>
LSTAT: % lower status of the population<br>

Target Column is Price which is specified as
MEDV: Median value of owner-occupied homes in $1000s<br>
If MEDV is 24.0 for a particular area (or observation in the dataset), it means that the median value of all the owner-occupied homes in that area is $24,000. This value is derived from actual housing prices and is intended to give a single, representative value for the home prices in that area.

### 3. EDA
#### A.Data Cleaning
We have 13 columns and no column contain any null values<br>
All columns are numerical also

#### B. Feature Engineering
No outliers are present in the data

#### C. Data Normalization
Normalization (min-max Normalization)<br>
In this approach we scale down the feature in between 0 to 1

we have numerical column where we can apply min-max Normalization.<br>

### 4. Choosing Best ML Model
List of the model that we can use for our problem<br>
a. Random Forest Regression model<br>

### 5. Model Creation
So,using a Random Forest Regresssion we got good accuracy as R2 score is close to 1(0.97), we can Hyperparameter tuning for best accuracy.

Algorithm that can be used for Hyperparameter tuning are :-

a. GridSearchCV<br>
b. RandomizedSearchCV<br>

Main parameters used by Random Forest Regression Algorithm are :-
a. n_estimators -> Number of trees in the forest.<br>
b. criterion -> 'mse' (mean squared error, default) or 'mae' (mean absolute error).<br>
c. max_depth -> The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain fewer than min_samples_split samples.<br>
d. min_samples_split -> The minimum number of samples required to split an internal node. Default is 2.<br>
e. min_samples_leaf -> The minimum number of samples required to be at a leaf node. Default is 1.<br>
f. max_leaf_nodes -> Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None (default), then unlimited number of leaf nodes.<br>
g. max_features -> The number of features to consider when looking for the best split. `None` (default): Consider all features.


### 6. Model Deployment
After creating model ,we integrate that model with beautiful UI. for the UI part we used HTML and Flask. We have added extra validation check also so that user doesn't enter Incorrect data. Then the model is deployed on render

### 7. Model Conclusion

Model predict 0.62 accurately on test data(R2 Score).

### 8. Project Innovation
a. Easy to use<br>
b. Open source<br>
c. Best accuracy<br>
d. GUI Based Application

### 9. Limitation And Next Step
Limitation are :-<br>
a. Mobile Application<br>
b. Accuracy can be improved more<br>
d. Feature is limited

Next Step are :-<br>
a. we can work on mobile application<br>

## Deployable Link
https://machine-learning-practical-01-boston.onrender.com/predict