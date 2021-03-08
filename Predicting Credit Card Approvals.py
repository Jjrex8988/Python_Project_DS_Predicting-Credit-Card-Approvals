# Title     : Predicting Credit Card Approvals
# Objective : Build credit card predictor
# Created by: Jjrex8988
# Created on: 8/3/2021

## 1. Credit card applications
# Import pandas
import pandas as pd

# Load dataset
cc_apps = pd.read_csv("cc_approvals.data.txt", header=None, index_col=False)

# Rename columns, The features of this dataset have been anonymized to protect the privacy
# Rename according http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html
cc_apps.columns = ["Gender", "Age", "Debt", "Married", "BankCustomer",
                   "EducationLevel", "Ethnicity", "YearsEmployed",
                   "PriorDefault", "Employed", "CreditScore",
                   "DriversLicense", "Citizen", "ZipCode",
                   "Income", "ApprovalStatus"]
# Inspect data
print(cc_apps.head())
print("-"*38)
print(cc_apps.columns)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## 2. Inspecting the application
# Print summary statistics
print(cc_apps.describe())
print("\n")

# Print DataFrame information
print(cc_apps.info())
print("\n")

# Inspect missing values in the dataset
print(cc_apps.tail(18))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## 3. Handling the missing values (part i)
# Import numpy
import numpy as np


# Replace the '?'s with NaN
cc_apps = cc_apps.replace("?", np.NaN)

# Inspect the missing values again
print(cc_apps.tail(18))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## 4. Handling the missing values (part ii)
# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace=True)

# Count the number of NaNs in the dataset to verify
print(cc_apps.isnull().sum())
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## 5. Handling the missing values (part iii)
# Iterate over each column of cc_apps
for col in cc_apps.columns:
    # Check if the column is of object type
    if cc_apps[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
print(cc_apps.isnull().sum())
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## 6. Preprocessing the data (part i)
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in cc_apps.columns:
    # Compare if the dtype is object
    if cc_apps[col].dtypes == 'object':
    # Use LabelEncoder to do the numeric transformation
        cc_apps[col] = le.fit_transform(cc_apps[col])
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## 7. Splitting the dataset into train and test sets
# Import train_test_split
from sklearn.model_selection import train_test_split

# Drop the features 11 and 13 and convert the DataFrame to a NumPy array
cc_apps = cc_apps.drop(['DriversLicense', 'ZipCode'], axis=1)
cc_apps = cc_apps.values

# Segregate features and labels into separate variables
X, y = cc_apps[:, 0:13], cc_apps[:, 13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## 8. Preprocessing the data (part ii)
# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## 9. Fitting a logistic regression model to the train set
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
# logreg.fit(X_train, y_train)
logreg.fit(rescaledX_train, y_train)

print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## 10. Making predictions and evaluating performance
# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

# Print the confusion matrix of the logreg model
print(confusion_matrix(y_test, y_pred))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## 11. Grid searching and making the model perform better
from sklearn.model_selection import GridSearchCV
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit data to grid_model
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#