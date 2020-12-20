# Data Preprocessing Tools

# Importing the libraries
import numpy as np                 #numpy used for creating array
import matplotlib.pyplot as plt    #matplotlib used for creating Charts
import pandas as pd                #pandas used for creating matrix

# Importing the dataset
dataset = pd.read_csv('filename.csv')   #import and put csv file data in the 'dataset' variable . have to upload the file in directory first
X = dataset.iloc[:, :-1].values         # X = matrix of feature . take all the rows with ':' . take all column except last one with ':-1'
y = dataset.iloc[:, -1].values          # y = dependent variable . take all the rows with ':' . take only last column with '-1'
print(X)                                
print(y)                                

# Taking care of missing data                                   #missing data will make error in data training . missing 1% can be removed . otherwise should handle appropriately e/ replace with average of data
from sklearn.impute import SimpleImputer                         # import simpleimputer from sklearn
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # make imputer object to replace the missing value (np.nan) by the mean
imputer.fit(X[:, 1:3])                                          # use the imputer object in the feature column with numerical value
X[:, 1:3] = imputer.transform(X[:, 1:3])                        # example = take the whole row and only the second and third collumn (index 1-2), assumed both is
print(X)

# Encoding categorical data                                     #encode non numerical data / categorical variable into numerical for better processing
# Encoding the Independent Variable                             #turn n different categoral variable into n binary column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') #incorporate OneHotEncoder in ColumnTransformer in CT object, assumed column index 0 (first column) is categorical value .
X = np.array(ct.fit_transform(X))                                                                 #connect and use directly with fit_transform
print(X)
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)                                                                           #simple since dependent variable only one
print(y)

# Splitting the dataset into the Training set and Test set                                       #train set = train Machine learning . Test set = evaluate in future data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)     # matrix of feature, dependent variable, test size= 1-train size, fixing state to get same split
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling                                                                               # scaling all variable in the same scale, prevent domination of one variable
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])                                               #take from the 4th column index:3, since 0,1,2 is assumed to be hot one encoded
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)
