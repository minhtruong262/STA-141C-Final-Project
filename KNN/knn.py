#---------------------------------------------
# Import necessary packages
#---------------------------------------------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

#----------------------------------------------------
# Make a function that computes success rate with KNN
#----------------------------------------------------

def knn_rate(xtrain,ytrain,xtest,ytest,k):
    """This function takes in a training dataset of features,
       a training dataset of labels, a testing dataset of features,
       a testing dataset of labels, and a specified number of 
       neighbors. It will run KNN algorithms on the training set,
       predict the testing set, compare the results with the true
       labels, and return success rate."""
    prediction = []
    count = 0
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtrain,ytrain)
    accu_rate = knn.score(xtest,ytest)
    return accu_rate
        
#---------------------------------------------
# Spambase Data
#--------------------------------------------- 

# Import spambase.data
df = pd.read_csv("spambase.data", header = None)

# Split the features and the labels
X = df.iloc[:,0:57]
y = df.iloc[:,57]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Reset the indices
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Run the knn_rate function on data using 1 to 25 neighbors
accu_list = [knn_rate(X_train,y_train,X_test,y_test,i) for i in range(1,26)]

# Make a plot
plt.close()
plt.rcParams["figure.figsize"] = [10, 8]
plt.plot(list(range(1,26)),accu_list,'-o')
plt.xticks(np.arange(0, 26, 1.0))
plt.title('Number of Neighbors vs. Success Rate for Spambase Dataset')
plt.xlabel('Number of Neighbors')
plt.ylabel('Success Rate')
plt.grid()
plt.savefig('knn_spam.png')

# Time the KNN algorithm
start = time.time()
knn_rate(X_train,y_train,X_test,y_test,1)
end = time.time()
print("It takes " +  str(end - start)  + " seconds to run KNN algorithm on spambase dataset with 1 neighbor.")

#---------------------------------------------
# Breast Cancer Data
#---------------------------------------------

# Import wdbc.data
df2 = pd.read_csv("wdbc.data", header = None)

# Split the features and the labels
X2 = df2.iloc[:,2:32]
y2 = df2.iloc[:,1]

# Split dataset into training and testing sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=5)

# Reset the indices
X_train2 = X_train2.reset_index(drop=True)
X_test2 = X_test2.reset_index(drop=True)
y_train2 = y_train2.reset_index(drop=True)
y_test2 = y_test2.reset_index(drop=True)

# Run the knn_rate function on data using 1 to 25 neighbors
accu_list2 = [knn_rate(X_train2,y_train2,X_test2,y_test2,i) for i in range(1,26)]

# Make a plot
plt.close()
plt.rcParams["figure.figsize"] = [10, 8]
plt.plot(list(range(1,26)),accu_list2,'-o')
plt.xticks(np.arange(0, 26, 1.0))
plt.title('Number of Neighbors vs. Success Rate for Breast Cancer Dataset')
plt.xlabel('Number of Neighbors')
plt.ylabel('Success Rate')
plt.grid()
plt.savefig('knn_breast.png')

# Time the KNN algorithm
start = time.time()
knn_rate(X_train2,y_train2,X_test2,y_test2,11)
end = time.time()
print("It takes " +  str(end - start)  + " seconds to run KNN algorithm on breast cancer dataset with 11 neighbors.")

#-----------------------------------------------------
# Adult Dataset
#-----------------------------------------------------


# Import adult.data
df3 = pd.read_csv("adult.data", header = None)

# Convert the categorical variables to numerical
categorical_features = [1,3,5,6,7,8,9,13]
le = LabelEncoder()
for i in range(0,len(categorical_features )):
    new = le.fit_transform(df3[categorical_features[i]])
    df3[categorical_features[i]] = new

# Split the features and the labels
X3 = df3.iloc[:,0:14]
y3 = df3.iloc[:,14]

# Split dataset into training and testing sets
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=5)

# Reset indices
X_train3 = X_train3.reset_index(drop=True)
X_test3 = X_test3.reset_index(drop=True)
y_train3 = y_train3.reset_index(drop=True)
y_test3 = y_test3.reset_index(drop=True)

# Run the knn_rate function on data using 1 to 25 neighbors
accu_list3 = [knn_rate(X_train3,y_train3,X_test3,y_test3,i) for i in range(1,26)]

# Make a plot
plt.close()
plt.rcParams["figure.figsize"] = [10, 8]
plt.plot(list(range(1,26)),accu_list3,'-o')
plt.xticks(np.arange(0, 26, 1.0))
plt.title('Number of Neighbors vs. Success Rate for Adult Dataset')
plt.xlabel('Number of Neighbors')
plt.ylabel('Success Rate')
plt.grid()
plt.savefig('knn_adult.png')

# Time the KNN algorithm
start = time.time()
knn_rate(X_train3,y_train3,X_test3,y_test3,22)
end = time.time()
print("It takes " +  str(end - start)  + " seconds to run KNN algorithm on adult dataset with 22 neighbors.")

#------------------------------------------------------------
# Madelon Dataset
#------------------------------------------------------------

# Read in features/labels training sets and features/labels testing set
x_train_m = pd.read_csv("madelon_train.data",header = None,sep = '\s+')
y_train_m = pd.read_csv("madelon_train.labels",header = None)
x_test_m = pd.read_csv("madelon_valid.data",header = None,sep = '\s+')
y_test_m = pd.read_csv("madelon_valid.labels",header = None)

# Run the knn_rate function on data using 1 to 25 neighbors
accu_list4 = [knn_rate(x_train_m,y_train_m,x_test_m,y_test_m,i) for i in range(1,26)]

# Make a plot
plt.close()
plt.rcParams["figure.figsize"] = [10, 8]
plt.plot(list(range(1,26)),accu_list4,'-o')
plt.xticks(np.arange(0, 26, 1.0))
plt.title('Number of Neighbors vs. Success Rate for Madelon Dataset')
plt.xlabel('Number of Neighbors')
plt.ylabel('Success Rate')
plt.grid()
plt.savefig('knn_madelon.png')

# Time the KNN algorithm
start = time.time()
knn_rate(X_train,y_train,X_test,y_test,21)
end = time.time()
print("It takes " +  str(end - start)  + " seconds to run KNN algorithm on madelon dataset with 21 neighbors.")


#---------------------------------------------
# Parkinson Data
#---------------------------------------------

# Import spambase.data
df_parkinson = pd.read_csv("pd_speech_features.csv")

# Split the features and the labels
X_p = df_parkinson.iloc[:,1:754]
y_p = df_parkinson.iloc[:,754]

# Split dataset into training and testing sets
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2, random_state=5)

# Reset the indices
X_train_p = X_train_p.reset_index(drop=True)
X_test_p = X_test_p.reset_index(drop=True)
y_train_p = y_train_p.reset_index(drop=True)
y_test_p = y_test_p.reset_index(drop=True)

# Run the knn_rate function on data using 1 to 25 neighbors
accu_list5 = [knn_rate(X_train_p,y_train_p,X_test_p,y_test_p,i) for i in range(1,26)]

# Make a plot
plt.close()
plt.rcParams["figure.figsize"] = [10, 8]
plt.plot(list(range(1,26)),accu_list5,'-o')
plt.xticks(np.arange(0, 26, 1.0))
plt.title('Number of Neighbors vs. Success Rate for Parkinson Dataset')
plt.xlabel('Number of Neighbors')
plt.ylabel('Success Rate')
plt.grid()
plt.savefig('knn_parkinson.png')

# Time the KNN algorithm
start = time.time()
knn_rate(X_train_p,y_train_p,X_test_p,y_test_p,23)
end = time.time()
print("It takes " +  str(end - start)  + " seconds to run KNN algorithm on parkinson dataset with 23 neighbors.")


