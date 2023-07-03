###############################################################
# Imports 
###############################################################
import matplotlib.pyplot as plt # so we can add to plot
from sklearn.metrics import accuracy_score, f1_score # grade the results
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler # standardize data
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC # the algorithm

from sklearn.decomposition import PCA # PCA package

###############################################################
# Constants 
###############################################################

MAX_PCA = 7
RANDOM_STATE = 0
COLOUMNS = ['Temperature','Irradiance','Isc','Voc','FF','MP','Eff','Class']
TEST_DATA_SIZE = 0.3 # 0.9 # for normal training it is 0.3 for 70:30 split. for the 10:90 case mentioned replace 0.3 for 0.9


###############################################################
# Class Initializations
###############################################################
oversample = SMOTE()
sc = StandardScaler() # create the standard scalar

KNNClassifier = KNeighborsClassifier(n_neighbors=5)
RNClassifier = RandomForestClassifier(n_estimators=51, random_state=RANDOM_STATE)
DTClassifier = DecisionTreeClassifier(criterion='entropy',max_depth=3 ,random_state=RANDOM_STATE)
# last three require normalization
LRClassifier = LogisticRegression(C=0.025, solver='liblinear', multi_class='ovr', random_state=RANDOM_STATE)
SVCClassifier= SVC(kernel='linear', C=0.007, random_state=RANDOM_STATE)
PPNClassifier = Perceptron(max_iter=7, tol=1e-3, eta0=0.001,
    fit_intercept=True, random_state=RANDOM_STATE, verbose=False)

Classifiers = [KNNClassifier, RNClassifier, DTClassifier, LRClassifier, SVCClassifier, PPNClassifier]
Classifiers_Names = ['KNN Classifier', 'Random Forest Classifier', 'Decision Tree Classifier', 
    'Logistic Regression Classifier', 'SVC Classifier', 'Perceptron Classifier']


###############################################################
# Aggregation Datastructures for Analytics 
###############################################################

excludedpanels = []
paneldata = []
pca_accurracies = {}
classifiers_accurracies = {}
classifiers_f1scores = {}

files = os.listdir('csvfiles')
# a loop to test on every panel
for file in files:
    ###############################################################
    # Step 1: Data Cleaning and Preparation
    ###############################################################
    df = pd.read_csv('csvfiles/'+file, names=COLOUMNS) #reading data from files
    df = df.dropna() # removing nulls and corrupted rows
    paneldata.append(df) 
    df2 = df.groupby(df['Class']) # separating data based on class groups to insure fair distribution of classes
    non_faulty = df2.get_group(1)
    non_faulty_count = non_faulty.shape[0]
    total_count = df.shape[0]
    if total_count == non_faulty_count: # excluding files that do not contain faulty samples (single class)
        excludedpanels.append((file, 'no faulty samples were found'))
        continue
    elif total_count < 50: # excluding files that do not contain data
        excludedpanels.append((file, 'sample count was too low'))
        continue

    faulty = df2.get_group(0)
    faulty_count = faulty.shape[0]
    print(non_faulty_count, faulty_count, total_count)
    
    # separating training and testing datasets
    train_non_faulty, test_non_faulty = train_test_split(non_faulty, test_size=TEST_DATA_SIZE, random_state=RANDOM_STATE)
    train_faulty, test_faulty = train_test_split(faulty, test_size=TEST_DATA_SIZE, random_state=RANDOM_STATE)
    # creating training data
    frames_train = [train_non_faulty, train_faulty]    
    train = pd.concat(frames_train)  # merging the classes data
    train = train.sample(frac = 1, random_state=RANDOM_STATE) # ensuring random order for dataset for appropriate training
    #creating test dataset
    frames_test = [test_non_faulty, test_faulty]    
    test = pd.concat(frames_test)  
    y_train, y_test = train['Class'], test['Class']
    # separating features from output
    x_train, x_test = train.drop('Class', axis=1), test.drop('Class', axis=1)  
    x_train, x_test = x_train.values, x_test.values


    ###############################################################
    # Step 2: Treating Data Imbalance and Augmentation With SMOTE
    ###############################################################
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    
    ###############################################################
    # step 3: Training Algorithms, Comparing Accurracy & F1 Scores
    ###############################################################
    sc.fit(x_train) # compute the required transformation
    X_train_std, X_test_std = sc.transform(x_train), sc.transform(x_test) # apply to the training data and test data 

    # holds loop values
    temp_array_acc = []  
    temp_array_f1 = []
    for i, classifier in enumerate(Classifiers):
        if i > 2: #for last three classifiers that need normalization
            classifier.fit(X_train_std, y_train) # training
            y_pred = classifier.predict(X_test_std) # prediction on test sample
        else: # for the rest classifiers
            classifier.fit(x_train, y_train) # training
            y_pred = classifier.predict(x_test) # prediction on test sample
        
        # computing accuracy for each classifier
        success_rate = accuracy_score(y_test, y_pred)
        temp_array_acc.append(success_rate)
        f1score = f1_score(y_test, y_pred)
        temp_array_f1.append(f1score)
        #print('Accurracy: ', success_rate, 'F1 score: ', f1score, Classifiers_Names[i], file)

    classifiers_accurracies[file] = temp_array_acc
    classifiers_f1scores[file] = temp_array_f1


    ###############################################################
    # step 4: Choosing best Performing Model for further analysis
    ###############################################################
    # please note we should use some code to determine highest accurracy 
    # and choose but since we know already we just put the wining classifier
    model = RNClassifier

    ###############################################################
    # step 5: Apply PCA on chosen model with n_components from 1:7
    ###############################################################
    accuracies = []
    for n in range(1,MAX_PCA + 1):
        pca = PCA(n_components=n) # only keep n best features!
        X_train_pca = pca.fit_transform(X_train_std) # apply to the train data
        X_test_pca = pca.transform(X_test_std) # do the same to the test data
        model.fit(X_train_pca, y_train) #training model
        y_pred = model.predict(X_test_pca) # prediction on test sample
        success_rate = accuracy_score(y_test, y_pred) 
        success_rate = round(100*success_rate,1)
        accuracies.append(success_rate)

    pca_accurracies[file] = accuracies







###############################################################
# step 6: Showing Aggregate Results and plotting
###############################################################
# please note not all graphs were here some of the data were just copied to matlab to use matlab graphing utilities

print('excluded:') # all excluded files
for el in excludedpanels:
    print(el[0], ' ', el[1])

print('pca elements accurracies: 1, 2, 3, 4, 5, 6, 7')
for key in pca_accurracies: # all pca results for each panel for each number of features
    print(key, ' ', pca_accurracies[key])

num_columns = 7
num_rows = len(pca_accurracies)
# computing average pca results accross panels for each feature size 1:7
averages = [sum(pca_accurracies[key][i] for key in pca_accurracies) / num_rows for i in range(num_columns)]
print(averages)

print('classifier accurracies knn, rn, dt, lr, svm, perceptron')
for key in classifiers_accurracies:
    print(key, ' ', classifiers_accurracies[key])

num_columns = len(classifiers_accurracies['panel01.csv'])
num_rows = len(classifiers_accurracies)

averages = [sum(classifiers_accurracies[key][i] for key in classifiers_accurracies) / num_rows for i in range(num_columns)]
print(averages)

plt.bar(Classifiers_Names, averages, color ='blue', width = 0.1)
plt.show()


print('classifier f1scores knn, rn, dt, lr, svm, perceptron')

for key in classifiers_f1scores:
    print(key, ' ', classifiers_f1scores[key])

num_columns = len(classifiers_f1scores['panel01.csv'])
num_rows = len(classifiers_f1scores)

averages2 = [sum(classifiers_f1scores[key][i] for key in classifiers_f1scores) / num_rows for i in range(num_columns)]
print(averages2)

  
 
# creating the bar plot
plt.bar(Classifiers_Names, averages2, color ='maroon', width = 0.1)

plt.show()

