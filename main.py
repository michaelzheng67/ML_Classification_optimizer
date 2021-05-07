import numpy as np
import pandas as pd
import variables

# Get dataset
dataset = pd.read_csv('Social_Network_Ads_modified.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Getting training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling the x variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Logistic regression model
from sklearn.linear_model import LogisticRegression
classifier_first = LogisticRegression(random_state = 0)
classifier_first.fit(X_train, y_train)
y_pred_first = classifier_first.predict(X_test)

# K-nearest neighbors model
from sklearn.neighbors import KNeighborsClassifier
classifier_second = KNeighborsClassifier(n_neighbors = 5, metric = variables.knn_metric, p = 2)
classifier_second.fit(X_train, y_train)
y_pred_second = classifier_second.predict(X_test)

# SVM model
from sklearn.svm import SVC
classifier_third = SVC(kernel = variables.svm_kernel, random_state = 0)
classifier_third.fit(X_train, y_train)
y_pred_third = classifier_third.predict(X_test)

# Kernel SVM model
from sklearn.svm import SVC
classifier_fourth = SVC(kernel = variables.kernel_svm_kernel, random_state = 0)
classifier_fourth.fit(X_train, y_train)
y_pred_fourth = classifier_fourth.predict(X_test)

# Naive Baynes model
from sklearn.naive_bayes import GaussianNB
classifier_fifth = GaussianNB()
classifier_fifth.fit(X_train, y_train)
y_pred_fifth = classifier_fifth.predict(X_test)

# Decision Tree classification model
from sklearn.tree import DecisionTreeClassifier
classifier_sixth = DecisionTreeClassifier(criterion = variables.decision_tree_criterion, random_state = 0)
classifier_sixth.fit(X_train, y_train)
y_pred_sixth = classifier_sixth.predict(X_test)

# Random Forest classification model
from sklearn.ensemble import RandomForestClassifier
classifier_seventh = RandomForestClassifier(n_estimators = 10, criterion = variables.random_forest_criterion, random_state = 0)
classifier_seventh.fit(X_train, y_train)
y_pred_seventh = classifier_seventh.predict(X_test)

# Compare accuracy scores for each model. Print the name of the model that achieves the highest score
from sklearn.metrics import accuracy_score
accuracy_first = accuracy_score(y_test, y_pred_first)
accuracy_second = accuracy_score(y_test, y_pred_second)
accuracy_third = accuracy_score(y_test, y_pred_third)
accuracy_fourth = accuracy_score(y_test, y_pred_fourth)
accuracy_fifth = accuracy_score(y_test, y_pred_fifth)
accuracy_sixth = accuracy_score(y_test, y_pred_sixth)
accuracy_seventh = accuracy_score(y_test, y_pred_seventh)

max_accuracy = max(accuracy_first, accuracy_second, accuracy_third, accuracy_fourth, accuracy_fifth,
                   accuracy_sixth, accuracy_seventh)

# 1: logistic model
# 2: KNN model
# 3: SVM model
# 4: Kernel SVM model
# 5: Naive Baynes model
# 6: Decision Tree model
# 7: Random Forest model

accuracy_score_dict = { accuracy_first:'logistic model', accuracy_second:'KNN model', accuracy_third:'SVM model',
                        accuracy_fourth:'Kernel SVM model', accuracy_fifth:'Naive Baynes model',
                        accuracy_sixth:'Decision Tree model', accuracy_seventh:'Random Forest model'}

print(f"{accuracy_score_dict[max_accuracy]} has the highest accuracy score")
