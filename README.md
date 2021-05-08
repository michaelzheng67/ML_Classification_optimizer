# ML_Classification_optimizer

* NOT DONE YET *

Python-based application

Imports: Pandas, Scikit-learn

tldr: Prints which machine learning classification model would work best for a given dataset. 

Inspiration and tutorial based on Udemy Machine Learning course by Kirill Eremenko. This algorithm works by having the user insert a .csv file of data that can be grouped and classified, and runs it through multiple classification models, in which the best possible model for the dataset is determined by metric assessment. Firstly, the .py file is configured so that the user is directing it to connect to data within a given .csv file. Then, the data is split into training set and test set, undergoes feature scaling, and then is plugged into seven different classification models from scikit-learn. Then, the models are judged on multiple metrics also derived from scikit-learn. 

Credit to the Machine Learning course for providing the test data and the foundational code for the basic way that the models can run and splitting / scaling the test data. 

notes: 
- The variables file import that the main.py file is referring to is another .py file that stores strings that the models use 
- In order for the algorithm to work, we must ensure that the dependent variables are placed before the independent variable in terms of column order. This means that the independent variable in which the classification is trying to guess is going to be in the last column of the .csv file 
