# -*- coding: utf-8 -*-
"""
Copyright (c) 2016, Cynthia S. Lo

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE. 
"""
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

# data curation
# classify passengers on the Titanic by whether they 'Survived' (labels), 
# based on their ['Pclass', 'Fare', 'Sex'] (features)
df = pd.read_csv('train.csv', header=0, index_col=0)

# data cleaning and integration
# drop all columns except ['Pclass', 'Fare', 'Sex']
# drop all rows containing at least one NaN
df.drop(labels=['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], 
        axis=1, inplace=True)
df.dropna(axis=0, inplace=True)

# define labels and features 
y = df['Survived'].copy()
X = df.drop(labels=['Survived'], axis=1)

# convert categorical feature to dummy/indicator variable (numerical)
X = pd.get_dummies(X,columns=['Sex'])

# data analytics
# split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, 
                                                      random_state=2016)

# perform logistic regression for classification
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

print('Validation score')
print(model.score(X_valid, y_valid))

y_true = y_valid
y_pred = model.predict(X_valid)

# print normalized confusion matrix to evaluate the quality of the model
cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print('Validation confusion matrix')

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized confusion matrix')
plt.colorbar()

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], '.3f'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label (Survived)')
plt.xlabel('Predicted label (Survived)')

plt.show()

# model testing
df2 = pd.read_csv('test.csv', header=0, index_col=0)

df2.drop(labels=['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], 
         axis=1, inplace=True)
X_test = df2.dropna(axis=0)

X_test = pd.get_dummies(X_test,columns=['Sex'])

print('Test labels')
print(model.predict(X_test))