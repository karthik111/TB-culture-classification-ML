# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:39:34 2018

@author: karthik.venkat
"""

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.learning_curve import validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))
    
# split the data with 50% in each set
X1, X2, y1, y2 = train_test_split(x_train, y_train, random_state=None,
                                  train_size=0.8)

model = clf

# fit the model on one set of data
model.fit(X1, y1)

# evaluate the model on the second set of data
y2_model = model.predict(X2)
print (accuracy_score(y2, y2_model))

y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
print (accuracy_score(y1, y1_model), accuracy_score(y2, y2_model))

print(cross_val_score(model, x_train, y_train, cv=5))

param_range = np.arange(1, 100, 1)

train_score1, val_score1 = validation_curve(model, X1, y1,
                                            param_name="n_estimators", 
                                             param_range=param_range, cv=7)

train_score2, val_score2 = validation_curve(model, X2, y2,
                                            param_name="n_estimators", 
                                             param_range=param_range, cv=7)

plt.plot(param_range, np.median(train_score2, 1), color='blue', alpha=0.4, linestyle='dashed', label='training score with a smaller dataset')
plt.plot(param_range, np.median(val_score2, 1), color='red', alpha=0.4, linestyle='dashed', label='validation score with a smaller dataset')
plt.plot(param_range, np.median(train_score1, 1), color='blue', label='training score with a larger dataset')
plt.plot(param_range, np.median(val_score1, 1), color='red', label='validation score with a larger dataset')
plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score');

train_scores = train_score2
test_scores = val_score2

## Calculate mean and standard deviation for training set scores
#train_mean = np.mean(train_scores, axis=1)
#train_std = np.std(train_scores, axis=1)
#
## Calculate mean and standard deviation for test set scores
#test_mean = np.mean(test_scores, axis=1)
#test_std = np.std(test_scores, axis=1)
#
## Plot mean accuracy scores for training and test sets
#plt.plot(param_range, train_mean, label="Training score", color="black")
#plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
#
## Plot accurancy bands for training and test sets
#plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
#plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")
#
# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()