import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

# create dataframe from csv file
df = pd.read_csv('Titanic.csv')

# explore dataset and find target variable
print(df.head())
print(df.describe())

# drop irrelevant factors
df = df.drop(['Passenger'], axis=1)

# make sure there are no missing values
pd.notnull(df)

# plot count plots of remaining factors
fig, axes = plt.subplots(2, 2)
sns.countplot(x="Class", data=df, ax=axes[0, 0])
sns.countplot(x="Sex", data=df, ax=axes[0, 1])
sns.countplot(x="Age", data=df, ax=axes[1, 0])
sns.countplot(x="Survived", data=df, ax=axes[1, 1])
plt.show()

# convert categorical variables into dummy variables
Class = pd.get_dummies(df["Class"])
Sex = pd.get_dummies(df["Sex"], drop_first=True)
Age = pd.get_dummies(df["Age"], drop_first=True)
Survived = pd.get_dummies(df["Survived"], drop_first=True)
df = pd.concat([Class, Sex, Age, Survived], axis=1)
print(df)

# partition the data into train and test sets
x = df.drop("Yes", axis=1)
y = df["Yes"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2022)

# fit the training data to a logistic regression model
logReg = LogisticRegression()
logReg.fit(x_train, y_train)

# display the accuracy, precision, and recall of survivability predictions
y_pred = logReg.predict(x_test)
print(classification_report(y_test, y_pred))
print("accuracy:", metrics.accuracy_score(y_test, y_pred))

# display the confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
plot_confusion_matrix(logReg, x_test, y_test)
plt.show()

# display the predicted value of survivability of an adult female passenger traveling 2nd class
print("predicted survivability of an adult female passenger in 2nd class:", logReg.predict([[0, 1, 0, 0, 0, 0]]))
