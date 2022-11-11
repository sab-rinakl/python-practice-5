import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot

# create a dataframe with the csv file
df = pd.read_csv('auto-mpg.csv')

# summarize the dataset
summary = df.describe()

# give the mean and median of mpg
print("mean of mpg: ", summary['mpg']['mean'])
print("median of mpg: ", summary['mpg']['50%'])

# answer questions and make plot to verify answer
print("the mean is higher than the median, meaning that there is a positive skew (toward the left)")
plt.hist(df['mpg'].values)
plt.show

# create pairplot matrix
sns.set()
sns.pairplot(summary.drop(['No'], axis=1))
plt.show()

# which attributes are strongly and weakly correlated
print("strongly correlated: displacement and horsepower")
print("weakly correlated: model year and weight")

# produce a displacement vs mpg scatterplot
x = df['displacement'].values
y = df['mpg'].values
plt.plot(x, y, '.')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.show()

# build a linear regression model
m, b = np.polyfit(x, y, 1)

# answer questions about linear regression model
print("the intercept value is", b)
print("the coefficient is", m)
print("the predicted value decreases")
print("the predicted value is", (m*200+b))

# display scatterplot and superimpose the linear regression line
x = df['displacement'].values
y = df['mpg'].values
reg_y = []
for i in range(0, len(x)):
    reg_y.append(m*x[i]+b)
plt.plot(x, y, '.')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.plot(x, reg_y, '-')
plt.show()

# plot the residuals
res_x = x.reshape(-1, 1)
ridge = Ridge()
v = ResidualsPlot(ridge)
v.fit(res_x, y)
v.show()
