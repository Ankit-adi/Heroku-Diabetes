import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv('diabetes.csv')
X=dataset.iloc[:,:8].values
Y=dataset.iloc[:,8].values

from sklearn.linear_model import LinearRegression
classifier=LinearRegression(random_state=0)
classifier.fit(X,Y)

pickle.dump(classifier,open('model.pkl', 'wb'))
 