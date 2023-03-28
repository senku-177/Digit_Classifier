## The project collect the MNIST dataset and train a model with that to predict hadn written numbers for a 784 pixel image

import pandas as pd
import numpy as np
from math import floor
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784',version=1)

X,Y= mnist["data"],mnist["target"]

Y= Y.astype(np.uint8)
X_train,Y_train,X_test,Y_test= X[:floor(0.15*X.shape[0])],Y[:floor(0.15*Y.size)],X[floor(0.85*X.shape[0]):],Y[floor(0.85*Y.size):]


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled=scaler.fit_transform(X_test.astype(np.float64))



from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train_scaled,Y_train)
predictions= forest_clf.predict(X_test_scaled)



from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,predictions)
print(accuracy)

# import pickle

# loaded_model=pickle.load(open("my_model_clf.pkl",'rb'))
# preds=loaded_model.predict(X_test_scaled)
# acc=accuracy_score(Y_test, preds)
# print(acc)