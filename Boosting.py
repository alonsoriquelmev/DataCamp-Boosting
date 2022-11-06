import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, mean_squared_error as MSE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.preprocessing import scale


data = pd.read_csv('data_breast.csv', sep  = ',')  #Cargado de la base
data = data.drop(data.iloc[:,12:], axis = 1) #Se eliminan las columnas que no sirven con _std y _worst
data = data.drop(columns = 'id') #Se elimina la columna de 'id' que no entrega información necesaria

print(data.head())
print(data.info())

print(pd.unique(data['diagnosis']))
data['diagnosis'] = pd.get_dummies(data['diagnosis'])['M']

#AdaBoost Classifier
X = data.drop(columns = 'diagnosis')
y = data['diagnosis']

X = pd.DataFrame(scale(X), columns = X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state= 42)

dt = DecisionTreeClassifier(max_depth = 1, random_state = 42)

adb_clf = AdaBoostClassifier(base_estimator = dt, n_estimators = 100)

adb_clf.fit(X_train, y_train)

y_pred_proba = adb_clf.predict_proba(X_test)[:,1]

adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

#ROC Curve for AdaBoostClassifier

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr, label = 'AdaBoost CLassifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AdaBoost Classifier ROC Curve')
plt.legend()

#Gradient Boosting

df = pd.read_csv('auto-mpg.csv')
df = df.drop(columns = 'car name')

df['horsepower'] = df['horsepower'].astype(str)
df = df[df['horsepower'] != '?']
df['horsepower'] = pd.to_numeric(df['horsepower'])

X = df.drop(columns = 'mpg')
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

gbt = GradientBoostingRegressor(n_estimators = 300, max_depth = 1, random_state = 5)

gbt.fit(X_train, y_train)
y_pred = gbt.predict(X_test)

rmse_test = MSE(y_test, y_pred)**(1/2)

print(f'Test set RMSE of Gradient Boosting: {rmse_test}')

#Stochastic Gradient Boosting

sgbt = GradientBoostingRegressor(n_estimators = 300, max_depth = 1, max_features = 0.2, subsample = 0.8, random_state = 5)

sgbt.fit(X_train, y_train)
y_pred = sgbt.predict(X_test)

rmse_test = MSE(y_test, y_pred)**(1/2)

print(f'Test set RMSE of Stochastic Gradient Boosting: {rmse_test}')



