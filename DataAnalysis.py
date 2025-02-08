import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
import sqlite3
import csv
from Token import Token
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Connecting to the database
connection = sqlite3.connect("DB.db")
cursor = connection.cursor()

DBTokens = []  # Array of city data retrieved from the database.
tokens = []  # Array of City objects retrieved from the database.
cursor.execute("SELECT * FROM tokens")
DBTokens = cursor.fetchall()

# Populating the array with City objects using data from the database:
tokens.append(["buysM5", "buysH1", "sellsM5", "sellsH1", "volM5", "volH1", "priceM5", "priceH1", "liquidity", "marketCap",
                 "paidProfile", "paidAd", "websites", "socials", "boosts", "pools", "buysToSells",
                 "volToLiquidity", "volToMC", "liquidityToMC", "liquidityToBuys", "MCToBuys", "poolsToLiquidity", "buysToVol",
                 "score", "highHolder", "lowLP", "mutable", "unlockedLP", "singleHolder", "highOwnership", "result"])
for row in DBTokens:
    buysToSells = float(row[3]) / float(row[5])
    volToLiquidity = float(row[7]) / float(row[10])
    volToMC = float(row[7]) / float(row[11])
    liquidityToMC = float(row[10]) / float(row[11])
    liquidityToBuys = float(row[10]) / float(row[3])
    MCToBuys = float(row[11]) / float(row[3])
    poolsToLiquidity = float(row[17]) / float(row[10])
    buysToVol = float(row[3]) / float(row[7])
    tokens.append([row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13]
                        , row[14], row[15], row[16], row[17], buysToSells,
                        volToLiquidity, volToMC, liquidityToMC, liquidityToBuys, MCToBuys, poolsToLiquidity, buysToVol,
                        row[20], row[21], row[22], row[23], row[24], row[25], row[26], row[27]])
#with open('new_data.csv', 'w', newline='') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerows(tokens)

data_frame = pd.read_csv("new_data.csv")
data_frame['result'] = data_frame['result'].map({'Failure': 0, 'Success': 1})
X = data_frame.drop('result', axis=1)
y = data_frame['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
#rf = RandomForestClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42, bootstrap=False, max_depth=80, max_features=2, min_samples_leaf=2, min_samples_split=7, n_estimators=500)
scaler = StandardScaler()
X_train_regression = scaler.fit_transform(X_train)
X_test3 = scaler.transform(X_test)
rmodel = LogisticRegression(penalty='l1', C=np.float64(0.23357214690901212), solver='saga', max_iter=1000)
#rmodel = LogisticRegression()
#rmodel = RandomForestRegressor(random_state=42, bootstrap=True, max_depth=20, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=1200)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#rf_random.fit(X_train, y_train)
#print(rf_random.best_params_)
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 40],
    'max_features': [2, 3, 'sqrt'],
    'min_samples_leaf': [1, 2, 3, 4],
    'min_samples_split': [1, 2, 4],
    'n_estimators': [800, 1000, 1200, 1400]
}
grid_search = GridSearchCV(estimator = rmodel, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
#grid_search.fit(X_train, y_train)
#print(grid_search.best_params_)

param_grid2 = [
    {'penalty':['l1','l2','elasticnet'],
    'C' : np.logspace(-4,4,20),
    'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter'  : [100,1000,2500,5000]
}]
clf = GridSearchCV(rmodel,param_grid = param_grid2, cv = 3, verbose=True,n_jobs=-1)
#clf.fit(X_train_regression, y_train)
#print(clf.best_params_)

#rf = BalancedRandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(y_test, y_pred)
print("Accuracy:", accuracy)

rmodel.fit(X_train_regression, y_train)
y_pred = rmodel.predict(X_test3)
accuracy = accuracy_score(y_test, y_pred)
print("Regression Accuracy:", accuracy)

connection.close()