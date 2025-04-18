import sqlite3
import datetime
import pytz
import csv
import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingClassifier

# Connecting to the database
connection = sqlite3.connect("DBR.db")
cursor = connection.cursor()

DBTokens = []  # Array of city data retrieved from the database.
tokens = []  # Array of City objects retrieved from the database.
cursor.execute("SELECT * FROM tokens")
DBTokens = cursor.fetchall()

# Populating the array with City objects using data from the database:
tokens.append(["buysM5", "buysH1", "sellsM5", "sellsH1", "volM5", "volH1", "priceM5", "priceH1", "liquidity", "marketCap",
                 "paidProfile", "paidAd", "websites", "socials", "boosts", "pools", "buysToSells",
                 "volToLiquidity", "volToMC", "liquidityToMC", "liquidityToBuys", "MCToBuys", "poolsToLiquidity", "buysToVol",
                 "priceToVol", "priceToVol5M", "score", "highHolder", "lowLP", "mutable", "unlockedLP", "singleHolder", "highOwnership", "result"])
for row in DBTokens:
    buysToSells = float(row[3]) / float(row[5])
    volToLiquidity = float(row[7]) / float(row[10])
    volToMC = float(row[7]) / float(row[11])
    liquidityToMC = float(row[10]) / float(row[11])
    liquidityToBuys = float(row[10]) / float(row[3])
    MCToBuys = float(row[11]) / float(row[3])
    poolsToLiquidity = float(row[17]) / float(row[10])
    buysToVol = float(row[3]) / float(row[7])
    priceToVol = float(row[9]) / float(row[7])
    priceToVol5M = float(row[8]) / float(row[6])
    tokens.append([row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13]
                        , row[14], row[15], row[16], row[17], buysToSells,
                        volToLiquidity, volToMC, liquidityToMC, liquidityToBuys, MCToBuys, poolsToLiquidity, buysToVol,
                        priceToVol, priceToVol5M,
                        row[20], row[21], row[22], row[23], row[24], row[25], row[26], row[27]])
#with open('rug_data.csv', 'w', newline='') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerows(tokens)

resultStr2 = ""
data_frame2 = pd.read_csv("rug_data.csv")
data_frame2['result'] = data_frame2['result'].map({'R25': 25, 'R': 1, 'R2': 2})
X2 = data_frame2.drop('result', axis=1)
y2 = data_frame2['result']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)

#for model tuning
#rf2 = RandomForestClassifier(random_state=42)
#rf2 = GradientBoostingClassifier()
rf2 = neighbors.KNeighborsClassifier(n_neighbors=10)

#for model tests
#rf2 = RandomForestClassifier(random_state=42, bootstrap=True, max_depth=30, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=200)
#rf2 = GradientBoostingClassifier()
#rf2 = neighbors.KNeighborsClassifier(n_neighbors=10)

rf2.fit(X_train2, y_train2)

# Number of trees in random forest
n_estimators = [int(x) for x in numpy.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in numpy.linspace(10, 110, num = 11)]
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
rf_random = RandomizedSearchCV(estimator = rf2, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#rf_random.fit(X_train2, y_train2)
#print(rf_random.best_params_)

param_grid = {
    'bootstrap': [True],
    'max_depth': [30, 50, 70, None],
    'max_features': [10, 15, 'sqrt'],
    'min_samples_leaf': [1, 2, 4, 6],
    'min_samples_split': [2, 3, 5],
    'n_estimators': [200, 300, 400, 600]
}
grid_search = GridSearchCV(estimator = rf2, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
#grid_search.fit(X_train2, y_train2)
#print(grid_search.best_params_)

y_pred2 = rf2.predict(X_test2)
accuracy = accuracy_score(y_test2, y_pred2)
print(y_test2, y_pred2)
print("Accuracy:", accuracy)