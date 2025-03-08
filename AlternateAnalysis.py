import sqlite3
import datetime
import pytz
import csv
import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier

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
#with open('new_data.csv', 'w', newline='') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerows(tokens)

resultStr2 = ""
data_frame2 = pd.read_csv("new_data.csv")
data_frame2['result'] = data_frame2['result'].map({'Failure': 0, 'Success': 1, 'RSuccess': 11, 'RFailure': -1, '2x': 2
                                                 , '3x': 3, '4x': 4, '10x': 10})
X2 = data_frame2.drop('result', axis=1)
y2 = data_frame2['result']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)

#for clarifying model
#rf2 = GradientBoostingClassifier(max_features='sqrt', max_depth=100, min_samples_leaf=4, min_samples_split=120)
#rf2 = neighbors.KNeighborsClassifier(n_neighbors=12)

#for original model
#rf2 = GradientBoostingClassifier(max_features='sqrt', max_depth=100, min_samples_leaf=4, min_samples_split=100)
#rf2 = neighbors.KNeighborsClassifier(n_neighbors=6)

#for model tuning
#rf2 = GradientBoostingClassifier()
#rf2 = svm.SVC(kernel='rbf')
#rf2 = neighbors.KNeighborsClassifier(n_neighbors=12)
#rf2 = GaussianNB(var_smoothing=0.00000001)

rf2.fit(X_train2, y_train2)

# Number of features to consider at every split
max_features = [8, 10, 12, 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in numpy.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [10, 50, 70, 100, 120]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Create the random grid
random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
rf_random = RandomizedSearchCV(estimator = rf2, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
#rf_random.fit(X_train2, y_train2)
#print(rf_random.best_params_)

param_grid = {'max_features': ['sqrt'],
               'max_depth': [60, 80, 100],
               'min_samples_split': [80, 100, 110, 120],
               'min_samples_leaf': [3, 4, 5]}
grid_search = GridSearchCV(estimator = rf2, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
#grid_search.fit(X_train2, y_train2)
#print(grid_search.best_params_)

y_pred2 = rf2.predict(X_test2)
correct1s = 0
correct0s = 0
incorrect1s = 0
incorrect0s = 0
accuracy = accuracy_score(y_test2, y_pred2)
'''print(y_test2[0])
for i in y_test2:
    value = int(y_test2[i])
    if value == 0:
        if value == 0:
            correct0s = correct0s + 1
        else:
            incorrect0s = incorrect0s + 1
    elif value == 1:
        if value == 0:
            correct1s = correct1s + 1
        else:
            incorrect1s = incorrect1s + 1'''
print(y_test2, y_pred2)
print("Accuracy:", accuracy)
'''print("Correct 1s: " + correct1s)
print("Incorrect 1s: " + incorrect1s)
print("Correct 0s: " + correct0s)
print("Incorrect 0s: " + incorrect0s)'''