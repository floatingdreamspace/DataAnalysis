import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
import sqlite3
import csv
from Token import Token

# Connecting to the database
connection = sqlite3.connect("DB.db")
cursor = connection.cursor()

DBTokens = []  # Array of city data retrieved from the database.
tokens = []  # Array of City objects retrieved from the database.
cursor.execute("SELECT * FROM tokens")
DBTokens = cursor.fetchall()

# Populating the array with City objects using data from the database:
tokens.append(["buysM5", "buysH1", "sellsM5", "sellsH1", "volM5", "volH1", "priceM5", "priceH1", "liquidity", "marketCap",
                 "paidProfile", "paidAd", "websites", "socials", "boosts", "pools", "dayOfWeek", "time", "result", "buysToSells",
                 "volToLiquidity", "volToMC", "liquidityToMC", "liquidityToBuys", "MCToBuys", "poolsToLiquidity", "buysToVol"])
'''for row in DBTokens:
    buysToSells = float(row[3]) / float(row[5])
    volToLiquidity = float(row[7]) / float(row[10])
    volToMC = float(row[7]) / float(row[11])
    liquidityToMC = float(row[10]) / float(row[11])
    liquidityToBuys = float(row[10]) / float(row[3])
    MCToBuys = float(row[11]) / float(row[3])
    poolsToLiquidity = float(row[17]) / float(row[10])
    buysToVol = float(row[3]) / float(row[7])
    tokens.append([row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13]
                        , row[14], row[15], row[16], row[17], row[18], row[19], row[20], buysToSells,
                        volToLiquidity, volToMC, liquidityToMC, liquidityToBuys, MCToBuys, poolsToLiquidity, buysToVol])
with open('university_records.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(tokens)'''

data_frame = pd.read_csv("university_records.csv")
data_frame['result'] = data_frame['result'].map({'Failure': 0, 'Success': 1})
X = data_frame.drop('result', axis=1)
y = data_frame['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier(n_estimators=10, class_weight='balanced')
#rf = RandomForestClassifier(n_estimators=10, class_weight='balanced_subsample')
#rf = BalancedRandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

connection.close()