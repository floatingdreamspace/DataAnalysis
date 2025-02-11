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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier

# Connecting to the database
connection = sqlite3.connect("DB2.db")
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
#with open('new_data2.csv', 'w', newline='') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerows(tokens)

resultStr2 = ""
data_frame2 = pd.read_csv("new_data2.csv")
data_frame2['result'] = data_frame2['result'].map({'Failure': 0, 'Success': 1, 'RSuccess': 11, 'RFailure': -1, '2x': 2
                                                 , '3x': 3, '4x': 4, '5x': 5, '6x': 6, '7x': 7
                                                 , '8x': 8, '9x': 9, '10x': 10})
X2 = data_frame2.drop('result', axis=1)
y2 = data_frame2['result']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
rf2 = RandomForestClassifier(random_state=42)
rf2.fit(X_train2, y_train2)
y_pred2 = rf2.predict(X_test2)
accuracy = accuracy_score(y_test2, y_pred2)
print(y_test2, y_pred2)
print("Accuracy:", accuracy)