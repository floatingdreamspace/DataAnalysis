import numpy
import csv
import pandas as pd
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV

#Original dataset
data_frame = pd.read_csv("new_data.csv")
data_frame['result'] = data_frame['result'].map({'Failure': 0, 'Success': 1})
X = data_frame.drop('result', axis=1)
y = data_frame['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
base_rf = RandomForestClassifier(random_state=42, bootstrap=True, max_depth=70, max_features='sqrt', min_samples_leaf=4, min_samples_split=110, n_estimators=400)
base_rf.fit(X_train, y_train)
base_gb = GradientBoostingClassifier(max_features='sqrt', max_depth=100, min_samples_leaf=4, min_samples_split=100)
base_gb.fit(X_train, y_train)
base_nn = neighbors.KNeighborsClassifier(n_neighbors=6)
base_nn.fit(X_train, y_train)

#Winners dataset
data_frame2 = pd.read_csv("new_data2.csv")
data_frame2['result'] = data_frame2['result'].map({'Failure': 0, 'Success': 1, 'RSuccess': 11, 'RFailure': -1, '2x': 2
                                                 , '3x': 3, '4x': 4, '10x': 10})
X2 = data_frame2.drop('result', axis=1)
y2 = data_frame2['result']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.01)
winners_rf = RandomForestClassifier(bootstrap=True, max_depth=90, max_features='sqrt', min_samples_leaf=2, min_samples_split=80, n_estimators=400)
winners_rf.fit(X_train2, y_train2)
winners_gb = GradientBoostingClassifier(max_features='sqrt', max_depth=100, min_samples_leaf=4, min_samples_split=120)
winners_gb.fit(X_train2, y_train2)
winners_nn = neighbors.KNeighborsClassifier(n_neighbors=12)
winners_nn.fit(X_train2, y_train2)

#RWins dataset
data_frameR = pd.read_csv("rug_data.csv")
data_frameR['result'] = data_frameR['result'].map({'R25': 25, 'R': 1, 'R2': 2})
XR = data_frameR.drop('result', axis=1)
yR = data_frameR['result']
X_trainR, X_testR, y_trainR, y_testR = train_test_split(XR, yR, test_size=0.01)
RWins_rf = RandomForestClassifier(random_state=42, bootstrap=True, max_depth=30, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=200)
RWins_rf.fit(X_trainR, y_trainR)
RWins_gb = GradientBoostingClassifier()
RWins_gb.fit(X_trainR, y_trainR)
RWins_nn = neighbors.KNeighborsClassifier(n_neighbors=10)
RWins_nn.fit(X_trainR, y_trainR)

baseActual = []
baseTokens = []
with open('test_data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        baseTokens.append(
            [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13]
                , row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23], row[24], row[25],
                row[26], row[27], row[28], row[29], row[30], row[31], row[32]])
        baseActual.append(row[33])
baseTokens.pop(0)
baseActual.pop(0)
#RandomForest
baserf_results = base_rf.predict(baseTokens)
baserf_correct1s = 0
baserf_incorrect1s = 0
baserf_correct0s = 0
baserf_incorrect0s = 0
for i in range(0, len(baseActual)):
    if baseActual[i] == 1:
        if baserf_results[i] == 1:
            baserf_correct1s = baserf_correct1s + 1
        elif baserf_results[i] == 0:
            baserf_incorrect1s = baserf_incorrect1s + 1
    elif baseActual[i] == 0:
        if baserf_results[i] == 0:
            baserf_correct0s = baserf_correct0s + 1
        elif baserf_results[i] == 1:
            baserf_incorrect0s = baserf_incorrect0s + 1
#GradientBooster
basegb_results = base_gb.predict(baseTokens)
basegb_correct1s = 0
basegb_incorrect1s = 0
basegb_correct0s = 0
basegb_incorrect0s = 0
for i in range(0, len(baseActual)):
    if baseActual[i] == 1:
        if basegb_results[i] == 1:
            basegb_correct1s = basegb_correct1s + 1
        elif baserf_results[i] == 0:
            basegb_incorrect1s = basegb_incorrect1s + 1
    elif baseActual[i] == 0:
        if basegb_results[i] == 0:
            basegb_correct0s = basegb_correct0s + 1
        elif basegb_results[i] == 1:
            basegb_incorrect0s = basegb_incorrect0s + 1
#NearestNeighbors
basenn_results = base_nn.predict(baseTokens)
basenn_correct1s = 0
basenn_incorrect1s = 0
basenn_correct0s = 0
basenn_incorrect0s = 0
for i in range(0, len(baseActual)):
    if baseActual[i] == 1:
        if basenn_results[i] == 1:
            basenn_correct1s = basenn_correct1s + 1
        elif baserf_results[i] == 0:
            basenn_incorrect1s = basenn_incorrect1s + 1
    elif baseActual[i] == 0:
        if baserf_results[i] == 0:
            basenn_correct0s = basenn_correct0s + 1
        elif basenn_results[i] == 1:
            basenn_incorrect0s = basenn_incorrect0s + 1
#Output
print("")