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

#----Base----
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
#GradientBooster
basegb_results = base_gb.predict(baseTokens)
basegb_correct1s = 0
basegb_incorrect1s = 0
basegb_correct0s = 0
basegb_incorrect0s = 0
#NearestNeighbors
baseTokensNumeric = []
for element in baseTokens:
    temp = []
    for item in element:
        temp.append(float(item))
    baseTokensNumeric.append(temp)
basenn_results = base_nn.predict(baseTokensNumeric)
basenn_correct1s = 0
basenn_incorrect1s = 0
basenn_correct0s = 0
basenn_incorrect0s = 0
#base model totals
base_two_thirdsc = 0
base_three_thirdsc = 0
base_two_thirdsi = 0
base_three_thirdsi = 0
for i in range(0, len(baseActual)):
    base_yes_votes = 0
    if baseActual[i] == 'Success':
        if baserf_results[i] == 1:
            baserf_correct1s = baserf_correct1s + 1
            base_yes_votes = base_yes_votes + 1
        elif baserf_results[i] == 0:
            baserf_incorrect1s = baserf_incorrect1s + 1
        if basegb_results[i] == 1:
            basegb_correct1s = basegb_correct1s + 1
            base_yes_votes = base_yes_votes + 1
        elif baserf_results[i] == 0:
            basegb_incorrect1s = basegb_incorrect1s + 1
        if basenn_results[i] == 1:
            basenn_correct1s = basenn_correct1s + 1
            base_yes_votes = base_yes_votes + 1
        elif baserf_results[i] == 0:
            basenn_incorrect1s = basenn_incorrect1s + 1
        if base_yes_votes == 2:
            base_two_thirdsc = base_two_thirdsc + 1
        elif base_yes_votes == 3:
            base_three_thirdsc = base_three_thirdsc + 1
    elif baseActual[i] == 'Failure':
        if baserf_results[i] == 0:
            baserf_correct0s = baserf_correct0s + 1
        elif baserf_results[i] == 1:
            baserf_incorrect0s = baserf_incorrect0s + 1
            base_yes_votes = base_yes_votes + 1
        if basegb_results[i] == 0:
            basegb_correct0s = basegb_correct0s + 1
        elif basegb_results[i] == 1:
            basegb_incorrect0s = basegb_incorrect0s + 1
            base_yes_votes = base_yes_votes + 1
        if baserf_results[i] == 0:
            basenn_correct0s = basenn_correct0s + 1
        elif basenn_results[i] == 1:
            basenn_incorrect0s = basenn_incorrect0s + 1
            base_yes_votes = base_yes_votes + 1
        if base_yes_votes == 2:
            base_two_thirdsi = base_two_thirdsi + 1
        elif base_yes_votes == 3:
            base_three_thirdsi = base_three_thirdsi + 1

#Output
print("Base Model:\nRF Only:\nCorrect 1s: " + str(baserf_correct1s) + " Incorrect 1s: " + str(baserf_incorrect1s))
print("Correct 0s: " + str(baserf_correct0s) + " Incorrect 0s: " + str(baserf_incorrect0s))
print("\nGB Only:\nCorrect 1s: " + str(basegb_correct1s) + " Incorrect 1s: " + str(basegb_incorrect1s))
print("Correct 0s: " + str(basegb_correct0s) + " Incorrect 0s: " + str(basegb_incorrect0s))
print("\nNN Only:\nCorrect 1s: " + str(basenn_correct1s) + " Incorrect 1s: " + str(basenn_incorrect1s))
print("Correct 0s: " + str(basenn_correct0s) + " Incorrect 0s: " + str(basenn_incorrect0s))
print("\nCorrect 2/3s: " + str(base_two_thirdsc) + " Incorrect 2/3s: " + str(base_two_thirdsi))
print("\nCorrect 3/3s: " + str(base_three_thirdsc) + " Incorrect 3/3s: " + str(base_three_thirdsi))

#----Winners----
winnersActual = []
winnersTokens = []
with open('winners_test_data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        winnersTokens.append(
            [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13]
                , row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23], row[24], row[25],
                row[26], row[27], row[28], row[29], row[30], row[31], row[32]])
        winnersActual.append(row[33])
winnersTokens.pop(0)
winnersActual.pop(0)
#RandomForest
winnersrf_results = winners_rf.predict(winnersTokens)
winnersrf_correct1s = 0
winnersrf_incorrect1s = 0
winnersrf_correct0s = 0
winnersrf_incorrect0s = 0
#GradientBooster
winnersgb_results = winners_gb.predict(winnersTokens)
winnersgb_correct1s = 0
winnersgb_incorrect1s = 0
winnersgb_correct0s = 0
winnersgb_incorrect0s = 0
#NearestNeighbors
winnersTokensNumeric = []
for element in winnersTokens:
    temp = []
    for item in element:
        temp.append(float(item))
    winnersTokensNumeric.append(temp)
winnersnn_results = winners_nn.predict(winnersTokensNumeric)
winnersnn_correct1s = 0
winnersnn_incorrect1s = 0
winnersnn_correct0s = 0
winnersnn_incorrect0s = 0
#winners model totals
winners_two_thirdsc = 0
winners_three_thirdsc = 0
winners_two_thirdsi = 0
winners_three_thirdsi = 0
for i in range(0, len(winnersActual)):
    winners_yes_votes = 0
    if winnersActual[i] == 'Success':
        if winnersrf_results[i] == 1:
            winnersrf_correct1s = winnersrf_correct1s + 1
            winners_yes_votes = winners_yes_votes + 1
        elif winnersrf_results[i] == 0:
            winnersrf_incorrect1s = winnersrf_incorrect1s + 1
        if winnersgb_results[i] == 1:
            winnersgb_correct1s = winnersgb_correct1s + 1
            winners_yes_votes = winners_yes_votes + 1
        elif winnersrf_results[i] == 0:
            winnersgb_incorrect1s = winnersgb_incorrect1s + 1
        if winnersnn_results[i] == 1:
            winnersnn_correct1s = winnersnn_correct1s + 1
            winners_yes_votes = winners_yes_votes + 1
        elif winnersrf_results[i] == 0:
            winnersnn_incorrect1s = winnersnn_incorrect1s + 1
        if winners_yes_votes == 2:
            winners_two_thirdsc = winners_two_thirdsc + 1
        elif winners_yes_votes == 3:
            winners_three_thirdsc = winners_three_thirdsc + 1
    elif winnersActual[i] == 'Failure':
        if winnersrf_results[i] == 0:
            winnersrf_correct0s = winnersrf_correct0s + 1
        elif winnersrf_results[i] == 1:
            winnersrf_incorrect0s = winnersrf_incorrect0s + 1
            winners_yes_votes = winners_yes_votes + 1
        if winnersgb_results[i] == 0:
            winnersgb_correct0s = winnersgb_correct0s + 1
        elif winnersgb_results[i] == 1:
            winnersgb_incorrect0s = winnersgb_incorrect0s + 1
            winners_yes_votes = winners_yes_votes + 1
        if winnersrf_results[i] == 0:
            winnersnn_correct0s = winnersnn_correct0s + 1
        elif winnersnn_results[i] == 1:
            winnersnn_incorrect0s = winnersnn_incorrect0s + 1
            winners_yes_votes = winners_yes_votes + 1
        if winners_yes_votes == 2:
            winners_two_thirdsi = winners_two_thirdsi + 1
        elif winners_yes_votes == 3:
            winners_three_thirdsi = winners_three_thirdsi + 1

#Output
print("Winners Model:\nRF Only:\nCorrect 1s: " + str(winnersrf_correct1s) + " Incorrect 1s: " + str(winnersrf_incorrect1s))
print("Correct 0s: " + str(winnersrf_correct0s) + " Incorrect 0s: " + str(winnersrf_incorrect0s))
print("\nGB Only:\nCorrect 1s: " + str(winnersgb_correct1s) + " Incorrect 1s: " + str(winnersgb_incorrect1s))
print("Correct 0s: " + str(winnersgb_correct0s) + " Incorrect 0s: " + str(winnersgb_incorrect0s))
print("\nNN Only:\nCorrect 1s: " + str(winnersnn_correct1s) + " Incorrect 1s: " + str(winnersnn_incorrect1s))
print("Correct 0s: " + str(winnersnn_correct0s) + " Incorrect 0s: " + str(winnersnn_incorrect0s))
print("\nCorrect 2/3s: " + str(winners_two_thirdsc) + " Incorrect 2/3s: " + str(winners_two_thirdsi))
print("\nCorrect 3/3s: " + str(winners_three_thirdsc) + " Incorrect 3/3s: " + str(winners_three_thirdsi))

#----Rugs----
RWinsActual = []
RWinsTokens = []
with open('RWins_test_data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        RWinsTokens.append(
            [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13]
                , row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23], row[24], row[25],
                row[26], row[27], row[28], row[29], row[30], row[31], row[32]])
        RWinsActual.append(row[33])
RWinsTokens.pop(0)
RWinsActual.pop(0)
#RandomForest
RWinsrf_results = RWins_rf.predict(RWinsTokens)
RWinsrf_correct1s = 0
RWinsrf_incorrect1s = 0
RWinsrf_correct0s = 0
RWinsrf_incorrect0s = 0
#GradientBooster
RWinsgb_results = RWins_gb.predict(RWinsTokens)
RWinsgb_correct1s = 0
RWinsgb_incorrect1s = 0
RWinsgb_correct0s = 0
RWinsgb_incorrect0s = 0
#NearestNeighbors
RWinsTokensNumeric = []
for element in RWinsTokens:
    temp = []
    for item in element:
        temp.append(float(item))
    RWinsTokensNumeric.append(temp)
RWinsnn_results = RWins_nn.predict(RWinsTokensNumeric)
RWinsnn_correct1s = 0
RWinsnn_incorrect1s = 0
RWinsnn_correct0s = 0
RWinsnn_incorrect0s = 0
#RWins model totals
RWins_two_thirdsc = 0
RWins_three_thirdsc = 0
RWins_two_thirdsi = 0
RWins_three_thirdsi = 0
for i in range(0, len(RWinsActual)):
    RWins_yes_votes = 0
    if RWinsActual[i] == 'R25':
        if RWinsrf_results[i] == 25:
            RWinsrf_correct1s = RWinsrf_correct1s + 1
            RWins_yes_votes = RWins_yes_votes + 1
        else:
            RWinsrf_incorrect1s = RWinsrf_incorrect1s + 1
        if RWinsgb_results[i] == 25:
            RWinsgb_correct1s = RWinsgb_correct1s + 1
            RWins_yes_votes = RWins_yes_votes + 1
        else:
            RWinsgb_incorrect1s = RWinsgb_incorrect1s + 1
        if RWinsnn_results[i] == 25:
            RWinsnn_correct1s = RWinsnn_correct1s + 1
            RWins_yes_votes = RWins_yes_votes + 1
        else:
            RWinsnn_incorrect1s = RWinsnn_incorrect1s + 1
        if RWins_yes_votes == 2:
            RWins_two_thirdsc = RWins_two_thirdsc + 1
        elif RWins_yes_votes == 3:
            RWins_three_thirdsc = RWins_three_thirdsc + 1
    elif RWinsActual[i] == 'R':
        if RWinsrf_results[i] == 1:
            RWinsrf_correct0s = RWinsrf_correct0s + 1
        else:
            RWinsrf_incorrect0s = RWinsrf_incorrect0s + 1
            RWins_yes_votes = RWins_yes_votes + 1
        if RWinsgb_results[i] == 1:
            RWinsgb_correct0s = RWinsgb_correct0s + 1
        else:
            RWinsgb_incorrect0s = RWinsgb_incorrect0s + 1
            RWins_yes_votes = RWins_yes_votes + 1
        if RWinsrf_results[i] == 1:
            RWinsnn_correct0s = RWinsnn_correct0s + 1
        else:
            RWinsnn_incorrect0s = RWinsnn_incorrect0s + 1
            RWins_yes_votes = RWins_yes_votes + 1
        if RWins_yes_votes == 2:
            RWins_two_thirdsi = RWins_two_thirdsi + 1
        elif RWins_yes_votes == 3:
            RWins_three_thirdsi = RWins_three_thirdsi + 1
    elif RWinsActual[i] == 'R2':
        if RWinsrf_results[i] == 2:
            RWinsrf_correct0s = RWinsrf_correct0s + 1
        else:
            RWinsrf_incorrect0s = RWinsrf_incorrect0s + 1
            RWins_yes_votes = RWins_yes_votes + 1
        if RWinsgb_results[i] == 2:
            RWinsgb_correct0s = RWinsgb_correct0s + 1
        else:
            RWinsgb_incorrect0s = RWinsgb_incorrect0s + 1
            RWins_yes_votes = RWins_yes_votes + 1
        if RWinsrf_results[i] == 2:
            RWinsnn_correct0s = RWinsnn_correct0s + 1
        else:
            RWinsnn_incorrect0s = RWinsnn_incorrect0s + 1
            RWins_yes_votes = RWins_yes_votes + 1
        if RWins_yes_votes == 2:
            RWins_two_thirdsi = RWins_two_thirdsi + 1
        elif RWins_yes_votes == 3:
            RWins_three_thirdsi = RWins_three_thirdsi + 1

#Output
print("Rugs Model:\nRF Only:\nCorrect 1s: " + str(RWinsrf_correct1s) + " Incorrect 1s: " + str(RWinsrf_incorrect1s))
print("Correct 0s: " + str(RWinsrf_correct0s) + " Incorrect 0s: " + str(RWinsrf_incorrect0s))
print("\nGB Only:\nCorrect 1s: " + str(RWinsgb_correct1s) + " Incorrect 1s: " + str(RWinsgb_incorrect1s))
print("Correct 0s: " + str(RWinsgb_correct0s) + " Incorrect 0s: " + str(RWinsgb_incorrect0s))
print("\nNN Only:\nCorrect 1s: " + str(RWinsnn_correct1s) + " Incorrect 1s: " + str(RWinsnn_incorrect1s))
print("Correct 0s: " + str(RWinsnn_correct0s) + " Incorrect 0s: " + str(RWinsnn_incorrect0s))
print("\nCorrect 2/3s: " + str(RWins_two_thirdsc) + " Incorrect 2/3s: " + str(RWins_two_thirdsi))
print("\nCorrect 3/3s: " + str(RWins_three_thirdsc) + " Incorrect 3/3s: " + str(RWins_three_thirdsi))