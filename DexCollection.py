import sqlite3
import streamlit
import requests
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
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

page_title = "Collecting Dex Data"
page_icon = ":seedling:"
layout = "centered"

# Connecting to the database
connection = sqlite3.connect("DB.db")
cursor = connection.cursor()

streamlit.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

streamlit.header(f"Welcome to SproutCast! Your Gardening Companion")
streamlit.subheader(f"Tell us about your garden! Please enter your information below.")
with ((streamlit.form("input_form"))):
    pair = streamlit.text_input("Enter your plant info: ", key="pairId")
    submitted = streamlit.form_submit_button("Submit Form")

    if submitted:
        pairId = str(streamlit.session_state["pairId"])
        response = requests.get(
            "https://api.dexscreener.com/latest/dex/pairs/solana/" + pairId,
            headers={},
        )
        if response.status_code == 200:
            data = response.json()

            pairAddress = data["pairs"][0]["pairAddress"]
            tokenAddress = data["pairs"][0]["baseToken"]["address"]

            buysM5 = data["pairs"][0]["txns"]["m5"]["buys"]
            buysH1 = data["pairs"][0]["txns"]["h1"]["buys"]
            sellsM5 = data["pairs"][0]["txns"]["m5"]["sells"]
            sellsH1 = data["pairs"][0]["txns"]["h1"]["sells"]

            volM5 = data["pairs"][0]["volume"]["m5"]
            volH1 = data["pairs"][0]["volume"]["h1"]

            priceM5 = data["pairs"][0]["priceChange"]["m5"]
            priceH1 = data["pairs"][0]["priceChange"]["h1"]

            liquidity = data["pairs"][0]["liquidity"]["usd"]

            marketCap = data["pairs"][0]["marketCap"]

            paidResponse = requests.get("https://api.dexscreener.com/orders/v1/solana/" + tokenAddress, headers={},)
            paidData = paidResponse.json()
            paidProfile = 0
            paidAd = 0
            for item in paidData:
                if item["type"] == "tokenProfile" and item["status"] == "approved":
                    paidProfile = 1
                elif item["type"] == "tokenAd" and item["status"] == "approved":
                    paidAd = 1

            socials = 0
            websites = 0
            boosts = 0

            if "info" in data["pairs"][0]:
                if "socials" in data["pairs"][0]["info"]:
                    if len(data["pairs"][0]["info"]["socials"]) > 0:
                        socials = 1
                if "websites" in data["pairs"][0]["info"]:
                    if len(data["pairs"][0]["info"]["websites"]) > 0:
                        websites = 1

            if "boosts" in data["pairs"][0]:
                boosts = data["pairs"][0]["boosts"]["active"]

            timezone = pytz.timezone("America/New_York")
            now = datetime.datetime.now(timezone)
            current_minutes = (now.hour * 60) + now.minute
            dayOfWeek = now.weekday()

            poolsResponse = requests.get(
                "https://api.dexscreener.com/token-pairs/v1/solana/" + tokenAddress,
                headers={},
            )
            poolsData = poolsResponse.json()
            pools = len(poolsData)

            rugResponse = requests.get(
                "https://api.rugcheck.xyz/v1/tokens/" + tokenAddress + "/report/summary",
                headers={},
            )
            rugData = rugResponse.json()
            print(rugData)
            highHolder = 0
            lowLP = 0
            mutable = 0
            unlockedLP = 0
            topTen = 0
            singleHolder = 0
            highOwnership = 0
            if 'risks' in rugData:
                for item in rugData['risks']:
                    if item['name'] == "High holder correlation":
                        highHolder = item['score']
                    elif item['name'] == "Low amount of LP Providers":
                        lowLP = item['score']
                    elif item['name'] == "Mutable metadata":
                        mutable = item['score']
                    elif item['name'] == "Large Amount of LP Unlocked":
                        unlockedLP = item['score']
                    elif item['name'] == "Top 10 holders high ownership":
                        topTen = item['score']
                    elif item['name'] == "Single holder ownership":
                        singleHolder = item['score']
                    elif item['name'] == "High ownership":
                        highOwnership = item['score']
                    elif item['name'] == "Mutable metadata":
                        mutable = item['score']
            score = highHolder + lowLP + mutable + unlockedLP + singleHolder + highOwnership

            command = ("INSERT INTO tokens VALUES('" + str(pairAddress) + "', '" + str(tokenAddress) + "', '" + str(
                buysM5) + "', '" +
                       str(buysH1) + "', '" + str(sellsM5) + "', '" + str(sellsH1) + "', '" + str(volM5) + "', '" + str(
                        volH1) + "', '" +
                       str(priceM5) + "', '" + str(priceH1) + "', '" + str(liquidity) + "', '" + str(
                        marketCap) + "', '" +
                       str(paidProfile) + "', '" + str(paidAd) + "', '" + str(websites) + "', '" + str(
                        socials) + "', '" + str(boosts) + "', '" +
                       str(pools) + "', '" + str(dayOfWeek) + "', '" + str(current_minutes) + "', '" + str(
                        score) + "', '" +
                       str(highHolder) + "', '" + str(lowLP) + "', '" + str(mutable) + "', '" + str(unlockedLP) +
                       "', '" + str(singleHolder) + "', '" + str(highOwnership) + "', 'default')")

            buysToSells = float(buysH1) / float(sellsH1)
            volToLiquidity = float(volH1) / float(liquidity)
            volToMC = float(volH1) / float(marketCap)
            liquidityToMC = float(liquidity) / float(marketCap)
            liquidityToBuys = float(liquidity) / float(buysH1)
            MCToBuys = float(marketCap) / float(buysH1)
            poolsToLiquidity = float(pools) / float(liquidity)
            buysToVol = float(buysH1) / float(volH1)
            priceToVol = float(priceH1) / float(volH1)
            priceToVol5M = float(priceM5) / float(volM5)
            tokenInfo = []
            tokenInfo.append(
                [buysM5, buysH1, sellsM5, sellsH1, volM5, volH1, priceM5, priceH1, liquidity, marketCap, paidProfile,
                 paidAd, websites, socials, boosts, pools, buysToSells, volToLiquidity,
                 volToMC, liquidityToMC, liquidityToBuys, MCToBuys, poolsToLiquidity, buysToVol, priceToVol, priceToVol5M,
                 score, highHolder, lowLP, mutable, unlockedLP, singleHolder, highOwnership])

            data_frame = pd.read_csv("new_data.csv")
            data_frame['result'] = data_frame['result'].map({'Failure': 0, 'Success': 1})
            X = data_frame.drop('result', axis=1)
            y = data_frame['result']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
            base_rf = RandomForestClassifier(random_state=42, bootstrap=True, max_depth=70, max_features='sqrt',
                                                 min_samples_leaf=4, min_samples_split=110, n_estimators=400)
            base_rf.fit(X_train, y_train)
            base_gb = GradientBoostingClassifier(max_features='sqrt', max_depth=100, min_samples_leaf=4,
                                                     min_samples_split=100)
            base_gb.fit(X_train, y_train)
            base_nn = neighbors.KNeighborsClassifier(n_neighbors=6)
            base_nn.fit(X_train, y_train)
            base_nb = GaussianNB(var_smoothing=0.0012328467394420659)
            base_nb.fit(X_train, y_train)
            rf_resultStr = str(base_rf.predict(tokenInfo))
            gb_resultStr = str(base_gb.predict(tokenInfo))
            nn_resultStr = str(base_nn.predict(tokenInfo))
            nb_resultStr = str(base_nb.predict(tokenInfo))

            data_frame2 = pd.read_csv("new_data2.csv")
            data_frame2['result'] = data_frame2['result'].map(
                {'Failure': 0, 'Success': 1, 'RSuccess': 11, 'RFailure': -1, '2x': 2
                    , '3x': 3, '4x': 4, '10x': 10})
            X2 = data_frame2.drop('result', axis=1)
            y2 = data_frame2['result']
            X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.01)
            winners_rf = RandomForestClassifier(bootstrap=True, max_depth=90, max_features='sqrt', min_samples_leaf=2,
                                                min_samples_split=80, n_estimators=400)
            winners_rf.fit(X_train2, y_train2)
            winners_gb = GradientBoostingClassifier(max_features='sqrt', max_depth=100, min_samples_leaf=4,
                                                    min_samples_split=120)
            winners_gb.fit(X_train2, y_train2)
            winners_nn = neighbors.KNeighborsClassifier(n_neighbors=12)
            winners_nn.fit(X_train2, y_train2)
            rf_resultStr = rf_resultStr + str(winners_rf.predict(tokenInfo))
            gb_resultStr = gb_resultStr + str(winners_gb.predict(tokenInfo))
            nn_resultStr = nn_resultStr + str(winners_nn.predict(tokenInfo))

            #data_frameR = pd.read_csv("rug_data.csv")
            #data_frameR['result'] = data_frameR['result'].map({'R25': 25, 'R': 1, 'R2': 2})
            #XR = data_frameR.drop('result', axis=1)
            #yR = data_frameR['result']
            #X_trainR, X_testR, y_trainR, y_testR = train_test_split(XR, yR, test_size=0.01)
            #RWins_rf = RandomForestClassifier(random_state=42, bootstrap=True, max_depth=30, max_features='sqrt',
            #                                  min_samples_leaf=1, min_samples_split=2, n_estimators=200)
            #RWins_rf.fit(X_trainR, y_trainR)
            #RWins_gb = GradientBoostingClassifier()
            #RWins_gb.fit(X_trainR, y_trainR)
            #RWins_nn = neighbors.KNeighborsClassifier(n_neighbors=10)
            #RWins_nn.fit(X_trainR, y_trainR)
            #rf_resultStr = rf_resultStr + str(RWins_rf.predict(tokenInfo))
            #gb_resultStr = gb_resultStr + str(RWins_gb.predict(tokenInfo))
            #nn_resultStr = nn_resultStr + str(RWins_nn.predict(tokenInfo))f

            streamlit.subheader(rf_resultStr)
            streamlit.subheader(gb_resultStr)
            streamlit.subheader(nn_resultStr)
            streamlit.subheader(nb_resultStr)
            streamlit.subheader(command)

connection.close()
