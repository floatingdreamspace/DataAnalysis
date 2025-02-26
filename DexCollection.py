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

page_title = "Collecting Dex Data"
page_icon = ":seedling:"
layout = "centered"

# Connecting to the database
connection = sqlite3.connect("DB.db")
cursor = connection.cursor()

streamlit.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

streamlit.header(f"Welcome to SproutCast! Your Gardening Companion")
streamlit.subheader(f"Tell us about your garden! Please enter your information below. SproutCast will use your inputs" +
                    " along with local weather data to predict how much water you should be giving your garden " +
                    "using machine learning!")
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
            tokenInfo = []
            tokenInfo.append(
                [buysM5, buysH1, sellsM5, sellsH1, volM5, volH1, priceM5, priceH1, liquidity, marketCap, paidProfile,
                 paidAd, websites, socials, boosts, pools, buysToSells, volToLiquidity,
                 volToMC, liquidityToMC, liquidityToBuys, MCToBuys, poolsToLiquidity, buysToVol, score, highHolder,
                 lowLP, mutable, unlockedLP, singleHolder, highOwnership])

            resultStr = ""
            regressionr = ""
            for i in range(0, 1):
                data_frame = pd.read_csv("new_data.csv")
                data_frame['result'] = data_frame['result'].map({'Failure': 0, 'Success': 1})
                X = data_frame.drop('result', axis=1)
                y = data_frame['result']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                #scaler = StandardScaler()
                #X_train_regression = scaler.fit_transform(X_train)
                #X_test = scaler.transform(X_test)
                #rmodel = LogisticRegression(penalty='l1', C=numpy.float64(0.0001), solver='liblinear')
                #rmodel = LogisticRegression()
                #rmodel.fit(X_train_regression, y_train)
                #y_pred = rmodel.predict(X_test)
                #accuracy = accuracy_score(y_test, y_pred)
                #streamlit.subheader("Accuracy: " + str(accuracy * 100))
                rf = RandomForestClassifier(random_state=42, bootstrap=True, max_depth=70, max_features='sqrt', min_samples_leaf=4, min_samples_split=10, n_estimators=400)
                #rf = BalancedRandomForestClassifier(n_estimators=10)
                rf.fit(X_train, y_train)
                #rf.fit(X, y)
                #regressionr = regressionr + str(rmodel.predict(tokenInfo))
                resultStr = resultStr + str(rf.predict(tokenInfo))
            #streamlit.subheader(resultStr)
            #streamlit.subheader(regressionr)

            if resultStr == "[1]":
                resultStr2 = ""
                data_frame2 = pd.read_csv("new_data2.csv")
                data_frame2['result'] = data_frame2['result'].map(
                    {'Failure': 0, 'Success': 1, 'RSuccess': 11, 'RFailure': -1, '2x': 2
                        , '3x': 3, '4x': 4, '5x': 5, '6x': 6, '7x': 7
                        , '8x': 8, '9x': 9, '10x': 10})
                X2 = data_frame2.drop('result', axis=1)
                y2 = data_frame2['result']
                X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)
                rf2 = RandomForestClassifier(bootstrap=True, max_depth=20, max_features=15, min_samples_leaf=4, min_samples_split=4, n_estimators=1400)
                rf2.fit(X_train2, y_train2)
                resultStr = resultStr + " " + str(rf2.predict(tokenInfo))
                streamlit.subheader(resultStr)

                #two more iterations if the first one came back success
                for j in range(0, 2):
                    resultStr = ""
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    rf.fit(X_train, y_train)
                    resultStr = resultStr + str(rf.predict(tokenInfo))
                    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2)
                    rf2.fit(X_train2, y_train2)
                    resultStr = resultStr + " " + str(rf2.predict(tokenInfo))
                    streamlit.subheader(resultStr)

            streamlit.subheader(command)

connection.close()
