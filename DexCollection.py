import sqlite3
import streamlit
import requests
import datetime
import pytz
import csv

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
    pair = streamlit.text_input("Enter your Dexscreener Pair ID: ", key="pairId")
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
            streamlit.subheader("pools: " + str(pools))

            command = ("INSERT INTO tokens VALUES('" + str(pairAddress) + "', '" + str(tokenAddress) + "', '" + str(buysM5) + "', '" +
                                    str(buysH1) + "', '" + str(sellsM5) + "', '" + str(sellsH1) + "', '" + str(volM5) + "', '" + str(volH1) + "', '" +
                                    str(priceM5) + "', '" + str(priceH1) + "', '" + str(liquidity) + "', '" + str(marketCap) + "', '" +
                                    str(paidProfile) + "', '" + str(paidAd) + "', '" + str(websites) + "', '" + str(socials) + "', '" + str(boosts) + "', '" +
                                    str(pools) + "', '" + str(dayOfWeek) + "', '" + str(current_minutes) + "', 'default')")
            streamlit.subheader(command)
            cursor.execute(command)
            connection.commit()
            with open('tokens.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([str(pairAddress), str(tokenAddress), str(buysM5), str(buysH1), str(sellsM5), str(sellsH1),
                                 str(volM5), str(volH1), str(paidProfile), str(paidAd), str(websites), str(socials), str(boosts),
                                 str(pools), str(dayOfWeek), str(current_minutes), "defaults"])
            cursor.execute("SELECT * FROM tokens")
            tokens = cursor.fetchall()
            #for row in tokens:
            #    if row[19] == "default":
            #        streamlit.subheader(row[0])

with streamlit.form("result_form"):
    resultPair = streamlit.text_input("Enter your Dexscreener Pair ID: ", key="resultPairId")
    result = streamlit.radio(
        "Select coin result:",
        [":Success", ":Failure"],
        captions=[
            "Success",
            "Failure",
        ],
        horizontal=True,
    )

    submittedb = streamlit.form_submit_button("Submit Form")

    if submittedb:
        if result == ":Success":
            streamlit.subheader("Success")

connection.close()