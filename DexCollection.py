import sqlite3
import streamlit
import requests
import json

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
with streamlit.form("input_form"):
    pair = streamlit.text_input("Enter your Dexscreener Pair ID: ", key="pairId")
    submitted = streamlit.form_submit_button("Submit Form")

    if submitted:
        pairId = str(streamlit.session_state["pairId"])
        response = requests.get(
            "https://api.dexscreener.com/latest/dex/pairs/solana/" + pairId,
            headers={},
        )
        data = json.loads(response.json())
        streamlit.subheader(data)

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