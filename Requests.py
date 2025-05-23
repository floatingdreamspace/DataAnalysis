import requests
import datetime
import pytz

pairId = input()
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

    command = ("INSERT INTO tokens VALUES('" + str(pairAddress) + "', '" + str(tokenAddress) + "', '" + str(buysM5) + "', '" +
                            str(buysH1) + "', '" + str(sellsM5) + "', '" + str(sellsH1) + "', '" + str(volM5) + "', '" + str(volH1) + "', '" +
                            str(priceM5) + "', '" + str(priceH1) + "', '" + str(liquidity) + "', '" + str(marketCap) + "', '" +
                            str(paidProfile) + "', '" + str(paidAd) + "', '" + str(websites) + "', '" + str(socials) + "', '" + str(boosts) + "', '" +
                            str(pools) + "', '" + str(dayOfWeek) + "', '" + str(current_minutes) + "', '" + str(score) + "', '" +
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
    tokenInfo.append([buysM5, buysH1, sellsM5, sellsH1, volM5, volH1, priceM5, priceH1, liquidity, marketCap, paidProfile,
                    paidAd, websites, socials, boosts, pools, dayOfWeek, current_minutes, buysToSells, volToLiquidity,
                    volToMC, liquidityToMC, liquidityToBuys, MCToBuys, poolsToLiquidity, buysToVol, score, highHolder,
                    lowLP, mutable, unlockedLP, singleHolder, highOwnership])