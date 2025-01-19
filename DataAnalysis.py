import pandas as pd
import numpy as np

# 1) Load your data
df = pd.read_csv("Dataset.csv")  # Adapt columns as needed

# 2) Create new features
df['NetBuySellCount'] = df['Buys'] - df['Sells']
df['NetBuySellVolRatio'] = (df['BuyVol'] - df['SellVol']) / (df['BuyVol'] + df['SellVol']).clip(lower=1)
df['BuyersPerSellers'] = (df['Buyers'].clip(lower=1)) / (df['Sellers'].clip(lower=1))
df['MCL'] = df['MarketCap'] / df['Liquidity'].clip(lower=1)
df['VOT'] = df['Volume'] / df['TXNs'].clip(lower=1)
df['PriceDelta'] = df['Price5M'] - df['Price1H']
df['PriceProduct'] = df['Price5M'] * df['Price1H']
# etc., for any other combos

# Optionally convert "Result" to 1 for success, 0 for failure
df['Label'] = (df['Result'] == 'Success').astype(int)

# 3) Test thresholds for one new feature
feature_name = 'MCL'  # for example
best_acc = 0
best_thr = None

vals = sorted(df[feature_name].unique())
for i in range(len(vals) - 1):
    # We'll pick a threshold between vals[i] and vals[i+1]
    thr = (vals[i] + vals[i + 1]) / 2.0

    # Predict success if feature > thr
    predictions = (df[feature_name] > thr).astype(int)
    # Compare to actual label
    correct = (predictions == df['Label']).sum()
    acc = correct / len(df)

    if acc > best_acc:
        best_acc = acc
        best_thr = thr

print(f"Best threshold for {feature_name} was {best_thr}, accuracy = {best_acc * 100:.2f}%")
