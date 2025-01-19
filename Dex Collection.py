import subprocess
subprocess.check_call(["pip", "install", "requests"])
import requests

pairId = "GNfZPYyhxmvEracmrKJ87HBJy4VPMf1h5ojDPitRqeqv"
response = requests.get(
    "https://api.dexscreener.com/latest/dex/pairs/solana/" + pairId,
    headers={},
)
data = response.json()

print(data)