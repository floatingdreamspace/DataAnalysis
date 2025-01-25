class Token:
    def __init__(self, buysM5, buysH1, sellsM5, sellsH1, volM5, volH1, priceM5, priceH1, liquidity, marketCap,
                 paidProfile, paidAd, websites, socials, boosts, pools, dayOfWeek, time, result, buysToSells,
                 volToLiquidity, volToMC, liquidityToMC, liquidityToBuys, MCToBuys, poolsToLiquidity, buysToVol):
        self.buysM5 = buysM5
        self.buysH1 = buysH1
        self.sellsM5 = sellsM5
        self.sellsH1 = sellsH1
        self.volM5 = volM5
        self.volH1 = volH1
        self.priceM5 = priceM5
        self.priceH1 = priceH1
        self.liquidity = liquidity
        self.marketCap = marketCap
        self.paidProfile = paidProfile
        self.paidAd = paidAd
        self.websites = websites
        self.socials = socials
        self.boosts = boosts
        self.pools = pools
        self.dayOfWeek = dayOfWeek
        self.time = time
        self.result = result
        self.buysToSells = buysToSells
        self.volToLiquidity = volToLiquidity
        self.volToMC = volToMC
        self.liquidityToMC = liquidityToMC
        self.liquidityToBuys = liquidityToBuys
        self.MCToBuys = MCToBuys
        self.poolsToLiquidity = poolsToLiquidity
        self.buysToVol = buysToVol
