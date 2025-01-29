import sqlite3
from sqlite3 import Error


def create_database(file):
    connection = None
    try:
        connection = sqlite3.connect(file)
    except Error as e:
        print(e)
    cursor = connection.cursor()

    cursor.execute("""DROP TABLE tokens""")

    cursor.execute("""CREATE TABLE tokens
                        (pairAddress TEXT, tokenAddress TEXT, buysM5 INTEGER, buysH1 INTEGER, sellsM5 INTEGER, sellsH1 INTEGER,
                        volM5 REAL, volH1 REAL, priceM5 REAL, priceH1 REAL, liquidity REAL, marketCap REAL, paidProfile INTEGER,
                        paidAd INTEGER, websites INTEGER, socials INTEGER, boosts INTEGER, pools INTEGER, dayOfWeek INTEGER, 
                        time INTEGER, result TEXT, score INTEGER, highHolder INTEGER, lowLP INTEGER, mutable INTEGER, 
                        unlockedLP INTEGER, singleHolder INTEGER, highOwnership INTEGER)""")
    if connection:
        connection.close()


if __name__ == '__main__':
    create_database("DB.db")