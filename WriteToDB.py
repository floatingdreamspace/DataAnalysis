import sqlite3

# Connecting to the database
connection = sqlite3.connect("DB.db")
cursor = connection.cursor()

file = open("2_16winners.txt", "r")
data = file.readlines()
for row in data:
    command = row
    cursor.execute(command)
    connection.commit()

connection.close()