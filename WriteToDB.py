import sqlite3

# Connecting to the database
connection = sqlite3.connect("DBR_test.db")
cursor = connection.cursor()

file = open("3_7Rwins_test.txt", "r")
data = file.readlines()
for row in data:
    command = row
    cursor.execute(command)
    connection.commit()

connection.close()