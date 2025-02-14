import sqlite3

# Connecting to the database
connection = sqlite3.connect("DB2.db")
cursor = connection.cursor()

file = open("2_13.txt", "r")
data = file.readlines()
for row in data:
    command = row
    cursor.execute(command)
    connection.commit()

connection.close()