import sqlite3
from config import DATABASE_PATH

connection = sqlite3.connect(DATABASE_PATH)

cursor = connection.cursor()

cursor.execute("""
ALTER TABLE candidates
ADD COLUMN photo_path TEXT
""")

connection.commit()

connection.close()

print("photo_path column added successfully!")