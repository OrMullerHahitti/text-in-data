import sqlite3
import pandas as pd
from searches import search_posts
conn = sqlite3.connect('data_for_text.db')
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS peptides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL)''')
peptides = []

# Query the data from the peptides table into a DataFrame
df = pd.read_sql_query("SELECT * FROM peptides", conn)
# Display the DataFrame
print(df)
# Close the database connection
conn.close()