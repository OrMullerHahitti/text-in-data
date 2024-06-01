import sqlite3
from searches import search_posts
conn = sqlite3.connect('data_for_text.db')
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS peptides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL)''')
peptides = []