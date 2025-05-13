import pandas as pd
import sqlite3

# Load the CSV
df = pd.read_csv("tcc_ceds_music.csv").dropna()
df['release_date'] = df['release_date'].astype(int)
df['era'] = df['release_date'].apply(lambda y: f"{(y // 10) * 10}s")

# Save to SQLite
conn = sqlite3.connect("music.db")
df.to_sql("songs", conn, if_exists="replace", index=False)
conn.close()
