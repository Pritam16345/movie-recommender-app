# database_setup.py
import pandas as pd
import json
import sqlite3
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

print("Starting database setup...")

# --- Step 1: Load and process the original dataset (same as before) ---
credits_df = pd.read_csv("tmdb_5000_credits.csv")
movies_df = pd.read_csv("tmdb_5000_movies.csv")
movies_df = movies_df.merge(credits_df, on="title")
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Helper functions to parse JSON-like strings
def parse_json_features(features_str):
    try:
        return [feature['name'] for feature in json.loads(features_str)]
    except: return []

def get_top_3_actors(cast_str):
    try:
        return [actor['name'] for actor in json.loads(cast_str)[:3]]
    except: return []

def get_director(crew_str):
    try:
        for member in json.loads(crew_str):
            if member['job'] == 'Director':
                return [member['name']]
        return []
    except: return []

for feature in ['genres', 'keywords']:
    movies_df[feature] = movies_df[feature].apply(parse_json_features)
movies_df['cast'] = movies_df['cast'].apply(get_top_3_actors)
movies_df['crew'] = movies_df['crew'].apply(get_director)

# Create the 'tags' column
def create_tags(row):
    tags = []
    if isinstance(row['overview'], str):
        tags.extend(row['overview'].split())
    for feature_list in ['genres', 'keywords', 'cast', 'crew']:
        tags.extend([item.replace(" ", "") for item in row[feature_list]])
    return " ".join(tags).lower()

movies_df['tags'] = movies_df.apply(create_tags, axis=1)
model_df = movies_df[['movie_id', 'title', 'tags']].dropna()

# --- Step 2: Vectorize the data and save the vectorizer ---
print("Vectorizing movie tags...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(model_df['tags']).toarray()

# CRITICAL: Save the CountVectorizer object to a file.
# The worker will need this exact object to create consistent vectors for new movies.
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)
print("Vectorizer saved to vectorizer.pkl")

# --- Step 3: Create SQLite database and table ---
db_file = 'movies.db'
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create the table schema
cursor.execute('''
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id INTEGER UNIQUE,
    title TEXT NOT NULL,
    vector BLOB NOT NULL
)
''')
print(f"Database '{db_file}' and table 'movies' created/ensured.")

# --- Step 4: Insert all movies into the database ---
print("Inserting movies into the database. This may take a minute...")
for index, row in model_df.iterrows():
    tmdb_id = row['movie_id']
    title = row['title']
    # Get the corresponding vector and pickle it for storage
    vector = pickle.dumps(vectors[index])
    
    # Use INSERT OR IGNORE to prevent errors if a tmdb_id is duplicated
    cursor.execute("INSERT OR IGNORE INTO movies (tmdb_id, title, vector) VALUES (?, ?, ?)", (tmdb_id, title, vector))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database setup complete! 'movies.db' is ready.")
