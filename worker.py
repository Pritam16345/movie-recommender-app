# worker.py
import requests
import sqlite3
import pickle
import time
import json

print("Starting worker...")

# --- Load the saved CountVectorizer ---
try:
    with open('vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    print("Vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: vectorizer.pkl not found. Please run database_setup.py first.")
    exit()

# --- Helper functions (must be identical to setup script) ---
def parse_json_features(features_list):
    return [feature['name'] for feature in features_list]

def get_top_3_actors(cast_list):
    return [actor['name'] for actor in cast_list[:3]]

def get_director(crew_list):
    for member in crew_list:
        if member['job'] == 'Director':
            return [member['name']]
    return []

def create_tags(row):
    tags = []
    if isinstance(row.get('overview'), str):
        tags.extend(row['overview'].split())
    for feature_list_key in ['genres', 'keywords', 'cast', 'crew']:
        if feature_list_key in row:
             tags.extend([item.replace(" ", "") for item in row[feature_list_key]])
    return " ".join(tags).lower()

# --- Main Worker Function ---
def update_database():
    print("\nChecking for new movies...")
    # IMPORTANT: Add your TMDB API Key here
    api_key = "4da5be8fd17a596eccb86cb093998ec2"
    
    # Fetch a list of popular movies (a good source for new releases)
    url = f"https://api.themoviedb.org/3/movie/popular?api_key={api_key}&language=en-US&page=1"
    
    try:
        popular_movies = requests.get(url).json()['results']
    except Exception as e:
        print(f"Could not fetch popular movies: {e}")
        return

    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()

    for movie in popular_movies:
        tmdb_id = movie['id']
        
        # Check if the movie already exists in our database
        cursor.execute("SELECT * FROM movies WHERE tmdb_id = ?", (tmdb_id,))
        if cursor.fetchone():
            continue # Skip if it already exists

        # If it's a new movie, fetch its full details
        print(f"New movie found: '{movie['title']}'. Fetching details...")
        details_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&append_to_response=credits,keywords"
        
        try:
            details = requests.get(details_url).json()
            
            # Process the details to create tags
            processed_data = {
                'overview': details.get('overview', ''),
                'genres': parse_json_features(details.get('genres', [])),
                'keywords': parse_json_features(details.get('keywords', {}).get('keywords', [])),
                'cast': get_top_3_actors(details.get('credits', {}).get('cast', [])),
                'crew': get_director(details.get('credits', {}).get('crew', []))
            }

            tags = create_tags(processed_data)
            
            # Use the loaded vectorizer to transform the new tags
            vector = cv.transform([tags]).toarray()[0]
            pickled_vector = pickle.dumps(vector)
            
            # Insert the new movie into the database
            cursor.execute("INSERT INTO movies (tmdb_id, title, vector) VALUES (?, ?, ?)",
                           (tmdb_id, details['title'], pickled_vector))
            conn.commit()
            print(f"Successfully added '{details['title']}' to the database.")

        except Exception as e:
            print(f"Failed to process movie ID {tmdb_id}: {e}")

    conn.close()

# --- Main Loop ---
if __name__ == "__main__":
    while True:
        update_database()
        # Sleep for 6 hours (21600 seconds) before checking again
        sleep_duration = 6 * 60 * 60 
        print(f"Worker sleeping for {sleep_duration / 3600} hours...")
        time.sleep(sleep_duration)
