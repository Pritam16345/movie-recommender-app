# app.py
from flask import Flask, render_template, request
import sqlite3
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)
api_session = requests.Session() # Using a session for poster fetching

# --- Helper Function to Fetch Movie Posters (No change) ---
def fetch_poster(movie_id):
    api_key = "4da5be8fd17a596eccb86cb093998ec2" # Your API key
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        response = api_session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except requests.exceptions.RequestException as e:
        print(f"API request failed for movie_id {movie_id}: {e}")
    return "https://placehold.co/500x750/1e1e1e/e0e0e0?text=Poster+Not+Found"

# --- Database Connection Function ---
def get_db_connection():
    conn = sqlite3.connect('movies.db')
    conn.row_factory = sqlite3.Row # This allows accessing columns by name
    return conn

# --- Define the routes ---
@app.route('/')
def home():
    conn = get_db_connection()
    # Fetch all movie titles, sorted alphabetically
    movie_titles = conn.execute('SELECT title FROM movies ORDER BY title ASC').fetchall()
    conn.close()
    # Extract just the title string from each row
    titles = [row['title'] for row in movie_titles]
    return render_template('index.html', movie_titles=titles)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    movie_title = request.form.get('movie')
    
    conn = get_db_connection()
    
    # Fetch all movies from the database
    all_movies_cursor = conn.execute('SELECT tmdb_id, title, vector FROM movies').fetchall()
    
    # Find the selected movie and its vector
    selected_movie = None
    for movie in all_movies_cursor:
        if movie['title'] == movie_title:
            selected_movie = movie
            break
            
    if not selected_movie:
        conn.close()
        movie_titles = [row['title'] for row in all_movies_cursor]
        return render_template('index.html', movie_titles=sorted(movie_titles), error=f"Movie '{movie_title}' not found.")

    # Unpickle all vectors and create a matrix
    all_vectors = [pickle.loads(movie['vector']) for movie in all_movies_cursor]
    vector_matrix = np.array(all_vectors)
    
    # Unpickle the selected movie's vector
    selected_vector = pickle.loads(selected_movie['vector']).reshape(1, -1)
    
    # Calculate similarity on the fly
    similarity_scores = cosine_similarity(selected_vector, vector_matrix)
    
    # Get top 5 recommendations
    similar_movies_indices = similarity_scores[0].argsort()[-6:-1][::-1]
    
    recommended_movies_data = []
    for index in similar_movies_indices:
        rec_movie = all_movies_cursor[index]
        recommended_movies_data.append({
            "title": rec_movie['title'],
            "poster": fetch_poster(rec_movie['tmdb_id']),
            "id": rec_movie['tmdb_id']
        })
        
    conn.close()
    movie_titles = [row['title'] for row in all_movies_cursor]
    return render_template('index.html', 
                           movie_titles=sorted(movie_titles), 
                           recommendations=recommended_movies_data,
                           selected_movie=movie_title)

if __name__ == '__main__':
    app.run(debug=True)
