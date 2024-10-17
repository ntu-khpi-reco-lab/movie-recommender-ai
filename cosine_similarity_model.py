import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from pprint import pprint


def load_api_key():
    load_dotenv()
    return os.getenv("API_KEY")


def get_movie_data(movie_id, api_key):
    base_url = "https://api.themoviedb.org/3"
    url = f"{base_url}/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    # pprint(response.json())
    return response.json()


def get_movies_id(api_key, page):
    base_url = "https://api.themoviedb.org/3"
    url = f"{base_url}/movie/popular?api_key={api_key}&language=en-US&page={page}"
    response = requests.get(url)
    return response.json().get('results', [])


def extract_genres(movies):
    all_genre_ids = set()
    for movie in movies:
        genres = movie.get('genres', [])
        for genre in genres:
            all_genre_ids.add(genre['id'])
    return list(all_genre_ids)


def create_feature_vector(movie, all_genre_ids):
    genres = movie.get('genres', [])
    genre_vector = np.zeros(len(all_genre_ids))
    for genre in genres:
        if genre['id'] in all_genre_ids:
            genre_vector[all_genre_ids.index(genre['id'])] = 1
    rating = movie.get('vote_average', 0)
    return np.concatenate((genre_vector, [rating]))


def compute_similarity(selected_movies, all_movies, all_genre_ids):
    # Create a feature vector for all films
    all_movie_vectors = np.array([create_feature_vector(movie, all_genre_ids) for movie in all_movies])

    # Create a feature vector for the selected films
    selected_vectors = np.array([create_feature_vector(movie, all_genre_ids) for movie in selected_movies])

    similarity_matrix = cosine_similarity(selected_vectors, all_movie_vectors)
    return similarity_matrix


def main():
    api_key = load_api_key()

    selected_movie_ids = [550, 680]
    movies_id = get_movies_id(api_key, 1)
    all_movie_ids = [movie['id'] for movie in movies_id]

    selected_movies = [get_movie_data(movie_id, api_key) for movie_id in selected_movie_ids]
    all_movies = [get_movie_data(movie_id, api_key) for movie_id in all_movie_ids]

    all_genre_ids = extract_genres(all_movies)

    # Calculate the cosine similarity between the selected films and all others
    similarity_matrix = compute_similarity(selected_movies, all_movies, all_genre_ids)

    print("Cosine similarity between selected movies and all movies:")
    pprint(similarity_matrix)

    for i, selected_movie_id in enumerate(selected_movie_ids):
        print(f"Selected movie ID: {selected_movie_id}")
        similarity_scores = similarity_matrix[i]
        similar_movies = sorted(zip(all_movie_ids, similarity_scores), key=lambda x: x[1], reverse=True)
        print("Top similar movies:")
        for movie_id, score in similar_movies:
            print(f"Movie ID: {movie_id}, Similarity: {score:.2f}")
        print()


if __name__ == "__main__":
    main()
