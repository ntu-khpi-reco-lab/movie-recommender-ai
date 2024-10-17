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
    return response.json()


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


def compute_similarity_matrix(movies, all_genre_ids):
    feature_vectors = np.array([create_feature_vector(movie, all_genre_ids) for movie in movies])
    return cosine_similarity(feature_vectors)


def main():
    api_key = load_api_key()

    movie_ids = [550, 680, 100]
    movies = [get_movie_data(movie_id, api_key) for movie_id in movie_ids]

    all_genre_ids = extract_genres(movies)
    similarity_matrix = compute_similarity_matrix(movies, all_genre_ids)

    print("Cosine similarity matrix:")
    pprint(similarity_matrix)


if __name__ == "__main__":
    main()
