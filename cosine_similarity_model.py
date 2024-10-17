import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from pprint import pprint


def load_api_key():
    """
    Loads the API key from the .env file using the python-dotenv library.

    This function calls `load_dotenv` to load environment variables from the .env file
    and retrieves the value of the "API_KEY" variable, which should be defined in the file.

    :return: The API key as a string, or None if the "API_KEY" variable is not found.
    """
    load_dotenv()
    return os.getenv("API_KEY")


def get_movie_data(movie_id, api_key):
    """
    Retrieves detailed data for a specific movie from the TMDb API.

    This function sends a GET request to The Movie Database (TMDb) API to fetch information
    about a movie, given its movie ID and an API key. The data is returned as a JSON object.

    :param movie_id: The unique ID of the movie to retrieve.
    :param api_key: The API key for authenticating the request to the TMDb API.
    :return: A JSON response containing detailed information about the movie, including title, genres, ratings, etc.
    """
    base_url = "https://api.themoviedb.org/3"
    url = f"{base_url}/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    # pprint(response.json())
    return response.json()


def get_movies_id(api_key, page):
    """
    Retrieves a list of popular movies from the TMDb API.

    This function sends a GET request to The Movie Database (TMDb) API to fetch a list of
    popular movies for a specified page. It returns the results as a list of movie dictionaries.

    :param api_key: The API key for authenticating the request to the TMDb API.
    :param page: The page number to retrieve results from, for pagination.
    :return: A list of dictionaries containing information about popular movies.
    """
    base_url = "https://api.themoviedb.org/3"
    url = f"{base_url}/movie/popular?api_key={api_key}&language=en-US&page={page}"
    response = requests.get(url)
    return response.json().get('results', [])


def extract_genres(movies):
    """
    Extracts unique genre IDs from a list of movies.

    This function takes a list of movie dictionaries and extracts all unique genre IDs present
    in the movies. The genre IDs are collected in a set to ensure uniqueness and are returned
    as a list.

    :param movies: A list of movie dictionaries containing genre information.
    :return: A list of unique genre IDs found in the provided movies.
    """
    all_genre_ids = set()
    for movie in movies:
        genres = movie.get('genres', [])
        for genre in genres:
            all_genre_ids.add(genre['id'])
    return list(all_genre_ids)


def create_feature_vector(movie, all_genre_ids):
    """
    Creates a feature vector for a movie based on its genres and rating.

    This function generates a binary feature vector representing the genres of the given movie,
    along with its average rating. The feature vector consists of a binary indicator for each
    genre (1 if the movie belongs to that genre, 0 otherwise) followed by the movie's average
    rating.

    :param movie: A dictionary containing information about the movie, including its genres and
                  average rating.
    :param all_genre_ids: A list of all unique genre IDs to create the binary genre vector.
    :return: A NumPy array representing the feature vector for the movie, which includes the
             binary genre indicators followed by the average rating.
    """
    genres = movie.get('genres', [])
    genre_vector = np.zeros(len(all_genre_ids))
    for genre in genres:
        if genre['id'] in all_genre_ids:
            genre_vector[all_genre_ids.index(genre['id'])] = 1
    rating = movie.get('vote_average', 0)
    return np.concatenate((genre_vector, [rating]))


def compute_similarity(selected_movies, all_movies, all_genre_ids):
    """
    Computes the cosine similarity between selected movies and all movies.

    This function generates feature vectors for both the selected movies and all movies, then
    calculates the cosine similarity between the selected movies and all other movies. The
    result is a similarity matrix where each row corresponds to a selected movie and each
    column corresponds to a movie from the list of all movies.

    :param selected_movies: A list of movie dictionaries for which similarities are being computed.
    :param all_movies: A list of all movie dictionaries to compare against the selected movies.
    :param all_genre_ids: A list of all unique genre IDs used to create feature vectors.
    :return: A 2D NumPy array representing the cosine similarity matrix, where the rows
             correspond to the selected movies and the columns correspond to all movies.
    """
    # Create a feature vector for all films
    all_movie_vectors = np.array([create_feature_vector(movie, all_genre_ids) for movie in all_movies])

    # Create a feature vector for the selected films
    selected_vectors = np.array([create_feature_vector(movie, all_genre_ids) for movie in selected_movies])

    similarity_matrix = cosine_similarity(selected_vectors, all_movie_vectors)
    return similarity_matrix


def print_similar_movies(api_key, selected_movie_ids, similarity_matrix, all_movie_ids, movie_titles):
    """
    Prints the titles and similarity scores of movies similar to the selected movies.

    This function retrieves the title of each selected movie, computes its similarity scores
    against all other movies, and then prints the top similar movies in a readable format.

    :param api_key: The API key used to access the movie database API.
    :param selected_movie_ids: A list of movie IDs for which similar movies will be found.
    :param similarity_matrix: A 2D array where each row contains similarity scores for a selected movie against all other movies.
    :param all_movie_ids: A list of all movie IDs to which the selected movies are compared.
    :param movie_titles: A dictionary mapping movie IDs to their titles.
    :return: None
    """
    for i, selected_movie_id in enumerate(selected_movie_ids):
        selected_movie_title = get_movie_data(selected_movie_id, api_key)['title']
        print(f"Selected movie ID: {selected_movie_id}, Title: {selected_movie_title}")
        similarity_scores = similarity_matrix[i]
        similar_movies = sorted(zip(all_movie_ids, similarity_scores), key=lambda x: x[1], reverse=True)
        print("Top similar movies:")
        for movie_id, score in similar_movies:
            print(f"Movie ID: {movie_id}, Title: {movie_titles[movie_id]} Similarity: {score:.2f}")
        print()


def main():
    """
    The main function that orchestrates the movie similarity computation process.

    This function loads the API key, retrieves a list of popular movies from the TMDb API,
    and fetches details for selected movies. It then extracts genre information, computes
    cosine similarity between the selected movies and all other movies, and prints the
    similarity matrix along with the top similar movies for each selected movie.

    It performs the following steps:
    1. Loads the API key from the environment.
    2. Defines a list of selected movie IDs.
    3. Retrieves a list of popular movies and their IDs.
    4. Creates a dictionary mapping movie IDs to movie titles.
    5. Fetches details for the selected movies and all movies.
    6. Extracts unique genre IDs from the list of all movies.
    7. Computes the cosine similarity matrix for the selected movies against all other movies.
    8. Prints the similarity matrix and lists the top similar movies for each selected movie.

    :return: None
    """
    api_key = load_api_key()

    selected_movie_ids = [105, 680]

    movies_id = get_movies_id(api_key, 1)
    all_movie_ids = [movie['id'] for movie in movies_id]

    movie_titles = {movie['id']: movie['title'] for movie in movies_id}

    selected_movies = [get_movie_data(movie_id, api_key) for movie_id in selected_movie_ids]
    all_movies = [get_movie_data(movie_id, api_key) for movie_id in all_movie_ids]

    all_genre_ids = extract_genres(all_movies)

    # Calculate the cosine similarity between the selected films and all others
    similarity_matrix = compute_similarity(selected_movies, all_movies, all_genre_ids)

    print("Cosine similarity between selected movies and all movies:")
    pprint(similarity_matrix)
    print()

    print_similar_movies(api_key, selected_movie_ids, similarity_matrix, all_movie_ids, movie_titles)


if __name__ == "__main__":
    main()
