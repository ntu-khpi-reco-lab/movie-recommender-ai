import os
from dotenv import load_dotenv
import pandas as pd
from app.utils.logger_config import get_logger
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class MovieTrainer:
    """
    A class to train a movie recommendation model by loading movie data,
    processing it, and computing a similarity matrix using various movie features.
    """
    def __init__(self):
        """
        Initializes the MovieTrainer instance with a logger for tracking operations.
        """
        self.logger = get_logger("MovieTrainer")

    def load_movies_from_db(self):
        """
        Loads movie data from a MongoDB database.

        The data includes movie titles, genres, cast, crew, and keywords.

        :return: A pandas DataFrame containing movie data.
        """
        self.logger.info("Loading movies from MongoDB.")
        mongo_uri = os.getenv("MONGO_URI")
        client = MongoClient(mongo_uri)
        db = client["movieDB"]
        collection = db["movieDetails"]

        projection = {
            "_id": 1,
            "title": 1,
            "genres": 1,
            "cast": 1,
            "crew": 1,
            "keywords": 1
        }

        movies = list(collection.find({}, projection))
        self.logger.info(f"Loaded {len(movies)} movies from the database.")
        return pd.DataFrame(movies)

    def create_soup(self, movies_df):
        """
        Creates a 'soup' of features by combining genres, cast, keywords, and directors
        into a single text field for each movie.

        This will be used for feature extraction to compute the movie similarity.

        :param movies_df: DataFrame containing movie information.
        :return: A pandas Series containing the combined feature strings (soup) for each movie.
        """
        self.logger.info("Creating soup for feature extraction.")
        genres = movies_df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        cast = movies_df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        keywords = movies_df['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        directors = movies_df['crew'].apply(
            lambda x: [i['name'] for i in x if isinstance(i, dict) and i.get("job") == "Director"]
            if isinstance(x, list) else []
        )

        movies_df['soup'] = (genres + cast + keywords + directors).apply(lambda x: " ".join(x))
        self.logger.info(f"Soup created for {len(movies_df)} movies.")

        return movies_df['soup']

    def prepare_vectors(self, movies_df):
        """
        Prepares feature vectors from the 'soup' text data using CountVectorizer.

        This transforms the soup (text data) into a numerical matrix that represents the movie features.

        :param movies_df: DataFrame containing movie information.
        :return: A sparse matrix representing the feature vectors of all movies.
        """
        self.logger.info("Preparing feature vectors.")
        movies_df = movies_df.copy()
        vectorizer = CountVectorizer(stop_words='english')
        soup = self.create_soup(movies_df)
        feature_matrix = vectorizer.fit_transform(soup)
        self.logger.info("Feature vectors prepared.")

        return feature_matrix

    def compute_similarity(self, all_movies):
        """
        Computes the cosine similarity matrix based on feature vectors.

        Cosine similarity measures the similarity between movies by comparing their feature vectors.

        :param all_movies: DataFrame containing movie data.
        :return: A 2D array representing the cosine similarity between all pairs of movies.
        """
        self.logger.info("Computing cosine similarity matrix.")
        similarity_matrices = self.prepare_vectors(all_movies)
        cosine_sim = cosine_similarity(similarity_matrices)
        self.logger.info("Cosine similarity matrix computation completed.")

        return cosine_sim

    def train_model(self, all_movies):
        """
        Trains the movie recommendation model by computing the similarity matrix.

        This method orchestrates the process of preparing feature vectors and computing the
        cosine similarity matrix which represents movie similarities.

        :param all_movies: DataFrame containing movie data.
        :return: The cosine similarity matrix representing the similarity between all movies.
        """
        self.logger.info("Training model by computing similarity matrix.")
        similarity_matrix = self.compute_similarity(all_movies)
        self.logger.info("Model training completed. Similarity matrix is ready.")
        return similarity_matrix
