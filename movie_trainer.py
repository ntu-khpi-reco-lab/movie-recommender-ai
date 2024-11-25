import os
from dotenv import load_dotenv
import pandas as pd
from logger_config import get_logger
from pymongo import MongoClient, errors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class MovieTrainer:
    def __init__(self):
        self.logger = get_logger("MovieTrainer")

    def load_movies_from_db(self):
        self.logger.info("Loading movies from MongoDB.")
        try:
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
        except errors.ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB: {str(e)}", exc_info=True)
        except Exception as e:
            self.logger.error(f"An error occurred while loading movies from the database: {str(e)}", exc_info=True)

        return pd.DataFrame()

    def create_soup(self, movies_df):
        try:
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
        except KeyError as e:
            self.logger.error(f"Missing expected column in DataFrame: {str(e)}", exc_info=True)
        except Exception as e:
            self.logger.error(f"An error occurred while creating soup: {str(e)}", exc_info=True)

        return pd.Series()

    def prepare_vectors(self, movies_df):
        try:
            self.logger.info("Preparing feature vectors.")
            movies_df = movies_df.copy()
            vectorizer = CountVectorizer(stop_words='english')
            soup = self.create_soup(movies_df)
            feature_matrix = vectorizer.fit_transform(soup)
            self.logger.info("Feature vectors prepared.")
            return feature_matrix
        except ValueError as e:
            self.logger.error(f"Error during vectorization: {str(e)}", exc_info=True)
        except Exception as e:
            self.logger.error(f"An error occurred while preparing vectors: {str(e)}", exc_info=True)

        return None

    def compute_similarity(self, all_movies):
        try:
            self.logger.info("Computing cosine similarity matrix.")
            similarity_matrices = self.prepare_vectors(all_movies)
            if similarity_matrices is not None:
                cosine_sim = cosine_similarity(similarity_matrices)
                self.logger.info("Cosine similarity matrix computation completed.")
                return cosine_sim
            else:
                self.logger.error("Feature vectors could not be prepared. Similarity computation aborted.")
        except Exception as e:
            self.logger.error(f"An error occurred while computing similarity matrix: {str(e)}", exc_info=True)

        return None

    def train_model(self, all_movies):
        self.logger.info("Training model by computing similarity matrix.")
        try:
            similarity_matrix = self.compute_similarity(all_movies)
            if similarity_matrix is not None:
                self.logger.info("Model training completed. Similarity matrix is ready.")
                return similarity_matrix
            else:
                self.logger.error("Model training failed due to previous errors.")
        except Exception as e:
            self.logger.error(f"An error occurred during model training: {str(e)}", exc_info=True)

        return None
