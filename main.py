from movie_trainer import MovieTrainer
from movie_recommender import MovieRecommender
from model_saver import MovieSaver
from logger_config import get_logger


def main():
    logger = get_logger("Main")
    logger.info("Starting the movie recommendation pipeline.")

    logger.info("Initializing modules.")
    trainer = MovieTrainer()
    movies_df = trainer.load_movies_from_db()
    recommender = MovieRecommender(movies_df)
    saver = MovieSaver()

    logger.info("Loading movies and training similarity model.")
    all_movies = trainer.load_movies_from_db()
    similarity_matrix = trainer.train_model(all_movies)

    model_path = "movie_similarity_model.pkl"
    logger.info(f"Saving similarity model to {model_path}.")
    saver.save_model(similarity_matrix, model_path)

    logger.info(f"Loading similarity model from {model_path}.")
    loaded_similarity_matrix = saver.load_model(model_path)

    liked_movie_ids = [105, 680, 569094, 574]
    logger.info(f"Generating recommendations for liked movies: {liked_movie_ids}.")
    recommendations = recommender.generate_recommendations(liked_movie_ids, movies_df, loaded_similarity_matrix)

    logger.info("Displaying top 20 recommendations.")
    recommendation_logs = "\n".join(
        [f"Movie ID: {rec['movie_id']}, Title: {rec['title']}, Similarity: {rec['average_similarity']:.2f}"
         for rec in recommendations[:20]]
    )
    logger.info(f"Top 20 Recommendations:\n{recommendation_logs}")

    for rec in recommendations[:20]:
        print(f"Movie ID: {rec['movie_id']}, Title: {rec['title']}, Similarity: {rec['average_similarity']:.2f}")

    logger.info("Movie recommendation pipeline completed successfully.")

if __name__ == "__main__":
    main()
