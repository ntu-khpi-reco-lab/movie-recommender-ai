from flask import Flask
from app.routes.train_routes import train_routes
from app.routes.main_routes import main_routes
# from app.services.movie_trainer import MovieTrainer
# from app.services.movie_recommender import MovieRecommender
# from app.services.model_saver import MovieSaver
from app.utils.logger_config import get_logger

app = Flask(__name__)
logger = get_logger("Main")

app.register_blueprint(main_routes)
app.register_blueprint(train_routes)

def main():
    logger.info("Starting Flask server for movie recommender.")
    app.run(debug=True, host="0.0.0.0", port=5000)

    # logger.info("Initializing modules.")
    # trainer = MovieTrainer()
    # movies_df = trainer.load_movies_from_db()
    # recommender = MovieRecommender(movies_df)
    # saver = MovieSaver()
    #
    # logger.info("Loading movies and training similarity model.")
    # all_movies = trainer.load_movies_from_db()
    # similarity_matrix = trainer.train_model(all_movies)
    #
    # model_path = "../movie_similarity_model.pkl"
    # logger.info(f"Saving similarity model to {model_path}.")
    # saver.save_model(similarity_matrix, model_path)
    #
    # logger.info(f"Loading similarity model from {model_path}.")
    # loaded_similarity_matrix = saver.load_model(model_path)
    #
    # liked_movie_ids = [105, 680, 569094, 574]
    # logger.info(f"Generating recommendations for liked movies: {liked_movie_ids}.")
    # recommendations = recommender.generate_recommendations(liked_movie_ids, movies_df, loaded_similarity_matrix)
    #
    # logger.info("Displaying top 20 recommendations.")
    # recommendation_logs = "\n".join(
    #     [f"Movie ID: {rec['movie_id']}, Title: {rec['title']}, Similarity: {rec['average_similarity']:.2f}"
    #      for rec in recommendations[:20]]
    # )
    # logger.info(f"Top 20 Recommendations:\n{recommendation_logs}")
    #
    # for rec in recommendations[:20]:
    #     print(f"Movie ID: {rec['movie_id']}, Title: {rec['title']}, Similarity: {rec['average_similarity']:.2f}")
    #
    # logger.info("Movie recommendation pipeline completed successfully.")

if __name__ == "__main__":
    main()


#  Invoke-WebRequest -Uri http://127.0.0.1:5000/train -Method POST