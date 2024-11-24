from movie_trainer import MovieTrainer
from movie_recommender import MovieRecommender
from model_saver import MovieSaver


def main():
    trainer = MovieTrainer()
    recommender = MovieRecommender()
    saver = MovieSaver()

    all_movies = trainer.load_movies_from_db()
    similarity_matrix = trainer.train_model(all_movies)

    model_path = "movie_similarity_model.pkl"
    saver.save_model(similarity_matrix, model_path)

    loaded_similarity_matrix = saver.load_model(model_path)
    recommender.prepare_data(all_movies)

    liked_movie_ids = [105, 680, 569094, 574]
    recommendations = recommender.generate_recommendations(liked_movie_ids, loaded_similarity_matrix)

    for rec in recommendations[:20]:
        print(f"Movie ID: {rec['movie_id']}, Title: {rec['title']}, Similarity: {rec['average_similarity']:.2f}")

if __name__ == "__main__":
    main()
