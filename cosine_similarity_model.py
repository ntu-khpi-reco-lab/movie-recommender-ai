import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
from collections import Counter
import logging

def setup_logger():
    logger = logging.getLogger('logfile.log')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def load_movies_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def extract_genres(movies):
    all_genre_ids = set()
    for movie in movies:
        details = movie.get('detail', [])
        for detail in details:
            genres = detail.get('genres', [])
            for genre in genres:
                all_genre_ids.add(genre['id'])
    return list(all_genre_ids)


def extract_actors(movies):
    all_actors = set()
    for movie in movies:
        credits = movie.get('credit', [])
        for credit in credits:
            cast = credit.get('cast', [])
            for actor in cast:
                all_actors.add(actor['name'])
    return list(all_actors)


def extract_keywords(movies):
    all_keywords = set()
    for movie in movies:
        if 'keyword' in movie:
            for keyword_group in movie['keyword']:
                if 'keywords' in keyword_group:
                    for keyword in keyword_group['keywords']:
                        all_keywords.add(keyword['name'])
    return list(all_keywords)


def create_feature_vector(movie, all_genre_ids):
    detail = movie.get('detail', [])
    if not detail:
        return np.zeros(len(all_genre_ids) + 1)

    genres = detail[0].get('genres', [])
    genre_vector = np.zeros(len(all_genre_ids))

    for genre in genres:
        if genre['id'] in all_genre_ids:
            genre_vector[all_genre_ids.index(genre['id'])] = 1
    rating = detail[0].get('vote_average', 0)

    return np.concatenate((genre_vector, [rating]))


def prepare_vectors(movies, all_genre_ids, all_actors, all_keywords, main_case, vector_by):
    vectors = []
    for i, movie in enumerate(movies):
        if i % 100 == 0:  # For log every 100th movie
            logger.info(f"Processing movie {i + 1}/{len(movies)}")
        if main_case == 'detail':
            details = movie.get('detail', [])
            for detail in details:
                genres = detail.get(vector_by, [])
                movie_genre_vector = [1 if genre_id in [g['id'] for g in genres] else 0 for genre_id in all_genre_ids]
                vectors.append(movie_genre_vector)
        elif main_case == 'credit':
            credits = movie.get('credit', [])
            for credit in credits:
                cast = credit.get(vector_by, [])
                movie_cast_vector = [1 if cast_name in [c['name'] for c in cast] else 0 for cast_name in all_actors]
                vectors.append(movie_cast_vector)
        elif main_case == 'keyword':
            keywords = movie.get('keyword', [])
            for keyword in keywords:
                keyword_item = keyword.get(vector_by, [])
                movie_keyword_vector = [1 if kw_name in [k['name'] for k in keyword_item] else 0 for kw_name in all_keywords]
                vectors.append(movie_keyword_vector)
    return vectors


def compute_similarity(selected_movies, all_movies, all_genre_ids, all_actors, all_keywords):
    logger.info('START genre_vectors')
    selected_genre_vectors = prepare_vectors(selected_movies, all_genre_ids, all_actors, all_keywords, 'detail', 'genres')
    all_genre_vectors = prepare_vectors(all_movies, all_genre_ids, all_actors, all_keywords, 'detail', 'genres')
    logger.info('END genre_vectors')

    logger.info('START actor_vectors')
    selected_actor_vectors = prepare_vectors(selected_movies, all_genre_ids, all_actors, all_keywords, 'credit', 'cast')
    all_actor_vectors = prepare_vectors(all_movies, all_genre_ids, all_actors, all_keywords, 'credit', 'cast')
    logger.info('END actor_vectors')

    logger.info('START keyword_vectors')
    selected_keyword_vectors = prepare_vectors(selected_movies, all_genre_ids, all_actors, all_keywords, 'keyword', 'keywords')
    all_keyword_vectors = prepare_vectors(all_movies, all_genre_ids, all_actors, all_keywords, 'keyword', 'keywords')
    logger.info('END keyword_vectors')

    genre_similarity = cosine_similarity(selected_genre_vectors, all_genre_vectors)
    actor_similarity = cosine_similarity(selected_actor_vectors, all_actor_vectors)
    keyword_similarity = cosine_similarity(selected_keyword_vectors, all_keyword_vectors)

    combined_similarity = (genre_similarity + actor_similarity + keyword_similarity) / 3.0
    return combined_similarity


def compute_average_similarity(movie_ids, similarity_matrix, all_movie_titles):
    predictions = []

    for j, movie_id in enumerate(movie_ids):
        similarity_scores = similarity_matrix[:, j]
        average_similarity = np.mean(similarity_scores)
        predictions.append({
            "movie_id": movie_id,
            "cosine": average_similarity,
            "title": all_movie_titles.get(movie_id, "Unknown")
        })

    return sorted(predictions, key=lambda x: x['cosine'], reverse=True)


def main():
    json_file_path = "movies.json"
    all_movies = load_movies_from_json(json_file_path)

    movies = all_movies[:1000]

    liked_movie_ids = [105, 680, 569094, 574, 5874]
    selected_movies = [movie for movie in movies if 'detail' in movie and any(detail['id'] in liked_movie_ids
                       for detail in movie['detail'])]

    all_movie_ids = [detail['id'] for movie in movies if 'detail' in movie for detail in movie['detail']]

    all_movie_titles = {detail['id']: detail['title'] for movie in movies if 'detail' in movie
                        for detail in movie['detail']}

    all_genre_ids = extract_genres(movies)

    all_keywords = extract_keywords(movies)
    keyword_counts = Counter(all_keywords)
    most_common_keywords = [kw for kw, count in keyword_counts.most_common(500)]
    all_keywords = most_common_keywords

    all_actors = extract_actors(movies)

    similarity_matrix = compute_similarity(selected_movies, movies, all_genre_ids, all_actors, all_keywords)

    print("Cosine similarity between selected movies and all movies:")
    pprint(similarity_matrix)
    print()

    predictions = compute_average_similarity(all_movie_ids, similarity_matrix, all_movie_titles)

    for prediction in predictions[:20]:
        print( f"Movie ID: {prediction['movie_id']}, Title: {prediction['title']}, "
               f"Similarity: {prediction['cosine']:.2f}")


if __name__ == "__main__":
    main()
