import mysql.connector
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="movies"
)

def preprocess_text(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()]).lower()

def recommend_movie(movie_title):
    query = "SELECT movie_id, title, overview, genres, keywords, cast, crew FROM movie_list"
    movie_data = pd.read_sql(query, con=mydb)


    movie_data['title_lower'] = movie_data['title'].str.lower()

    movie_title_lower = movie_title.lower()

    if movie_title_lower in movie_data['title_lower'].values:
        movie_index = movie_data[movie_data['title_lower'] == movie_title_lower].index[0]
    else:

        movie_index = movie_data['title_lower'].str.contains(movie_title_lower).idxmax()

    movie_data['tags'] = (movie_data['overview'] + ' ' + movie_data['genres'] + ' ' +
                          movie_data['keywords'] + ' ' + movie_data['cast'] + ' ' +
                          movie_data['crew']).apply(preprocess_text)


    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(movie_data['tags']).toarray()


    similarity = cosine_similarity(vector)

    similarities = cosine_similarity(similarity[movie_index].reshape(1, -1), similarity).flatten()

    similar_indices = np.argsort(similarities)[::-1][1:6]
    similarity_scores = similarities[similar_indices] * 100

    print(f"Recommendations for '{movie_data.iloc[movie_index]['title']}':")
    for idx, (similarity_score, movie_idx) in enumerate(zip(similarity_scores, similar_indices), 1):
        print(f"{idx}: {movie_data.iloc[movie_idx]['title']} (Similarity: {similarity_score:.2f}%)")

movie_name = input("Enter the movie name: ")
recommend_movie(movie_name)