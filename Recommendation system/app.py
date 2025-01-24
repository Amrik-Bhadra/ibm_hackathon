import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load datasets
df_ratings = pd.read_csv('Ratings.csv', sep=';', encoding='latin-1')
df_books = pd.read_csv('Books.csv', sep=';', encoding='latin-1')
df_users = pd.read_csv('Users.csv', sep=';', encoding='latin-1', low_memory=False)

# Preprocessing
df_ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']
df_books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']
df_users.columns = ['User-ID', 'Age']

# Merge ratings with books
data = pd.merge(df_ratings, df_books, on='ISBN')

# Filter dataset to keep users and books with sufficient ratings
# Filter books with a minimum number of ratings
min_ratings = 5
df_books = df_books[df_books['ISBN'].isin(df_ratings['ISBN'].value_counts()[df_ratings['ISBN'].value_counts() >= min_ratings].index)]

# Filter users with a minimum number of ratings
min_user_ratings = 5
df_ratings = df_ratings[df_ratings['User-ID'].isin(df_ratings['User-ID'].value_counts()[df_ratings['User-ID'].value_counts() >= min_user_ratings].index)]

user_counts = data['User-ID'].value_counts()
book_counts = data['ISBN'].value_counts()

filtered_data = data[data['User-ID'].isin(user_counts[user_counts >= min_user_ratings].index)]
filtered_data = filtered_data[filtered_data['ISBN'].isin(book_counts[book_counts >= min_ratings].index)]

# Aggregate duplicate ratings by averaging them
filtered_data = filtered_data.groupby(['User-ID', 'ISBN'], as_index=False)['Book-Rating'].mean()

# Create pivot table for collaborative filtering
pivot_table = filtered_data.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# Convert pivot table to sparse matrix
sparse_matrix = csr_matrix(pivot_table.values)

# Calculate cosine similarity for collaborative filtering
cosine_sim = cosine_similarity(sparse_matrix)

# Preprocessing for content-based filtering
df_books['Content'] = (
    df_books['Book-Title'] + " " +
    df_books['Book-Author'] + " " +
    df_books['Publisher'].fillna("")
)

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN values in the 'Content' column with an empty string
df_books['Content'] = df_books['Content'].fillna('')

tfidf_matrix = tfidf.fit_transform(df_books['Content'])

# Calculate cosine similarity between books
content_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Collaborative filtering recommendation function
def recommend_books(user_id, n_recommendations=5):
    # Convert user_id to int to match the pivot table
    user_id = int(user_id)
    
    # Check if user_id exists in the pivot table
    if user_id not in pivot_table.index:
        return f"User {user_id} not found in the dataset."
    
    user_index = pivot_table.index.get_loc(user_id)
    similarity_scores = cosine_sim[user_index]
    similar_users = np.argsort(-similarity_scores)[1:]

    recommendations = {}
    for similar_user_index in similar_users:
        similar_user_id = pivot_table.index[similar_user_index]
        user_books = pivot_table.loc[similar_user_id]

        for book, rating in user_books.items():
            if pivot_table.loc[user_id, book] == 0 and rating > 0:
                recommendations[book] = recommendations.get(book, 0) + rating

        if len(recommendations) >= n_recommendations:
            break

    recommended_books = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    recommended_isbns = [book for book, _ in recommended_books]

    return df_books[df_books['ISBN'].isin(recommended_isbns)][['Book-Title', 'Book-Author']]

# Content-based filtering recommendation function
def recommend_books_content_based(user_id, n_recommendations=5):
    # Check if user_id exists in the pivot table
    if user_id not in pivot_table.index:
        return f"User {user_id} not found in the dataset."
    
    # Get books rated highly by the user
    user_rated_books = pivot_table.loc[user_id][pivot_table.loc[user_id] > 0]
    
    # Calculate similarity scores for all books
    recommendations = {}
    for isbn in user_rated_books.index:
        if isbn in df_books['ISBN'].values:
            book_idx = df_books[df_books['ISBN'] == isbn].index[0]
            sim_scores = list(enumerate(content_sim[book_idx]))
            
            for book_id, score in sim_scores:
                book_isbn = df_books.iloc[book_id]['ISBN']
                if book_isbn not in user_rated_books.index:  # Avoid already rated books
                    recommendations[book_isbn] = recommendations.get(book_isbn, 0) + score
    
    # Sort and return top recommendations
    recommended_books = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    recommended_isbns = [book for book, _ in recommended_books]
    return df_books[df_books['ISBN'].isin(recommended_isbns)][['Book-Title', 'Book-Author']]

# Hybrid recommendation function
def hybrid_recommendation(user_id, n_recommendations=5):
    # Collaborative Filtering Recommendations
    collab_recs = recommend_books(user_id, n_recommendations)
    
    # Content-Based Recommendations
    content_recs = recommend_books_content_based(user_id, n_recommendations)
    
    # Combine both
    hybrid_recs = pd.concat([collab_recs, content_recs]).drop_duplicates().head(n_recommendations)
    return hybrid_recs

# Input user ID
user_id = int(input("Enter a User ID from the available list: "))
print("\nCollaborative Filtering Recommendations:")
print(recommend_books(user_id))

print("\nContent-Based Recommendations:")
print(recommend_books_content_based(user_id))

print("\nHybrid Recommendations:")
print(hybrid_recommendation(user_id))