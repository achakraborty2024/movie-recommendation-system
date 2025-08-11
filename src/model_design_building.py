def run_model_design_building():
    def get_item_item_recommendations(title, df, cosine_sim=cosine_sim, num_recommendations=10):
        """
        Generates movie recommendations based on item-item collaborative filtering
        using pre-calculated cosine similarity.
    
        Args:
            title (str): The title of the input movie.
            df (pd.DataFrame): The DataFrame containing movie information (merged_df).
            cosine_sim (np.array): The cosine similarity matrix based on the 'soup'.
            num_recommendations (int, optional): The number of recommendations to generate. Defaults to 10.
    
        Returns:
            pd.DataFrame: A DataFrame containing the recommended movies and their similarity scores.
                          Returns an empty DataFrame if the movie is not found.
        """
        # Create a reverse mapping of movie titles to their indices if it doesn't exist
        if 'indices' not in globals():
             global indices
             indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    
        # Get the index of the movie that matches the title
        if title not in indices:
            print(f"Movie '{title}' not found in the dataset.")
            return pd.DataFrame()
    
        idx = indices[title]
    
        # Get the pairwise similarity scores for all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
    
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
        # Get the scores of the num_recommendations most similar movies
        # Skip the first element as it is the movie itself
        sim_scores = sim_scores[1:num_recommendations+1]
    
        # Get the movie indices and their similarity scores
        movie_indices = [(i[0], i[1]) for i in sim_scores]
    
        # Create a list of recommended movies and their similarity scores
        recommendations_list = []
        for idx, similarity_score in movie_indices:
            recommendations_list.append({
                'Recommended Movie': df['title'].iloc[idx],
                'Similarity Score (Cosine)': similarity_score
            })
    
        return pd.DataFrame(recommendations_list)
    
    # Example Usage: Get item-item recommendations for a movie
    movie_title_for_item_item = 'Avatar'  #@param {type:"string"}
    num_recommendations_item_item = 10  #@param {type:"slider", min:1, max:20, step:1}
    
    item_item_recommendations = get_item_item_recommendations(
        movie_title_for_item_item,
        merged_df,
        cosine_sim=cosine_sim,
        num_recommendations=num_recommendations_item_item
    )
    
    print(f"\nItem-Item Recommendations for '{movie_title_for_item_item}':")
    display(item_item_recommendations)
    
    # Define a function to generate combined recommendations
    def get_combined_recommendations_weighted(title, df, cosine_sim, num_recommendations=5, sentiment_weight=0.5, content_weight=0.5):
        """
        Generates movie recommendations based on combined sentiment and content similarity,
        with customizable weights for each component.
    
        Args:
            title (str): The title of the input movie.
            df (pd.DataFrame): The DataFrame containing movie information (merged_df).
            cosine_sim (np.array): The cosine similarity matrix based on the 'soup'.
            num_recommendations (int, optional): The number of recommendations to generate. Defaults to 5.
            sentiment_weight (float, optional): Weight for sentiment similarity. Defaults to 0.5.
            content_weight (float, optional): Weight for content similarity. Defaults to 0.5.
    
        Returns:
            pd.DataFrame: A DataFrame containing the recommended movies and their combined scores.
                          Returns an empty DataFrame if the movie is not found.
        """
        # Create a reverse mapping of movie titles to their indices if it doesn't exist
        if 'indices' not in globals():
             global indices
             indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
        if title not in indices:
            print(f"Movie '{title}' not found in the dataset.")
            return pd.DataFrame()
    
        idx = indices[title]
        input_sentiment_score = df.loc[idx, 'overview_sentiment_score']
    
        # Get sentiment similarity scores (closer to 0 difference is better)
        # We need to invert this difference to get a similarity score (higher is better)
        df_temp = df.copy()
        df_temp['sentiment_difference'] = abs(df_temp['overview_sentiment_score'] - input_sentiment_score)
        df_temp['sentiment_rank'] = df_temp['sentiment_difference'].rank(method='min', ascending=True)
        # Normalize sentiment rank (higher rank = less similar, so invert)
        df_temp['normalized_sentiment_sim'] = 1 / df_temp['sentiment_rank']
        df_temp['normalized_sentiment_sim'] = df_temp['normalized_sentiment_sim'] / df_temp['normalized_sentiment_sim'].max() # Normalize to 0-1
    
    
        # Get content similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Convert similarity scores to a Series
        content_sim_series = pd.Series([score for index, score in sim_scores])
        df_temp['content_sim'] = content_sim_series
        # Normalize content similarity
        df_temp['normalized_content_sim'] = df_temp['content_sim'] / df_temp['content_sim'].max()
    
    
        # Combine scores using weights
        df_temp['combined_score'] = (df_temp['normalized_sentiment_sim'] * sentiment_weight) + (df_temp['normalized_content_sim'] * content_weight)
    
        # Sort movies based on the combined score
        # Exclude the input movie itself
        recommended_movies = df_temp.sort_values(by='combined_score', ascending=False).head(num_recommendations + 1)
    
        # Filter out the input movie
        recommended_movies = recommended_movies[recommended_movies['title'] != title]
    
        # Return the top recommendations with relevant information
        return recommended_movies[['title', 'overview_sentiment_score', 'combined_score']].reset_index(drop=True)
    
    # Experiment with different weighting schemes and display recommendations
    movie_title_for_combined = 'Avatar'
    num_recommendations_combined = 10
    
    # Experiment 1: Equal weights
    print(f"\nCombined recommendations for '{movie_title_for_combined}' (Sentiment weight: 0.5, Content weight: 0.5):")
    recommendations_equal_weights = get_combined_recommendations_weighted(
        movie_title_for_combined,
        merged_df,
        cosine_sim=cosine_sim,
        num_recommendations=num_recommendations_combined,
        sentiment_weight=0.5,
        content_weight=0.5
    )
    display(recommendations_equal_weights)
    
    # Experiment 2: Higher content weight
    print(f"\nCombined recommendations for '{movie_title_for_combined}' (Sentiment weight: 0.2, Content weight: 0.8):")
    recommendations_higher_content = get_combined_recommendations_weighted(
        movie_title_for_combined,
        merged_df,
        cosine_sim=cosine_sim,
        num_recommendations=num_recommendations_combined,
        sentiment_weight=0.2,
        content_weight=0.8
    )
    display(recommendations_higher_content)
    
    # Experiment 3: Higher sentiment weight
    print(f"\nCombined recommendations for '{movie_title_for_combined}' (Sentiment weight: 0.8, Content weight: 0.2):")
    recommendations_higher_sentiment = get_combined_recommendations_weighted(
        movie_title_for_combined,
        merged_df,
        cosine_sim=cosine_sim,
        num_recommendations=num_recommendations_combined,
        sentiment_weight=0.8,
        content_weight=0.2
    )
    display(recommendations_higher_sentiment)
    
    # create recommendations using kNN from merged_df using feature engineering and input might be partial word
    
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    
    def generate_recommendations_knn(query, df, tfidf_matrix, num_recommendations=10):
        """
        Generates movie recommendations using k-Nearest Neighbors on the TF-IDF matrix.
        Supports partial word search in title and searches in the 'soup'.
    
        Args:
            query (str): The input query (movie title, keyword, plot, etc.).
            df (pd.DataFrame): The DataFrame containing movie information (merged_df).
            tfidf_matrix (sparse matrix): The TF-IDF matrix based on the 'soup'.
            num_recommendations (int, optional): The number of recommendations to generate. Defaults to 10.
    
        Returns:
            pd.DataFrame: A DataFrame containing the recommended movies and their similarity scores (distances).
                          Returns an empty DataFrame if no matches are found or no neighbors are found.
        """
        # Initialize KNN model (using cosine similarity which is 1 - distance for normalized vectors)
        # n_neighbors will be num_recommendations + 1 (including the item itself)
        knn = NearestNeighbors(n_neighbors=num_recommendations + 1, metric='cosine')
        knn.fit(tfidf_matrix)
    
        # --- 1. Try to find the query in the movie titles first (handles partial word matching) ---
        # Find indices where the title contains the query (case-insensitive)
        title_match_indices = df[df['title'].str.contains(str(query), case=False, na=False)].index.tolist()
    
        if title_match_indices:
            print(f"Found potential title matches for '{query}': {df.loc[title_match_indices, 'title'].tolist()}")
    
            # Prioritize an exact title match if found
            exact_match_indices = df[df['title'].str.lower() == str(query).lower()].index.tolist()
    
            if exact_match_indices:
                # If exact match, use its index for KNN
                search_index = exact_match_indices[0]
                print(f"Using exact title match '{df.loc[search_index, 'title']}' for KNN search.")
            else:
                # If no exact match, use the index of the first partial title match for KNN search
                # This might not be ideal, a better approach might be averaging vectors or
                # doing a keyword search, but for simplicity, we take the first match.
                search_index = title_match_indices[0]
                print(f"Using first partial title match '{df.loc[search_index, 'title']}' for KNN search.")
    
            # Get the vector for the chosen movie
            query_vector = tfidf_matrix[search_index]
    
        else:
            # --- 2. If no title match, treat the query as a keyword/plot search ---
            print(f"No title matches found for '{query}'. Treating as a keyword/content search.")
            try:
                # Transform the query using the fitted TF-IDF vectorizer
                query_vector = tfidf.transform([str(query)])
                # Check if the vector is empty (query not in vocabulary)
                if query_vector.sum() == 0:
                    print(f"Query '{query}' does not contain words in the vocabulary.")
                    return pd.DataFrame()
            except Exception as e:
                 print(f"Error transforming query '{query}': {e}")
                 return pd.DataFrame()
    
    
        # Find the k nearest neighbors
        distances, indices = knn.kneighbors(query_vector)
    
        # Flatten the results and get the indices and distances
        # indices[0] contains the indices of the neighbors
        # distances[0] contains the distances to the neighbors
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]
    
        # Create a list of recommendations
        recommendations_list = []
        for i in range(len(neighbor_indices)):
            idx = neighbor_indices[i]
            distance = neighbor_distances[i]
    
            # If we started with a movie index (title match), skip the first neighbor (the movie itself)
            if title_match_indices and idx == search_index:
                continue
    
            # Cosine similarity is 1 - cosine distance
            similarity_score = 1 - distance
    
            recommendations_list.append({
                'Recommended Movie': df['title'].iloc[idx],
                'Reason': 'Based on content similarity (TF-IDF + KNN)',
                'Confidence Score (Cosine Similarity)': similarity_score,
                'Overview': df['overview'].iloc[idx],
                'Genres': df['genres'].iloc[idx],
                'Keywords': df['keywords'].iloc[idx]
            })
    
        # Create DataFrame, sort by confidence score, and limit to num_recommendations
        recommendations_df = pd.DataFrame(recommendations_list)
    
        if recommendations_df.empty:
            print("No recommendations found.")
            return pd.DataFrame()
    
        # Sort by confidence score (similarity) descending
        recommendations_df = recommendations_df.sort_values(by='Confidence Score (Cosine Similarity)', ascending=False).head(num_recommendations).reset_index(drop=True)
    
    
        return recommendations_df
    
    
    # @title Generate Recommendations using kNN with partial word search
    
    knn_search_query = 'war'  #@param {type:"string"}
    num_recommendations_knn = 10  #@param {type:"slider", min:1, max:20, step:1}
    
    knn_recommendations = generate_recommendations_knn(
        knn_search_query,
        merged_df,
        tfidf_matrix, # Use the pre-calculated TF-IDF matrix
        num_recommendations=num_recommendations_knn
    )
    
    print(f"\nk-NN Recommendations for query '{knn_search_query}':")
    display(knn_recommendations)
    
    def generate_knn_recommendations_with_reasons(query, df, tfidf_matrix, num_recommendations=10):
        """
        Generates movie recommendations using k-Nearest Neighbors on the TF-IDF matrix,
        providing reasons based on content similarity.
        Supports partial word search in title and searches in the 'soup'.
    
        Args:
            query (str): The input query (movie title, keyword, plot, etc.).
            df (pd.DataFrame): The DataFrame containing movie information (merged_df).
            tfidf_matrix (sparse matrix): The TF-IDF matrix based on the 'soup'.
            num_recommendations (int, optional): The number of recommendations to generate. Defaults to 10.
    
        Returns:
            pd.DataFrame: A DataFrame containing the recommended movies, reason, and confidence score.
                          Returns an empty DataFrame if no matches are found or no neighbors are found.
        """
        # Initialize KNN model (using cosine similarity)
        knn = NearestNeighbors(n_neighbors=num_recommendations + 1, metric='cosine')
        knn.fit(tfidf_matrix)
    
        input_genres = ""
        input_keywords = ""
        input_title = ""
        search_index = -1 # To keep track if we're searching based on a specific movie index
    
        # --- 1. Try to find the query in the movie titles first (handles partial word matching) ---
        title_match_indices = df[df['title'].str.contains(str(query), case=False, na=False)].index.tolist()
    
        if title_match_indices:
            print(f"Found potential title matches for '{query}': {df.loc[title_match_indices, 'title'].tolist()}")
    
            exact_match_indices = df[df['title'].str.lower() == str(query).lower()].index.tolist()
    
            if exact_match_indices:
                search_index = exact_match_indices[0]
                input_title = df.loc[search_index, 'title']
                input_genres = df.loc[search_index, 'genres']
                input_keywords = df.loc[search_index, 'keywords']
                print(f"Using exact title match '{input_title}' for KNN search.")
                query_vector = tfidf_matrix[search_index]
            else:
                # If no exact match, use the first partial match's index
                search_index = title_match_indices[0]
                input_title = df.loc[search_index, 'title']
                input_genres = df.loc[search_index, 'genres']
                input_keywords = df.loc[search_index, 'keywords']
                print(f"Using first partial title match '{input_title}' for KNN search.")
                query_vector = tfidf_matrix[search_index]
    
        else:
            # --- 2. If no title match, treat the query as a keyword/plot search ---
            print(f"No title matches found for '{query}'. Treating as a keyword/content search.")
            try:
                query_vector = tfidf.transform([str(query)])
                if query_vector.sum() == 0:
                    print(f"Query '{query}' does not contain words in the vocabulary.")
                    return pd.DataFrame()
            except Exception as e:
                 print(f"Error transforming query '{query}': {e}")
                 return pd.DataFrame()
    
    
        # Find the k nearest neighbors
        distances, indices = knn.kneighbors(query_vector)
    
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]
    
        recommendations_list = []
        for i in range(len(neighbor_indices)):
            idx = neighbor_indices[i]
            distance = neighbor_distances[i]
    
            # If we started with a movie index, skip the first neighbor (the movie itself)
            if search_index != -1 and idx == search_index:
                continue
    
            similarity_score = 1 - distance # Cosine similarity
    
            rec_title = df['title'].iloc[idx]
            rec_genres = df['genres'].iloc[idx]
            rec_keywords = df['keywords'].iloc[idx]
            rec_cast = df['cast'].iloc[idx]
            rec_director = df['director'].iloc[idx]
    
    
            # Dynamically generate the reason based on what's similar
            reason_parts = []
            if search_index != -1: # If we started from a specific movie title
                 reason_parts.append(f"Similar to '{input_title}' based on content")
                 shared_genres = set(input_genres.split()) & set(rec_genres.split())
                 shared_keywords = set(input_keywords.split()) & set(rec_keywords.split())
    
                 if shared_genres:
                      reason_parts.append(f"shares genres like {', '.join(list(shared_genres)[:3])}")
                 if shared_keywords:
                      reason_parts.append(f"and keywords such as {', '.join(list(shared_keywords)[:3])}")
    
            else: # If we searched by keyword/plot
                 reason_parts.append(f"Matches content related to '{query}'")
                 # We could try to see which words from the query are in the recommended movie's soup
                 query_words = set(str(query).lower().split())
                 rec_soup_words = set(df['soup'].iloc[idx].lower().split())
                 matched_words = query_words & rec_soup_words
                 if matched_words:
                     reason_parts.append(f"shares terms like {', '.join(list(matched_words)[:3])}")
    
            reason = ", ".join(reason_parts).capitalize() + "."
    
            recommendations_list.append({
                'Recommended Movie': rec_title,
                'Reason': reason,
                'Confidence Score (Cosine Similarity)': similarity_score
            })
    
        # Create DataFrame, sort by confidence score, and limit to num_recommendations
        recommendations_df = pd.DataFrame(recommendations_list)
    
        if recommendations_df.empty:
            print("No recommendations found.")
            return pd.DataFrame()
    
        # Sort by confidence score (similarity) descending
        recommendations_df = recommendations_df.sort_values(by='Confidence Score (Cosine Similarity)', ascending=False).head(num_recommendations).reset_index(drop=True)
    
    
        return recommendations_df
    
    # @title Generate Recommendations using kNN with multiple keywords
    
    knn_search_query_reasons = 'Galaxy'  #@param {type:"string"}
    num_recommendations_knn_reasons = 10  #@param {type:"slider", min:1, max:20, step:1}
    
    
    knn_recommendations_with_reasons = generate_knn_recommendations_with_reasons(
        knn_search_query_reasons,
        merged_df,
        tfidf_matrix, # Use the pre-calculated TF-IDF matrix
        num_recommendations=num_recommendations_knn_reasons
    )
    
    print(f"\nk-NN Recommendations with Reasons for query '{knn_search_query_reasons}':")
    display(knn_recommendations_with_reasons)
    
    
    def generate_knn_recommendations_with_spellcheck(query, df, tfidf, tfidf_matrix, num_recommendations=10):
        """
        Generates movie recommendations using k-Nearest Neighbors on the TF-IDF matrix.
        Includes basic spell checking for the query using TF-IDF vectorizer vocabulary.
        Supports partial word search in title and searches in the 'soup'.
    
        Args:
            query (str): The input query (movie title, keyword, plot, etc.).
            df (pd.DataFrame): The DataFrame containing movie information (merged_df).
            tfidf (TfidfVectorizer): The fitted TF-IDF vectorizer.
            tfidf_matrix (sparse matrix): The TF-IDF matrix based on the 'soup'.
            num_recommendations (int, optional): The number of recommendations to generate. Defaults to 10.
    
        Returns:
            pd.DataFrame: A DataFrame containing the recommended movies, reason, and confidence score.
                          Returns an empty DataFrame if no matches are found or no neighbors are found.
        """
        # Basic Spell Checking: Check if query words are in the TF-IDF vocabulary
        query_words = str(query).lower().split()
        latest_query_words = []
        vocabulary = tfidf.vocabulary_
        inverse_vocabulary = {i: word for word, i in vocabulary.items()}
    
        # This is a very simple "correction" - just keeps words that are in the vocabulary.
        # A more robust spell checker would use edit distance or phonetic algorithms.
        for word in query_words:
            if word in vocabulary:
                latest_query_words.append(word)
            else:
                # Optionally find the closest word in the vocabulary (more complex)
                # For now, just drop out-of-vocabulary words
                print(f"Warning: Word '{word}' not found in vocabulary. Skipping or attempting simple correction.")
                # Simple attempt to find closest based on first few characters (very basic)
                closest_matches = [vocab_word for vocab_word in vocabulary if vocab_word.startswith(word[:3])]
                if closest_matches:
                    # Take the first closest match as a 'correction'
                    latest_word = closest_matches[0]
                    print(f"  Suggesting '{latest_word}' for '{word}'.")
                    latest_query_words.append(latest_word)
    
    
        latest_query = " ".join(latest_query_words)
    
        if not latest_query:
            print("Latest query is empty. Cannot proceed with recommendation.")
            return pd.DataFrame()
    
        print(f"Original Query: '{query}'")
        print(f"Processed Query (after basic spellcheck): '{latest_query}'")
    
    
        knn = NearestNeighbors(n_neighbors=num_recommendations + 1, metric='cosine')
        knn.fit(tfidf_matrix)
    
        input_genres = ""
        input_keywords = ""
        input_title = ""
        search_index = -1 # To keep track if we're searching based on a specific movie index
    
        # --- 1. Try to find the latest query in the movie titles first ---
        # Use the original query for title matching to allow partial original query words
        # Although, if the user typed 'Avatr', they might mean 'Avatar', so use the latest
        # Let's use the latest query for title matching for consistency after spellcheck.
        title_match_indices = df[df['title'].str.contains(str(latest_query), case=False, na=False)].index.tolist()
    
        if title_match_indices:
            print(f"Found potential title matches for '{latest_query}': {df.loc[title_match_indices, 'title'].tolist()}")
    
            exact_match_indices = df[df['title'].str.lower() == str(latest_query).lower()].index.tolist()
    
            if exact_match_indices:
                search_index = exact_match_indices[0]
                input_title = df.loc[search_index, 'title']
                input_genres = df.loc[search_index, 'genres']
                input_keywords = df.loc[search_index, 'keywords']
                print(f"Using exact title match '{input_title}' for KNN search.")
                query_vector = tfidf_matrix[search_index]
            else:
                search_index = title_match_indices[0]
                input_title = df.loc[search_index, 'title']
                input_genres = df.loc[search_index, 'genres']
                input_keywords = df.loc[search_index, 'keywords']
                print(f"Using first partial title match '{input_title}' for KNN search.")
                query_vector = tfidf_matrix[search_index]
    
        else:
            # --- 2. If no title match, treat the latest query as a keyword/plot search ---
            print(f"No title matches found for '{latest_query}'. Treating as a keyword/content search.")
            try:
                query_vector = tfidf.transform([str(latest_query)])
                if query_vector.sum() == 0:
                    print(f"Latest query '{latest_query}' does not contain words in the vocabulary or resulted in an empty vector.")
                    return pd.DataFrame()
            except Exception as e:
                 print(f"Error transforming latest query '{latest_query}': {e}")
                 return pd.DataFrame()
    
    
        # Find the k nearest neighbors
        distances, indices = knn.kneighbors(query_vector)
    
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]
    
        recommendations_list = []
        for i in range(len(neighbor_indices)):
            idx = neighbor_indices[i]
            distance = neighbor_distances[i]
    
            # If we started with a movie index, skip the first neighbor (the movie itself)
            if search_index != -1 and idx == search_index:
                continue
    
            similarity_score = 1 - distance # Cosine similarity
    
            rec_title = df['title'].iloc[idx]
            rec_genres = df['genres'].iloc[idx]
            rec_keywords = df['keywords'].iloc[idx]
            rec_cast = df['cast'].iloc[idx]
            rec_director = df['director'].iloc[idx]
    
    
            # Dynamically generate the reason based on what's similar
            reason_parts = []
            if search_index != -1: # If we started from a specific movie title
                 reason_parts.append(f"Similar to '{input_title}' based on content")
                 shared_genres = set(input_genres.split()) & set(rec_genres.split())
                 shared_keywords = set(input_keywords.split()) & set(rec_keywords.split())
    
                 if shared_genres:
                      reason_parts.append(f"shares genres like {', '.join(list(shared_genres)[:3])}")
                 if shared_keywords:
                      reason_parts.append(f"and keywords such as {', '.join(list(shared_keywords)[:3])}")
    
            else: # If we searched by keyword/plot
                 reason_parts.append(f"Matches content related to '{latest_query}'")
                 # We could try to see which words from the latest query are in the recommended movie's soup
                 query_words_set = set(latest_query.lower().split())
                 rec_soup_words = set(df['soup'].iloc[idx].lower().split())
                 matched_words = query_words_set & rec_soup_words
                 if matched_words:
                     reason_parts.append(f"shares terms like {', '.join(list(matched_words)[:3])}")
    
    
            reason = ", ".join(reason_parts).capitalize() + "."
    
            recommendations_list.append({
                'Recommended Movie': rec_title,
                'Reason': reason,
                'Confidence Score (Cosine Similarity)': similarity_score
            })
    
        recommendations_df = pd.DataFrame(recommendations_list)
    
        if recommendations_df.empty:
            print("No recommendations found.")
            return pd.DataFrame()
    
        recommendations_df = recommendations_df.sort_values(by='Confidence Score (Cosine Similarity)', ascending=False).head(num_recommendations).reset_index(drop=True)
    
        return recommendations_df
    
    # @title Generate Recommendations using kNN with Query improvements
    
    spellcheck_search_query = 'Holer'  #@param {type:"string"}
    num_recommendations_spellcheck = 10  #@param {type:"slider", min:1, max:20, step:1}
    
    
    spellcheck_knn_recommendations = generate_knn_recommendations_with_spellcheck(
        spellcheck_search_query,
        merged_df,
        tfidf,          # Pass the fitted TF-IDF vectorizer
        tfidf_matrix,   # Pass the pre-calculated TF-IDF matrix
        num_recommendations=num_recommendations_spellcheck
    )
    
    print(f"\nk-NN Recommendations with Basic Spellcheck for query '{spellcheck_search_query}':")
    display(spellcheck_knn_recommendations)
    
    spellcheck_search_query_2 = 'sciene fiction'  #@param {type:"string"}
    num_recommendations_spellcheck_2 = 10  #@param {type:"slider", min:1, max:20, step:1}
    
    
    spellcheck_knn_recommendations_2 = generate_knn_recommendations_with_spellcheck(
        spellcheck_search_query_2,
        merged_df,
        tfidf,          # Pass the fitted TF-IDF vectorizer
        tfidf_matrix,   # Pass the pre-calculated TF-IDF matrix
        num_recommendations=num_recommendations_spellcheck_2
    )
    
    print(f"\nk-NN Recommendations with Basic Spellcheck for query '{spellcheck_search_query_2}':")
    display(spellcheck_knn_recommendations_2)
    
    
    !pip install tensorflow
    !pip install keras
    
    # Create recommendations using Deep Learning from merged_df
    
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    # The TF-IDF matrix represents item features. We can use this as input to an Autoencoder.
    
    # Scale the TF-IDF matrix data
    scaler = MinMaxScaler()
    tfidf_scaled = scaler.fit_transform(tfidf_matrix.toarray()) # Convert sparse matrix to dense array for scaling
    
    # Autoencoder Model Parameters
    input_dim = tfidf_scaled.shape[1] # Number of features from TF-IDF
    encoding_dim = 128 # Size of the latent representation (can be tuned)
    
    # Build the Autoencoder Model
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder_layer = Dense(encoding_dim, activation='relu')(input_layer) # Latent space
    
    # Decoder
    decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer) # Reconstruct input
    
    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
    
    # Compile the Autoencoder
    autoencoder.compile(optimizer='adam', loss='mse') # Mean Squared Error loss for reconstruction
    
    # Train the Autoencoder
    # Use the scaled TF-IDF matrix as both input and target
    # Split data for training and validation (optional but good practice)
    X_train, X_val = train_test_split(tfidf_scaled, test_size=0.1, random_state=42)
    
    print("\nTraining Autoencoder...")
    history = autoencoder.fit(X_train, X_train,
                    epochs=20,        # Number of training epochs
                    batch_size=256,   # Batch size
                    shuffle=True,
                    validation_data=(X_val, X_val))
    print("Autoencoder Training Complete.")
    
    # Get the Encoder model (to extract the latent representations)
    encoder = Model(inputs=input_layer, outputs=encoder_layer)
    
    # Get the latent representations for all movies
    # These are dense, lower-dimensional feature vectors learned by the autoencoder
    latent_features = encoder.predict(tfidf_scaled)
    
    print("\nShape of learned latent features:", latent_features.shape)
    
    # Now, we can use these latent features to find similar movies
    # We can use a distance metric like Euclidean distance or Cosine similarity on these features.
    # Cosine similarity is often preferred for text/feature vectors.
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Calculate cosine similarity matrix on the latent features
    latent_cosine_sim = cosine_similarity(latent_features, latent_features)
    
    print("Shape of Latent Cosine Similarity matrix:", latent_cosine_sim.shape)
    
    # Create a reverse mapping of movie soup column to their indices if it doesn't exist
    if 'indices' not in globals():
         global indices
         indices = pd.Series(merged_df.index, index=merged_df['soup']).drop_duplicates()
    
    
    # Function to get recommendations based on cosine similarity of the latent features
    def get_autoencoder_recommendations(title, df, latent_cosine_sim=latent_cosine_sim, num_recommendations=10):
        """
        Generates movie recommendations based on cosine similarity of Autoencoder-learned latent features.
    
        Args:
            title (str): The title of the input movie.
            df (pd.DataFrame): The DataFrame containing movie information (merged_df).
            latent_cosine_sim (np.array): The cosine similarity matrix based on latent features.
            num_recommendations (int, optional): The number of recommendations to generate. Defaults to 10.
    
        Returns:
            pd.DataFrame: A DataFrame containing the recommended movies and their similarity scores.
                          Returns an empty DataFrame if the movie is not found.
        """
        if title not in indices:
            print(f"Movie '{title}' not found in the dataset.")
            return pd.DataFrame()
    
        idx = indices[title]
    
        # Get the pairwise similarity scores for all movies with that movie
        sim_scores = list(enumerate(latent_cosine_sim[idx]))
    
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
        # Get the scores of the num_recommendations most similar movies
        # Skip the first element as it is the movie itself
        sim_scores = sim_scores[1:num_recommendations+1]
    
        # Get the movie indices and their similarity scores
        movie_indices = [(i[0], i[1]) for i in sim_scores]
    
        # Create a list of recommended movies and their similarity scores
        recommendations_list = []
        for idx, similarity_score in movie_indices:
            recommendations_list.append({
                'Recommended Movie': df['title'].iloc[idx],
                'Similarity Score (Autoencoder Latent Features)': similarity_score,
                'Overview': df['overview'].iloc[idx],
                'Genres': df['genres'].iloc[idx],
                'Keywords': df['keywords'].iloc[idx]
            })
    
        return pd.DataFrame(recommendations_list)
    
    # @title Plot the Autoencoder Training Loss
    
    print("## Plotting Autoencoder Training Loss\n")
    
    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Autoencoder Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.show()
    
    print("\nThe plot above shows the Mean Squared Error (MSE) decreasing over the training epochs.")
    print("A decreasing loss indicates that the Autoencoder is learning to reconstruct the input data.")
    
    # @title Generate Recommendations using Autoencoder Latent Features
    
    autoencoder_movie_title = 'Liar Liar'  #@param {type:"string"}
    num_recommendations_autoencoder = 10  #@param {type:"slider", min:1, max:20, step:1}
    
    
    autoencoder_recommendations = get_autoencoder_recommendations(
        autoencoder_movie_title,
        merged_df,
        latent_cosine_sim=latent_cosine_sim,
        num_recommendations=num_recommendations_autoencoder
    )
    
    print(f"\nAutoencoder-based Recommendations for '{autoencoder_movie_title}':")
    display(autoencoder_recommendations)
    
    # @title Similarity Score (Autoencoder Latent Features)
    
    from matplotlib import pyplot as plt
    autoencoder_recommendations['Similarity Score (Autoencoder Latent Features)'].plot(kind='hist', bins=20, title='Similarity Score (Autoencoder Latent Features)')
    plt.gca().spines[['top', 'right',]].set_visible(False)
    
    # @title Plot the loss function for autoencoder model
    
    import matplotlib.pyplot as plt
    # To evaluate the Autoencoder model's reconstruction performance, we plot the training loss.
    # The history object from autoencoder.fit contains the loss values per epoch.
    
    plt.figure(figsize=(10, 6))
    plt.plot(autoencoder.history.history['loss'], label='Training Loss')
    if 'val_loss' in autoencoder.history.history:
        plt.plot(autoencoder.history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Reconstruction Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # elbow curve to find the optimum number of clusters
    
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    
    # We perform clustering on the latent features learned by the autoencoder
    # It's computationally expensive to run for a very large range, let's pick a reasonable range
    inertia = []
    cluster_range = range(1, 150, 10) # Test number of clusters from 1 to 150 with step 10
    
    print("Calculating inertia for different numbers of clusters...")
    for k in cluster_range:
        # n_init is set explicitly
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Set n_init to 10
        kmeans.fit(latent_features)
        inertia.append(kmeans.inertia_)
        print(f"Completed KMeans for k={k}, Inertia: {kmeans.inertia_:.2f}")
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, inertia, marker='o', linestyle='-')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.xticks(cluster_range) # Set x-axis ticks to the values in cluster_range
    plt.grid(True)
    plt.show()
    
    print("\nObserve the plot to find the 'elbow' point, where the rate of decrease in inertia slows down.")
    print("This point suggests a potentially optimal number of clusters.")
    
    
    import numpy as np
    import joblib
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD  # PCA also fine if embeddings are dense
    from sklearn.metrics.pairwise import cosine_similarity
    
    # latent_features: np.ndarray of shape (n_movies, D_IN)  # encoder output (e.g., 256)
    # merged_df: DataFrame with at least 'title' column
    
    # ---- Config ----
    USE_REDUCER   = True           # set False to cluster on encoder dims directly
    N_COMPONENTS  = 128            # target dim for KMeans space
    N_CLUSTERS    = 10             # elbow choice
    RANDOM_STATE  = 42
    N_INIT        = 10             # explicit for older sklearn
    
    # ---- Sanity on encoder output ----
    D_IN = int(latent_features.shape[1])
    print(f"[ENCODER] latent_features shape: {latent_features.shape}  (D_IN={D_IN})")
    
    
    # ---- Prepare feature space for KMeans ----
    if USE_REDUCER:
        # If a reducer was previously loaded in the notebook, only reuse it if it matches D_IN -> N_COMPONENTS.
        # Otherwise (or if none), fit a fresh reducer on the encoder embeddings.
        reuse = False
        if 'reducer' in globals() and reducer is not None:
            rin  = getattr(reducer, "n_features_in_", None)
            rout = getattr(reducer, "n_components", None)
            reuse = (rin == D_IN and rout == N_COMPONENTS)
            print(f"[Reducer] Found existing reducer: in={rin} out={rout}  -> reuse={reuse}")
    
        if not reuse:
            reducer = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
            reducer.fit(latent_features)
            rin  = getattr(reducer, "n_features_in_", None)
            rout = getattr(reducer, "n_components", None)
            if rin != D_IN or rout != N_COMPONENTS:
                raise ValueError(f"[Reducer] Mismatch after fit: got in={rin}, out={rout}, "
                                 f"expected in={D_IN}, out={N_COMPONENTS}")
    
        features_for_kmeans = reducer.transform(latent_features)  # shape: (n_movies, N_COMPONENTS)
        print(f"[Reducer] OK  ({type(reducer).__name__}): in={D_IN} → out={features_for_kmeans.shape[1]}")
    else:
        reducer = None
        features_for_kmeans = latent_features
        print(f"[Reducer] Disabled. KMeans will use {features_for_kmeans.shape[1]}-D embeddings")
    
    # ---- KMeans on the chosen feature space ----
    dim_for_kmeans = features_for_kmeans.shape[1]
    print(f"[KMeans] Fitting K={N_CLUSTERS} on dim={dim_for_kmeans} ...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
    cluster_labels = kmeans.fit_predict(features_for_kmeans)
    merged_df['cluster'] = cluster_labels
    print("[KMeans] Done. Cluster counts:", np.bincount(cluster_labels))
    
    # ---- Verify KMeans expects the same dim we trained with ----
    km_in = getattr(kmeans, "n_features_in_", None)
    if km_in is None and hasattr(kmeans, "cluster_centers_"):
        km_in = int(kmeans.cluster_centers_.shape[1])
    if km_in != dim_for_kmeans:
        raise RuntimeError(f"[KMeans] n_features_in_={km_in} but trained on dim={dim_for_kmeans} — something’s off.")
    
    # ---- Save artifacts for inference (HF Spaces) ----
    joblib.dump(kmeans, "kmeans_model.joblib")
    print("[Save] kmeans_model.joblib")
    if reducer is not None:
        joblib.dump(reducer, "kmeans_input_reducer.joblib")
        print(f"[Save] kmeans_input_reducer.joblib (apply AFTER encoder: {D_IN}→{N_COMPONENTS})")
    
    # ---- Helper to get cluster-based recommendations (ranked by similarity) ----
    # We compute similarity in the SAME space used by KMeans.
    def get_clustering_recommendations(title, df, features_for_kmeans, num_recommendations=10):
        """
        Recommend movies from the same cluster, ranked by cosine similarity
        in the KMeans feature space (reducer output if used, otherwise encoder space).
        """
        if title not in df['title'].values:
            print(f"Movie '{title}' not found in the dataset.")
            return df.iloc[0:0][['title', 'overview_sentiment_score', 'genres', 'keywords', 'cluster']]
    
        idx = df.index[df['title'] == title][0]
        c = int(df.at[idx, 'cluster'])
        members_idx = df.index[df['cluster'] == c].tolist()
        members_idx = [i for i in members_idx if i != idx]
        if not members_idx:
            print(f"No other movies found in the same cluster as '{title}'.")
            return df.iloc[0:0][['title', 'overview_sentiment_score', 'genres', 'keywords', 'cluster']]
    
        q = features_for_kmeans[idx].reshape(1, -1)
        M = features_for_kmeans[members_idx]
        sims = cosine_similarity(q, M).flatten()
        order = np.argsort(-sims)[:num_recommendations]
        top_idx = [members_idx[i] for i in order]
        top_scores = sims[order]
    
        out = df.loc[top_idx, ['title', 'overview_sentiment_score', 'genres', 'keywords', 'cluster']].copy()
        out.insert(1, 'cluster_similarity', top_scores)
        return out
    
    # ---- Example usage ----
    clustering_movie_title = 'Avatar'
    num_recommendations_clustering = 10
    
    clustering_recommendations = get_clustering_recommendations(
        clustering_movie_title,
        merged_df,
        features_for_kmeans,
        num_recommendations=num_recommendations_clustering
    )
    
    print(f"\nItem-Based Clustering Recommendations for '{clustering_movie_title}':")
    display(clustering_recommendations)
    
    print("\nSample Movies from Cluster 0:")
    display(merged_df[merged_df['cluster'] == 0].head())
    
    print("\nSample Movies from Cluster 1:")
    display(merged_df[merged_df['cluster'] == 1].head())
    
    # @title Scatter Plot for KMeans clusters from merged_df
    
    import matplotlib.pyplot as plt
    # Visualize the clusters in 2D or 3D (using PCA or t-SNE for dimensionality reduction)
    
    # Use PCA to reduce latent features to 2 components for visualization
    from sklearn.decomposition import PCA
    
    # the latent representations for all movies
    # These are dense, lower-dimensional feature vectors learned by the autoencoder
    pca = PCA(n_components=2)
    latent_features_2d = pca.fit_transform(latent_features)
    
    # Add the 2D PCA coordinates to the dataframe
    merged_df['pca_comp1'] = latent_features_2d[:, 0]
    merged_df['pca_comp2'] = latent_features_2d[:, 1]
    
    # Create a scatter plot of the clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=merged_df,
        x='pca_comp1',
        y='pca_comp2',
        hue='cluster',  # Color points by cluster label
        palette='viridis', # Color palette
        legend='full',
        alpha=0.6,
        s=30 # point size
    )
    
    plt.title('KMeans Clusters Visualized using PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()
    
    # clustering for movie sentiment analysis from merged_df and then use that model to predict
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    
    # Perform KMeans clustering on the latent features
    n_clusters = 11  # Number of clusters based on elbow curve above
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Explicitly set n_init
    cluster_labels = kmeans.fit_predict(latent_features)
    
    # Add cluster labels to the dataframe
    merged_df['cluster'] = cluster_labels
    
    # Function to get recommendations based on item-based clustering
    def get_clustering_recommendations(title, df, num_recommendations=10):
        """
        Generates movie recommendations based on finding other movies in the same cluster.
    
        Args:
            title (str): The title of the input movie.
            df (pd.DataFrame): The DataFrame containing movie information (merged_df).
            num_recommendations (int, optional): The number of recommendations to generate. Defaults to 10.
    
        Returns:
            pd.DataFrame: A DataFrame containing recommended movies from the same cluster.
                          Returns an empty DataFrame if the movie is not found or cluster is empty.
        """
        if title not in df['title'].values:
            print(f"Movie '{title}' not found in the dataset.")
            return pd.DataFrame()
    
        # Get the cluster of the input movie
        input_movie_cluster = df[df['title'] == title]['cluster'].iloc[0]
    
        # Find all movies in the same cluster
        movies_in_same_cluster = df[df['cluster'] == input_movie_cluster]
    
        # Exclude the input movie itself
        recommended_movies = movies_in_same_cluster[movies_in_same_cluster['title'] != title]
    
        # If there are more movies in the cluster than needed, sample them randomly
        if len(recommended_movies) > num_recommendations:
            recommended_movies = recommended_movies.sample(n=num_recommendations, random_state=42)
        elif recommended_movies.empty:
             print(f"No other movies found in the same cluster as '{title}'.")
    
    
        # Return the relevant columns for recommendations
        return recommended_movies[['title', 'overview_sentiment_score', 'genres', 'keywords', 'cluster']]
    
    
    # @title Generate Recommendations using Item-Based Clustering
    
    clustering_movie_title = 'Water'  #@param {type:"string"}
    num_recommendations_clustering = 10  #@param {type:"slider", min:1, max:20, step:1}
    
    clustering_recommendations = get_clustering_recommendations(
        clustering_movie_title,
        merged_df,
        num_recommendations=num_recommendations_clustering
    )
    
    print(f"\nItem-Based Clustering Recommendations for '{clustering_movie_title}':")
    display(clustering_recommendations)
    
    
    # analyze the contents of a few clusters to understand what kind of movies are grouped together.
    print(f"\nSample Movies from Cluster 0:")
    display(merged_df[merged_df['cluster'] == 0].head())
    
    print(f"\nSample Movies from Cluster 1:")
    display(merged_df[merged_df['cluster'] == 1].head())
    
    # @title Scatter Plot for KMeans clusters from merged_df
    pca_3d = PCA(n_components=3)
    latent_features_3d = pca_3d.fit_transform(latent_features)
    
    merged_df['pca_comp3'] = latent_features_3d[:, 2]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        merged_df['pca_comp1'],
        merged_df['pca_comp2'],
        merged_df['pca_comp3'],
        c=merged_df['cluster'],
        cmap='viridis',
        s=30,
        alpha=0.6
    )
    
    ax.set_title('KMeans Clusters Visualized using 3D PCA')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    
    # Add legend
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    plt.show()
    
    def query_clustering_recommendations(query, df, tfidf, kmeans, num_recommendations=10):
        """
        Generates movie recommendations by finding the cluster of a movie
        identified by a query (which can be a title or content keyword),
        and then recommending other movies within that same cluster.
    
        Args:
            query (str): The input query (movie title, keyword, plot, etc.).
            df (pd.DataFrame): The DataFrame containing movie information (merged_df)
                               with a 'cluster' column.
            tfidf (TfidfVectorizer): The fitted TF-IDF vectorizer.
            kmeans (KMeans): The fitted KMeans model.
            num_recommendations (int, optional): The number of recommendations to generate. Defaults to 10.
    
        Returns:
            pd.DataFrame: A DataFrame containing recommended movies from the identified cluster,
                          including sentiment score and other relevant features.
                          Returns an empty DataFrame if no cluster is identified or cluster is empty.
        """
        input_movie_cluster = None
        query_vector = None
    
        # --- 1. Find the most relevant movie based on the query using TF-IDF and cosine similarity ---
        # This step is similar to the initial keyword search but aims to identify a single
        # representative movie from which to find the cluster.
    
        try:
            query_vector = tfidf.transform([str(query)])
            if query_vector.sum() == 0:
                print(f"Query '{query}' does not contain words in the vocabulary.")
                return pd.DataFrame()
        except Exception as e:
             print(f"Error transforming query '{query}': {e}")
             return pd.DataFrame()
    
        # Calculate cosine similarity between the query vector and all movie soup vectors
        keyword_sim_scores = linear_kernel(query_vector, tfidf_matrix).flatten()
    
        # Find the index of the movie with the highest similarity score
        # This movie will be used to identify the cluster
        most_similar_movie_idx = keyword_sim_scores.argmax()
        confidence_score = keyword_sim_scores[most_similar_movie_idx]
    
        if confidence_score == 0:
            print(f"Query '{query}' did not match any movie content significantly.")
            return pd.DataFrame()
    
        # Get the title of the most similar movie
        identified_movie_title = df['title'].iloc[most_similar_movie_idx]
        print(f"Query '{query}' is most similar to movie: '{identified_movie_title}' (Confidence: {confidence_score:.4f})")
    
    
        # --- 2. Get the cluster of the identified movie ---
        if identified_movie_title not in df['title'].values:
            print(f"Identified movie '{identified_movie_title}' not found in the dataframe.")
            return pd.DataFrame()
    
        input_movie_cluster = df[df['title'] == identified_movie_title]['cluster'].iloc[0]
        print(f"Identified movie '{identified_movie_title}' belongs to Cluster: {input_movie_cluster}")
    
    
        # --- 3. Find other movies in the same cluster ---
        movies_in_same_cluster = df[df['cluster'] == input_movie_cluster].copy() # Create a copy to avoid SettingWithCopyWarning
    
    
        # Exclude the identified movie itself
        recommended_movies = movies_in_same_cluster[movies_in_same_cluster['title'] != identified_movie_title]
    
        # If there are more movies in the cluster than needed, sample them randomly
        if len(recommended_movies) > num_recommendations:
            # Use the confidence score from the query match as a potential way to rank within the cluster
            # However, within a cluster, items are existed similar. Random sampling or ranking by
            # sentiment/vote might be more appropriate than the initial query match score.
            # Let's add the initial confidence score for context but sample randomly or sort by sentiment/popularity.
    
            # Option A: Sample randomly
            # recommended_movies = recommended_movies.sample(n=num_recommendations, random_state=42)
    
            # Option B: Sort by a metric like overview_sentiment_score or vote_average (if available and relevant)
            # Sorting by sentiment might recommend movies in the cluster with similar emotional tone.
            recommended_movies = recommended_movies.sort_values(by='overview_sentiment_score', ascending=False).head(num_recommendations)
    
        elif recommended_movies.empty:
             print(f"No other movies found in the same cluster as '{identified_movie_title}'.")
             return pd.DataFrame()
    
    
        # Add a 'Reason' and 'Confidence Score' column (using the initial query match score for context)
        # Note: The confidence score here is for the initial query match, not similarity within the cluster.
        # Within the cluster, items are existed to be similar.
        recommended_movies['Reason'] = f"Recommended from Cluster {input_movie_cluster} (Most similar to query '{query}' was '{identified_movie_title}')"
        recommended_movies['Initial Query Confidence Score'] = confidence_score
    
    
        # Return the relevant columns for recommendations
        return recommended_movies[['title', 'Reason', 'Initial Query Confidence Score',
                                  'overview_sentiment_score', 'genres', 'keywords', 'cluster']].rename(columns={'title': 'Recommended Movie'})
    
    
    # Check if latent_features is defined (it should be from the autoencoder section)
    if 'latent_features' not in globals():
        print("Latent features not found. Please run the Autoencoder section first.")
    else:
        # Use the elbow method to find the optimal number of clusters for KMeans
        # It's computationally expensive to run for a very large range, let's pick a reasonable range
        inertia = []
        # A smaller step size might give a better elbow point but takes longer
        # Let's try a range of 1 to 150 with a step of 10 first.
        cluster_range = range(1, 150, 10)
    
        print("Calculating inertia for different numbers of clusters (Elbow Method)...")
        for k in cluster_range:
            # n_init is set explicitly to avoid warning in newer scikit-learn versions
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(latent_features)
            inertia.append(kmeans.inertia_)
            # Optional: Print progress
            print(f"Completed KMeans for k={k}, Inertia: {kmeans.inertia_:.2f}")
    
        # Plot the elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, inertia, marker='o', linestyle='-')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Within-cluster sum of squares)')
        # Set x-axis ticks to the values in cluster_range for clarity
        plt.xticks(cluster_range)
        plt.grid(True)
        plt.show()
    
        print("\nObserve the plot to find the 'elbow' point, where the rate of decrease in inertia slows down.")
        print("This point suggests a potentially optimal number of clusters.")
        print("Based on the previous run, around 11-21 seemed reasonable. Let's proceed with a chosen number.")
    
        # Perform KMeans clustering with the chosen number of clusters
        # Choose the number of clusters based on the elbow plot observation
        # Let's pick a number from the visually determined elbow range, e.g., 15
        n_clusters_chosen = 11
    
        print(f"\nPerforming KMeans clustering with {n_clusters_chosen} clusters on latent features...")
        kmeans_model = KMeans(n_clusters=n_clusters_chosen, random_state=42, n_init=10) # Explicitly set n_init
        cluster_labels = kmeans_model.fit_predict(latent_features)
    
        # Add cluster labels to the dataframe
        # Ensure the column name doesn't clash if we ran this section before
        merged_df['kmeans_cluster'] = cluster_labels
    
        print(f"Clustering complete. Added '{'kmeans_cluster'}' column to merged_df.")
        print("\nDistribution of movies per cluster:")
        print(merged_df['kmeans_cluster'].value_counts().sort_index())
    
    
        # @title Generate Recommendations using Item-Based Clustering with Query
    
        clustering_search_query = 'Liar'  #@param {type:"string"}
        num_recommendations_clustering_query = 10  #@param {type:"slider", min:1, max:20, step:1}
    
        clustering_recommendations_query = query_clustering_recommendations(
            clustering_search_query,
            merged_df,
            tfidf, # Pass the fitted TF-IDF vectorizer
            kmeans_model, # Pass the trained KMeans model
            num_recommendations=num_recommendations_clustering_query
        )
    
        print(f"\nItem-Based Clustering Recommendations for query '{clustering_search_query}':")
        display(clustering_recommendations_query)
    
    # Experiment with Text Preprocessing
    
    print("## Experimenting with Text Preprocessing Techniques\n")
    
    import re
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    # from nltk.stem.wordnet import WordNetLemmatizer # Will replace with spaCy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    import spacy # Import spaCy
    
    # Download necessary NLTK data (if not already downloaded)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Load the English spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        !python -m spacy download en_core_web_sm
        nlp = spacy.load("en_core_web_sm")
    
    
    stemmer = PorterStemmer()
    # lemmatizer = WordNetLemmatizer() # Will use spaCy for lemmatization
    stop_words = set(stopwords.words('english'))
    
    # Define different preprocessing functions
    def preprocess_text_basic(text):
        """Basic preprocessing: lowercase, remove non-alphanumeric."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text) # Keep only lowercase letters and numbers
        return text
    
    def preprocess_text_stopwords(text):
        """Preprocessing with stop word removal."""
        if pd.isna(text):
            return ""
        text = preprocess_text_basic(text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    
    def preprocess_text_stemming(text):
        """Preprocessing with stemming."""
        if pd.isna(text):
            return ""
        text = preprocess_text_stopwords(text) # Start after stop words removed
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        return text
    
    def preprocess_text_lemmatization_spacy(text):
        """Preprocessing with spaCy lemmatization."""
        if pd.isna(text):
            return ""
        # Use basic preprocessing and stop word removal first
        text = preprocess_text_stopwords(text)
        doc = nlp(text)
        # Keep only the lemma for each token, and join them back
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        return lemmatized_text
    
    
    # Apply different preprocessing methods to the 'soup' column
    # Create new columns or dataframes for experimentation to avoid overwriting original 'soup'
    merged_df_preprocessed = merged_df.copy()
    
    print("Applying preprocessing techniques...")
    merged_df_preprocessed['soup_basic'] = merged_df_preprocessed['soup'].apply(preprocess_text_basic)
    merged_df_preprocessed['soup_stopwords'] = merged_df_preprocessed['soup'].apply(preprocess_text_stopwords)
    merged_df_preprocessed['soup_stemmed'] = merged_df_preprocessed['soup'].apply(preprocess_text_stemming)
    # Use the new spaCy lemmatization function
    merged_df_preprocessed['soup_lemmatized'] = merged_df_preprocessed['soup'].apply(preprocess_text_lemmatization_spacy)
    print("Preprocessing complete.")
    
    # Now, we can re-run TF-IDF and recommendation generation using these new preprocessed 'soup' columns
    # and qualitatively compare the recommendations.
    
    print("\nGenerating recommendations using different preprocessed text:")
    
    preprocessing_experiments = {
        'Original_soup': merged_df['soup'].fillna(''), # Use original soup with TfidfVectorizer stop_words
        'Basic_Preprocessing': merged_df_preprocessed['soup_basic'],
        'Stopwords_Removed': merged_df_preprocessed['soup_stopwords'],
        'Stemming': merged_df_preprocessed['soup_stemmed'],
        'Lemmatization_spaCy': merged_df_preprocessed['soup_lemmatized'] # Use the spaCy lemmatized text
    }
    
    sample_movie_for_preprocessing = "Pirates of the Caribbean: At World\'s End"
    num_recommendations_preprocessing_exp = 5
    
    # Reuse the get_content_based_recommendations function.
    # Note: This function find cosine_sim is calculated on TF-IDF of 'soup'.
    # We are recalculating cosine_sim for each preprocessed version.
    
    for name, processed_soup in preprocessing_experiments.items():
         print(f"\n--- Preprocessing Experiment: {name} ---")
         try:
             # Refit TF-IDF for the current preprocessed text
             # If stop words were already removed in preprocessing, don't use TfidfVectorizer stop_words
             if name in ['Stopwords_Removed', 'Stemming', 'Lemmatization_spaCy']:
                  current_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=86621) # Don't use stop_words here
             else:
                  current_tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=86621) # Use stop_words for others
    
    
             current_tfidf_matrix = current_tfidf.fit_transform(processed_soup)
             current_cosine_sim = linear_kernel(current_tfidf_matrix, current_tfidf_matrix)
    
             # Use the existing get_content_based_recommendations function
             # Need to temporarily modify merged_df to use the processed_soup for the recommendation function's internal index mapping
             original_soup_col = merged_df['soup'] # Store original soup
             merged_df['soup'] = processed_soup # Temporarily replace with processed soup
    
             current_recommendations = get_content_based_recommendations(
                 sample_movie_for_preprocessing,
                 merged_df,
                 cosine_sim=current_cosine_sim,
                 num_recommendations=num_recommendations_preprocessing_exp
             )
    
             # Restore original soup column
             merged_df['soup'] = original_soup_col
    
             print(f"Recommendations for '{sample_movie_for_preprocessing}' with {name}:")
             display(current_recommendations)
    
             print(f"Qualitative Assessment for {name}: [Observe the recommendations above and note changes.]")
    
         except Exception as e:
             print(f"Error running preprocessing experiment {name}: {e}")
    
    merged_df.columns
    
    # Incorporate Additional Features into 'soup'
    import json
    
    # Function to safely extract names from JSON-like strings
    def extract_names(json_string):
        if isinstance(json_string, str):
            try:
                list_of_dicts = json.loads(json_string)
                return ' '.join([d['name'].replace(" ", "") for d in list_of_dicts]) # Remove spaces in names for single tokens
            except (json.JSONDecodeError, TypeError):
                return ''
        return ''
    
    # Apply the extraction function to relevant columns
    merged_df['production_companies_names'] = merged_df['production_companies'].apply(extract_names)
    merged_df['production_countries_names'] = merged_df['production_countries'].apply(extract_names)
    merged_df['spoken_languages_names'] = merged_df['original_language'].apply(extract_names)
    
    
    # Create a new 'enhanced_soup' column by combining the original 'soup' with the new features
    # handle potential None/NaN values before combining
    merged_df['enhanced_soup'] = merged_df['soup'].fillna('') + ' ' + \
                               merged_df['production_companies_names'].fillna('') + ' ' + \
                               merged_df['production_countries_names'].fillna('') + ' ' + \
                               merged_df['spoken_languages_names'].fillna('')
    
    print("Created 'enhanced_soup' column with additional features.")
    print("Sample of enhanced_soup for the first movie:")
    print(merged_df['enhanced_soup'].iloc[0])
    
    
    
    # Refit TF-IDF using the enhanced_soup
    # Using the original TF-IDF parameters (stop_words, ngram_range) for now
    tfidf_enhanced = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=86621)
    tfidf_matrix_enhanced = tfidf_enhanced.fit_transform(merged_df['enhanced_soup'])
    cosine_sim_enhanced = linear_kernel(tfidf_matrix_enhanced, tfidf_matrix_enhanced)
    
    
    # sample movie to evaluate recommendations qualitatively with the enhanced features
    sample_movie_for_enhanced = 'Avatar'
    num_recommendations_enhanced_exp = 5
    
    # Use the existing get_content_based_recommendations function with the new cosine similarity matrix
    # Temporarily replace the 'soup' column with 'enhanced_soup' for the function to work correctly
    # Create a copy of merged_df to avoid modifying the original DataFrame
    merged_df_copy = merged_df.copy()
    merged_df_copy['soup'] = merged_df_copy['enhanced_soup']
    
    
    enhanced_recommendations = get_content_based_recommendations(
        sample_movie_for_enhanced,
        merged_df_copy, # Pass the copy of the DataFrame
        cosine_sim=cosine_sim_enhanced,
        num_recommendations=num_recommendations_enhanced_exp
    )
    
    # No need to restore the original 'soup' column in merged_df since we worked on a copy
    print(f"\nRecommendations for '{sample_movie_for_enhanced}' using Enhanced Features:")
    display(enhanced_recommendations)
    
    import matplotlib.pyplot as plt
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping
    from tensorflow.keras.regularizers import l2 # Import l2 regularizer
    from IPython.display import display
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Dummy get_autoencoder_recommendations for demonstration if not available
    def get_autoencoder_recommendations(movie_title, df, latent_cosine_sim, num_recommendations=5):
        if movie_title not in df['title'].values:
            return f"Movie '{movie_title}' not found in the dataset."
        idx = df[df['title'] == movie_title].index[0]
        sim_scores = list(enumerate(latent_cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        return df['title'].iloc[movie_indices]
    
    
    print("\n## Experimenting with Autoencoder Hyperparameters (with Early Stopping and Weight Decay)\n")
    
    # Autoencoder Model Parameter Experiments
    # Increased epochs to 100 to allow EarlyStopping to work effectively
    autoencoder_experiments = {
        'Original_Optimized': {'encoding_dim': 128, 'epochs': 100, 'batch_size': 256},
        'Small_Latent_Space': {'encoding_dim': 64, 'epochs': 100, 'batch_size': 256},
        'Large_Latent_Space': {'encoding_dim': 256, 'epochs': 100, 'batch_size': 256},
    }
    
    # A movie to evaluate recommendations qualitatively
    sample_movie_for_autoencoder = 'The Matrix'
    num_recommendations_autoencoder_exp = 5
    
    # Define the EarlyStopping callback
    # It will monitor validation loss and stop if there's no improvement after 5 epochs.
    # It will also restore the best weights found during training.
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print(f"Generating recommendations for '{sample_movie_for_autoencoder}' using different Autoencoder settings:")
    current_encoder = None # To hold the final or best encoder for saving later
    
    for name, params in autoencoder_experiments.items():
        print(f"\n--- Autoencoder Experiment: {name} ---")
        try:
            # Build the Autoencoder Model with L2 Regularization (Weight Decay)
            input_layer = Input(shape=(tfidf_scaled.shape[1],))
            encoder_layer = Dense(params['encoding_dim'], activation='relu', kernel_regularizer=l2(1e-5))(input_layer)
            decoder_layer = Dense(tfidf_scaled.shape[1], activation='sigmoid', kernel_regularizer=l2(1e-5))(encoder_layer)
            optimized_autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
    
            optimized_autoencoder.compile(optimizer='adam', loss='mse')
    
            # Train the Autoencoder with the EarlyStopping callback
            print(f"Training Autoencoder for {name} with up to {params['epochs']} epochs...")
            history = optimized_autoencoder.fit(tfidf_scaled, tfidf_scaled,
                                                epochs=params['epochs'],
                                                batch_size=params['batch_size'],
                                                shuffle=True,
                                                validation_split=0.1, # Use 10% of data for validation
                                                callbacks=[early_stopping], # callback here
                                                verbose=0)
            print(f"Training Complete. Stopped at epoch: {len(history.history['loss'])}")
    
            # --- PLOT THE LOSS CURVE (WITH VALIDATION LOSS) ---
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Loss Curve for Experiment: {name}')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error (Loss)')
            plt.legend()
            plt.grid(True)
            plt.show()
            # --- END OF PLOTTING ---
    
            # Get the Encoder model and latent features
            current_encoder = Model(inputs=input_layer, outputs=encoder_layer)
            current_latent_features = current_encoder.predict(tfidf_scaled, verbose=0)
    
            # Calculate cosine similarity matrix on the latent features
            current_latent_cosine_sim = cosine_similarity(current_latent_features)
    
            # Generate recommendations
            current_autoencoder_recommendations = get_autoencoder_recommendations(
                sample_movie_for_autoencoder,
                merged_df,
                latent_cosine_sim=current_latent_cosine_sim,
                num_recommendations=num_recommendations_autoencoder_exp
            )
    
            print(f"Recommendations for '{sample_movie_for_autoencoder}' with {name} Autoencoder:")
            display(current_autoencoder_recommendations)
    
        except Exception as e:
            print(f"Error running Autoencoder experiment {name}: {e}")
    
    # Experiment with kNN hyperparameters
    
    print("\n## Experimenting with kNN Hyperparameters\n")
    
    # kNN Hyperparameter Experiments
    knn_experiments = {
        'Original_k10': {'n_neighbors': 10}, # Original setting
        'k_5': {'n_neighbors': 5},       # Fewer neighbors
        'k_20': {'n_neighbors': 20},     # More neighbors
        'k_50': {'n_neighbors': 50}      # Even more neighbors
    }
    
    # Choose a sample query (can be a movie title or keyword)
    sample_query_knn = 'science fiction action'
    num_recommendations_knn_exp = 10 # Number of recommendations to display
    
    print(f"Generating kNN recommendations for query '{sample_query_knn}' using different 'n_neighbors' settings:")
    
    for name, params in knn_experiments.items():
        print(f"\n--- kNN Experiment: {name} ---")
        try:
            # Generate recommendations using the current n_neighbors
            current_knn_recommendations = generate_recommendations_knn(
                sample_query_knn,
                merged_df,
                tfidf_matrix, # Use the original TF-IDF matrix
                num_recommendations=params['n_neighbors'] # Set num_recommendations to n_neighbors for this test
            )
    
            print(f"Recommendations for '{sample_query_knn}' with {name} kNN ({params['n_neighbors']} neighbors):")
            display(current_knn_recommendations)
    
            # Qualitative assessment (manual observation of displayed recommendations)
            print(f"Qualitative Assessment for {name}: [Observe the recommendations above and note changes compared to other k values]")
    
        except Exception as e:
            print(f"Error running kNN experiment {name}: {e}")
    
    # @title kNN Confidence Score (Cosine Similarity)
    
    from matplotlib import pyplot as plt
    current_knn_recommendations['Confidence Score (Cosine Similarity)'].plot(kind='line', figsize=(8, 4), title='kNN Confidence Score (Cosine Similarity)')
    plt.gca().spines[['top', 'right']].set_visible(False)
    
    # =========================
    # Re-evaluate Optimal K (Elbow) and Perform Clustering
    # =========================
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD, PCA # Import reducers
    from tensorflow.keras.models import Model # Import Keras Model to check encoder output shape
    import pandas as pd # pandas is imported
    
    print("\n## Re-evaluating Optimal Number of Clusters (Elbow Method) for KMeans\n")
    
    if 'current_encoder' not in globals() or not isinstance(current_encoder, Model) or current_encoder.output_shape[1] != 256:
        print("current_encoder (256-feature output) not found or is incorrect. Cannot proceed.")
        # Exit or raise an error if encoder is missing
        raise NameError("Required 'current_encoder' (256-feature output) is not available.")
    
    # Regenerate latent features from the 256-feature encoder
    print("Generating 256-feature latent features from current_encoder...")
    latent_features_256d = current_encoder.predict(tfidf_scaled, verbose=0)
    print(f"256-feature latent features generated. Shape: {latent_features_256d.shape}")
    
    
    # Determine the feature space KMeans was fitted on or should be fitted on
    # This depends on whether a reducer was used before fitting KMeans previously.
    # We need to check the dimensionality of features_for_kmeans if it exists,
    # or a target dimension if the previous steps defined one (e.g., N_COMPONENTS=128).
    
    
    target_kmeans_dim = 128 # KMeans should operate on 128 features to HF deployment support
    
    
    # If the encoder output dim is different from the target KMeans dim, a reducer is needed
    reducer_needed = latent_features_256d.shape[1] != target_kmeans_dim
    
    if reducer_needed:
        print(f"Encoder output ({latent_features_256d.shape[1]}d) differs from target KMeans dim ({target_kmeans_dim}d). A reducer is needed.")
        # Create and fit a reducer (e.g., TruncatedSVD) to get to the target dimension
        # Fit the reducer on the 256-feature latent space
        reducer = TruncatedSVD(n_components=target_kmeans_dim, random_state=42)
        print(f"Fitting TruncatedSVD reducer from {latent_features_256d.shape[1]}d to {target_kmeans_dim}d...")
        features_for_kmeans = reducer.fit_transform(latent_features_256d)
        print(f"Features for KMeans (reduced) generated. Shape: {features_for_kmeans.shape}")
    
        # Store the fitted reducer in globals so query_clustering_recommendations can access it
        globals()['reducer'] = reducer
    
    else:
        print(f"Encoder output ({latent_features_256d.shape[1]}d) matches target KMeans dim ({target_kmeans_dim}d). No reducer needed.")
        features_for_kmeans = latent_features_256d
        # Ensure reducer is None or not needed in globals for query function
        if 'reducer' in globals():
            del globals()['reducer']
    
    
    # --- Elbow Method (run on features_for_kmeans) ---
    inertia = []
    cluster_range = range(1, 150, 10)  # 1, 11, 21, ... 141
    
    print(f"\nCalculating inertia for different numbers of clusters (using {features_for_kmeans.shape[1]}d feature space)...")
    X_kmeans_space = features_for_kmeans  # Use the features KMeans will be fitted on
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in cluster_range:
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_tmp.fit(X_kmeans_space)
            inertia.append(km_tmp.inertia_)
            print(f"Completed KMeans for k={k}, Inertia: {km_tmp.inertia_:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(list(cluster_range), inertia, marker='o', linestyle='-')
    plt.title(f'Elbow Method for Optimal Number of Clusters (on {features_for_kmeans.shape[1]}d features)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.xticks(list(cluster_range))
    plt.grid(True)
    plt.show()
    
    print("\nObserve the plot again to find the 'elbow' point.")
    print("This point is a guideline for choosing the number of clusters.")
    
    # =========================
    # Perform Clustering with Chosen Number and Assess Recommendations
    # =========================
    
    n_clusters_tuned = 10 # chosen value
    
    print(f"\nPerforming KMeans clustering with chosen number of clusters ({n_clusters_tuned})...")
    
    kmeans_model_tuned = KMeans(n_clusters=n_clusters_tuned, random_state=42, n_init=10)
    cluster_labels_tuned = kmeans_model_tuned.fit_predict(X_kmeans_space) # Fit on the prepared KMeans features
    
    # Ensure merged_df is available and not empty before adding columns
    if 'merged_df' in globals() and not merged_df.empty:
        merged_df['kmeans_cluster_tuned'] = cluster_labels_tuned
        print(f"Clustering complete with {n_clusters_tuned} clusters.")
        print("\nDistribution of movies per new cluster:")
        print(merged_df['kmeans_cluster_tuned'].value_counts().sort_index())
    
        # =========================
        # Query → Recos using tuned KMeans (same space as elbow/KMeans)
        # =========================
        # This uses the fitted TF-IDF, encoder, and (optional) reducer to map the text query
        # into the exact KMeans feature space, assigns the cluster, then ranks items in that cluster
        # by cosine similarity in that same space.
        def query_clustering_recommendations(
            query_text: str,
            df,
            tfidf_vectorizer,
            encoder_model, # Expects 256d output
            kmeans_model, # Expects the dimension KMeans was fitted on (e.g., 128d)
            reducer=None, # The reducer that transforms 256d to kmeans_model.n_features_in_
            num_recommendations: int = 10
        ):
            """
            Generate recommendations for a free-text query using tuned KMeans clusters.
    
            Steps:
              TF-IDF(query) -> encoder -> (optional reducer) => query_vec_in_kmeans_space
              cluster = kmeans.predict(query_vec)
              rank members of that cluster by cosine similarity in the same space
            """
            if not isinstance(query_text, str) or not query_text.strip():
                return df.iloc[0:0]
    
            # 1) TF-IDF
            # Ensure query_text is in a list for transform
            q_tfidf = tfidf_vectorizer.transform([query_text])
            # ensure dense if the encoder expects dense (Keras Dense layer does)
            q_dense = q_tfidf.toarray()
    
            # 2) Encoder → 256-D
            # Ensure encoder_model is defined and fitted
            if encoder_model is None:
                 print("Error: encoder_model is not available.")
                 return df.iloc[0:0]
            q_emb = encoder_model.predict(q_dense, verbose=0)  # shape: (1, 256)
    
            # 3) Optional reducer to match KMeans expected dim
            # Check if reducer is needed based on KMeans expected input features
            # and if the passed reducer is the correct one (transforming 256d to expected dim)
            kmeans_expected_dim = kmeans_model.n_features_in_ if hasattr(kmeans_model, 'n_features_in_') else None
    
            if kmeans_expected_dim is not None and q_emb.shape[1] != kmeans_expected_dim:
                # Reducer is needed if encoder output doesn't match KMeans input
                if reducer is not None and hasattr(reducer, 'transform'):
                    # Check if the reducer is expected to transform from 256d
                     if hasattr(reducer, 'n_features_in_') and reducer.n_features_in_ == q_emb.shape[1]:
                        print(f"Applying reducer from {q_emb.shape[1]}d to {reducer.n_components}d...")
                        q_kspace = reducer.transform(q_emb)  # shape: (1, n_components)
                     else:
                        print(f"Error: Provided reducer expects {getattr(reducer, 'n_features_in_', 'unknown')} features, but query embedding has {q_emb.shape[1]}. Cannot apply reducer.")
                        return df.iloc[0:0]
                else:
                    print(f"Error: Reducer is needed to transform {q_emb.shape[1]}d to {kmeans_expected_dim}d but is not provided or not fitted correctly.")
                    return df.iloc[0:0]
            else:
                # No reducer needed, or encoder output matches KMeans input
                q_kspace = q_emb  # shape matches kmeans_model.n_features_in_
    
    
            # Check if the transformed query embedding matches KMeans expected dimension
            if hasattr(kmeans_model, 'n_features_in_') and q_kspace.shape[1] != kmeans_model.n_features_in_:
                print(f"Error: Query embedding in KMeans space ({q_kspace.shape[1]}d) does not match KMeans expected input ({kmeans_model.n_features_in_}d).")
                return df.iloc[0:0]
    
    
            # 4) Predict cluster
            # Ensure kmeans_model is defined and fitted
            if kmeans_model is None or not hasattr(kmeans_model, 'predict'):
                 print("Error: kmeans_model is not available or not fitted.")
                 return df.iloc[0:0]
            cluster_id = int(kmeans_model.predict(q_kspace)[0])
    
            # 5) Members in that cluster
            # Ensure df has the correct cluster labels column
            cluster_col_name = 'kmeans_cluster_tuned' # Use the tuned column name
            if cluster_col_name not in df.columns:
                 print(f"Error: Cluster column '{cluster_col_name}' not found in DataFrame.")
                 return df.iloc[0:0]
    
            members_idx = df.index[df[cluster_col_name] == cluster_id].tolist()
            if not members_idx:
                print(f"No members found in cluster {cluster_id}.")
                return df.iloc[0:0]
    
            # 6) Rank by cosine similarity in KMeans space
            # Need the features for KMeans for the members of the cluster
            # X_kmeans_space contains features for ALL movies. Filter it for cluster members.
            if X_kmeans_space is None or X_kmeans_space.shape[0] != df.shape[0]:
                 print("Error: X_kmeans_space is not available or does not match DataFrame size.")
                 return df.iloc[0:0]
    
            member_vecs = X_kmeans_space[members_idx] # Select features for cluster members
    
            if member_vecs.shape[0] == 0:
                 print(f"No feature vectors found for members in cluster {cluster_id}.")
                 return df.iloc[0:0]
    
            # Calculate similarity between the single query vector and all member vectors
            sims = cosine_similarity(q_kspace, member_vecs).flatten()
    
            # Get indices of top recommendations among cluster members
            # Exclude the query itself if it's in the results (cosine sim of 1.0)
            # However, for a free-text query, the query itself won't be in the dataset,
            # so we just sort and take the top N.
            order = np.argsort(-sims)[:num_recommendations]
            top_idx = [members_idx[i] for i in order]
            top_scores = sims[order]
    
            # Ensure required display columns exist in df
            required_display_cols = ['title', 'overview_sentiment_score', 'genres', 'keywords']
            if not all(col in df.columns for col in required_display_cols):
                 print(f"Error: Required display columns ({required_display_cols}) not found in DataFrame.")
                 # Return partial data if possible, or empty
                 available_cols = [col for col in required_display_cols if col in df.columns]
                 if available_cols:
                     out = df.loc[top_idx, available_cols].copy()
                     out.insert(0, 'cluster_id', cluster_id)
                     out.insert(1, 'cluster_similarity', top_scores)
                     return out
                 else:
                     return df.iloc[0:0]
    
    
            out = df.loc[top_idx, required_display_cols].copy()
            out.insert(0, 'cluster_id', cluster_id)
            out.insert(1, 'cluster_similarity', top_scores)
            return out
    
        # Figure out which reducer (if any) you used for KMeans space
        # The reducer should transform from 256d (encoder output) to the dimension KMeans expects.
        # We explicitly create and fit the reducer above if needed, and store it in globals()['reducer'].
        # The query_clustering_recommendations function expects this reducer if needed.
        reducer_for_query = globals().get('reducer', None)
    
    
        # Example query
        clustering_search_query_tuned = 'space adventure'
    
        print(f"\nGenerating Clustering Recommendations for query '{clustering_search_query_tuned}' using {n_clusters_tuned} clusters:")
        clustering_recommendations_tuned = query_clustering_recommendations(
            clustering_search_query_tuned,
            merged_df,
            current_tfidf,            #  fitted TF-IDF (needed for query vectorization)
            current_encoder,          #  trained encoder (256-D output)
            kmeans_model_tuned,       # tuned KMeans fitted on X_kmeans_space
            reducer=reducer_for_query, # Pass the fitted reducer if needed
            num_recommendations=10
        )
        display(clustering_recommendations_tuned)
    
        print("\nQualitative Assessment for Clustering Tuning: [Observe the recommendations and their clusters.]")
        print("Consider if the movies within the clusters seem more related with the new number of clusters.")
        print("Also, check if the recommendations for the sample query are relevant based on the identified cluster.")
    
    else:
        print("\nSkipping KMeans tuning and recommendation generation due to missing merged_df.")
    
    # @title Step 4: Implement the Agent
    
    print("## Implementing the Agentic Flow\n")
    
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_openai import ChatOpenAI # Correct import from langchain-openai
    from langchain.prompts import ChatPromptTemplate
    # get_autoencoder_recommendations_tool, get_clustering_recommendations_tool) are defined in previous cells.
    
    # --- Check if prerequisite tool functions are defined ---
    required_tool_functions = [
        'detect_sentiment',
        'get_content_recommendations_tool',
        'get_autoencoder_recommendations_tool',
        'get_clustering_recommendations_tool'
    ]
    
    all_tools_defined = True
    for func_name in required_tool_functions:
        if func_name not in globals():
            print(f"Error: Required tool function '{func_name}' is not defined.")
            all_tools_defined = False
    
    if not all_tools_defined:
        print("\nPlease run the cells defining the sentiment detection tool (cell 6e475914) and recommendation tools (cell 8c77ca40) before running this cell.")
        # Stop execution here gracefully if tools are missing
        raise NameError("Required tool functions are not defined. Cannot proceed with agent creation.")
    
    
    # --- Proceed with agent creation if all tools are defined ---
    if all_tools_defined:
        # Ensure the LLM for the agent is initialized
        try:
            llm_agent = ChatOpenAI(temperature=0, model="gpt-4o")
            print("OpenAI LLM for agent initialized.")
        except Exception as e:
            print(f"Error initializing OpenAI LLM for agent: {e}")
            llm_agent = None # Set to None if initialization fails
    
    
        # List of tools available to the agent
        # The tools themselves have checks for component availability
        tools = [
            detect_sentiment,
            get_content_recommendations_tool,
            get_autoencoder_recommendations_tool,
            get_clustering_recommendations_tool
        ]
    
        # Define the agent's prompt
        # Modified prompt structure to a more standard format for tool-calling agents
        # placing chat_history and agent_scratchpad after the human input message.
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a movie recommendation assistant. Your goal is to help the user find movies they might like.
            You have access to tools to:
            1. Detect the sentiment of the user's input.
            2. Get movie recommendations based on content (keywords/query).
            3. Get movie recommendations based on Autoencoder latent features (requires a specific movie title).
            4. Get movie recommendations based on Clustering/KMeans (keywords/query).
    
            Based on the user's request, decide which tool(s) are most appropriate.
            - If the user expresses a strong sentiment about the *type* of movie they want (e.g., "happy movie", "sad film"), use the sentiment detection tool first. You can then use this sentiment to potentially refine your recommendation approach or the query you pass to the recommendation tools.
            - If the user mentions a specific movie title, consider using the Autoencoder tool or Content-based tool with that title.
            - If the user provides keywords or describes the content they like, use the Content-based tool or Clustering tool with the keywords.
            - You can use multiple tools if needed to understand the request or provide different perspectives on recommendations.
            - When using a recommendation tool, provide the necessary input (a movie title for Autoencoder, a query for Content/Clustering).
            - After getting recommendations from a tool, present them clearly to the user.
            - If no relevant tools are applicable or tools fail, inform the user you cannot provide recommendations for that request.
            """),
            ("human", "{input}"), # Place the user input here
            ("placeholder", "{chat_history}"), # Place user chat history after input for future reference
            ("placeholder", "{agent_scratchpad}"), # Place agent scratchpad (internal thoughts) after chat history for future reference
        ])
    
        # Create the agent
        if llm_agent is not None:
            agent = create_tool_calling_agent(llm_agent, tools, agent_prompt)
    
            # Create the agent executor
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Set verbose=True to see agent's thought process
            print("\nAgent and AgentExecutor created.")
            print("We can now interact with the agent using agent_executor.invoke({'input': 'User movie request here'})")
    
        else:
            print("\nAgent could not be created due to LLM initialization failure.")
    
    agent_executor.invoke({'input': 'I like adventure and romance movies, not horror or sci-fi'})
    
    # --- Save All Essential Models & Data for Frontend Integration ---
    
    import os
    import joblib
    import pandas as pd
    
    # Optionally, for SentenceTransformer or Keras models:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None
    try:
        from tensorflow import keras
    except ImportError:
        keras = None
    
    # ---- Step 1: Define Save Directory ----
    save_dir = './recommendation_models'  # Change as needed
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving models to directory: {save_dir}")
    
    # ---- Step 2: Save TF-IDF Vectorizer ----
    if 'tfidf' in globals():
        tfidf_path = os.path.join(save_dir, 'tfidf_vectorizer.joblib')
        print("vocab size:", len(current_tfidf.vocabulary_))  # should be 86621
        print(current_tfidf.get_params())
        joblib.dump(current_tfidf, tfidf_path)
        print(f"Saved TF-IDF Vectorizer to {tfidf_path}")
    else:
        print("TF-IDF Vectorizer not found. Skipping save.")
    
    # ---- Step 3: Save Autoencoder Encoder Model ----
    if 'current_encoder' in globals() and keras is not None:
        encoder_path = os.path.join(save_dir, 'autoencoder_encoder_model.keras')
        current_encoder.save(encoder_path)
        print(f"Saved Autoencoder Encoder Model to {encoder_path}")
    else:
        print("Autoencoder Encoder Model not found or Keras not installed. Skipping save.")
    
    # ---- Step 4: Save KMeans Model ----
    if 'kmeans_model_tuned' in globals():
        kmeans_path = os.path.join(save_dir, 'kmeans_model.joblib')
        joblib.dump(kmeans_model_tuned, kmeans_path)
        print(f"Saved KMeans Model to {kmeans_path}")
    else:
        print("KMeans Model not found. Skipping save.")
    
    # ---- Step 5: Save Sentence-BERT or Embedding Model ----
    if 'model' in globals():
        sentence_bert_path = os.path.join(save_dir, 'sentence_bert_model')
        # If it's a SentenceTransformer, save using its .save() method
        if SentenceTransformer is not None and isinstance(model, SentenceTransformer):
            model.save(sentence_bert_path)
            print(f"Saved Sentence-BERT Model to {sentence_bert_path}")
        # Otherwise, try saving as a Keras model (optional fallback)
        elif keras is not None and hasattr(model, "save"):
            model.save(sentence_bert_path)
            print(f"Saved model using Keras save() to {sentence_bert_path}")
        else:
            print("Model type unknown for 'model'. Not saved.")
    else:
        print("Sentence-BERT Model not found. Skipping save.")
    
    # ---- Step 6: Save Merged Movie Data (with essential columns) ----
    if 'merged_df' in globals() and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
        essential_cols = [
            'id', 'title', 'genres', 'keywords', 'overview',
            'overview_sentiment_score', 'soup', 'enhanced_soup'
        ]
        cols_to_save = [col for col in essential_cols if col in merged_df.columns]
        if cols_to_save:
            merged_df_path = os.path.join(save_dir, 'merged_movie_data.csv')
            merged_df[cols_to_save].to_csv(merged_df_path, index=False)
            print(f"Saved essential merged_df columns to {merged_df_path}")
        else:
            print("No essential columns found in merged_df to save.")
    else:
        print("merged_df not found or is empty. Skipping save.")
    
    # ---- Step 7: Save Indices Series (for mapping in frontend/backend) ----
    if 'indices' in globals() and hasattr(indices, 'empty') and not indices.empty:
        indices_path = os.path.join(save_dir, 'indices.pkl')
        indices.to_pickle(indices_path)
        print(f"Saved indices Series to {indices_path}")
    else:
        print("indices Series not found or is empty. Skipping save.")
    
    
    print("\nSaving process complete.")
    print(f"All files are available in: {save_dir}")
    print("To use these models in a separate application, load them using the respective libraries (joblib.load, keras.models.load_model, SentenceTransformer, pd.read_csv, pd.read_pickle, etc.).")

