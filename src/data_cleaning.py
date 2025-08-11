def run_data_cleaning():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
    
    movies_df = pd.read_csv('/content/movies.csv')
    
    movies_df.head(-5)
    
    credits_df = pd.read_csv('/content/credits.csv')
    credits_df.head(-5)
    
    # @title popularity vs revenue
    
    from matplotlib import pyplot as plt
    movies_df.plot(kind='scatter', x='popularity', y='revenue', s=32, alpha=.8)
    plt.gca().spines[['top', 'right',]].set_visible(False)
    
    # @title popularity vs runtime
    
    from matplotlib import pyplot as plt
    movies_df.plot(kind='scatter', x='popularity', y='runtime', s=32, alpha=.8)
    plt.gca().spines[['top', 'right',]].set_visible(False)
    
    # @title popularity vs budget
    
    from matplotlib import pyplot as plt
    movies_df.plot(kind='scatter', x='popularity', y='budget', s=32, alpha=.8)
    plt.gca().spines[['top', 'right',]].set_visible(False)
    
    print("Columns in movies_df:", movies_df.columns)
    print("Columns in credits_df:", credits_df.columns)
    print("Columns in merged_df:", merged_df.columns)
    
    print("\nSample of movies_df:")
    display(movies_df.head())
    
    print("\nSample of credits_df:")
    display(credits_df.head())
    
    print("\nSample of merged_df:")
    display(merged_df.head())
    
    print("\nValue counts for 'vote_count' in merged_df:")
    print(merged_df['vote_count'].value_counts().head())
    
    print("\nValue counts for 'vote_average' in merged_df:")
    print(merged_df['vote_average'].value_counts().head())
    
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

