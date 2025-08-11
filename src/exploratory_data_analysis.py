"""
Exploratory Data Analysis for Movie Recommendations
install nltk
!pip install nltk
"""

def run_exploratory_data_analysis():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd


    # Get the shape of each dataframe (number of rows and columns)
    print("\nShape of Movies DataFrame:", movies_df.shape)
    print("Shape of Credits DataFrame:", credits_df.shape)
    
    # Get information about the data types and non-null values
    print("\nInfo for Movies DataFrame:")
    movies_df.info()
    
    print("\nInfo for Credits DataFrame:")
    credits_df.info()
    
    # Get descriptive statistics for numerical columns
    print("\nDescription for Movies DataFrame:")
    print(movies_df.describe())
    
    print("\nDescription for Credits DataFrame:")
    print(credits_df.describe())
    
    # Check for missing values
    print("\nMissing values in Movies DataFrame:")
    print(movies_df.isnull().sum())
    
    print("\nMissing values in Credits DataFrame:")
    print(credits_df.isnull().sum())
    
    # Check for duplicate rows
    print("\nNumber of duplicate rows in Movies DataFrame:", movies_df.duplicated().sum())
    print("Number of duplicate rows in Credits DataFrame:", credits_df.duplicated().sum())
    
    
    
    # Explore the distribution of key columns using histograms
    movies_df.hist(figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    # Explore the distribution of categorical columns using bar plots
    
    # Explore the distribution of budget, revenue, and runtime using box plots
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.boxplot(y=movies_df['budget'])
    plt.title('Box plot of Budget')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(y=movies_df['revenue'])
    plt.title('Box plot of Revenue')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(y=movies_df['runtime'])
    plt.title('Box plot of Runtime')
    plt.tight_layout()
    plt.show()
    
    # Visualize the correlation matrix of numerical columns in movies_df
    plt.figure(figsize=(10, 8))
    # Select only numerical columns for correlation matrix
    numerical_movies_df = movies_df.select_dtypes(include=np.number)
    sns.heatmap(numerical_movies_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Columns in Movies DataFrame')
    plt.show()
    
    # @title Perform feature engineering with movies_df and credit_df to prepare for generating recommendations
    
    # Combine the dataframes
    movies_df['title'] = movies_df['title'].astype(str)
    credits_df['title'] = credits_df['title'].astype(str)
    
    # Merge the dataframes on the 'title' column
    merged_df = movies_df.merge(credits_df, on='title')
    
    # Drop irrelevant columns for recommendations
    #merged_df.drop(['homepage', 'status', 'production_countries', 'spoken_languages', 'tagline', 'poster_path', 'production_companies'], axis=1, inplace=True)
    merged_df.drop(['homepage', 'spoken_languages', 'tagline'], axis=1, inplace=True)
    
    # Handle missing values (example: fill NaNs in 'runtime' with the mean)
    merged_df['runtime'].fillna(merged_df['runtime'].mean(), inplace=True)
    merged_df['vote_average'].fillna(0, inplace=True)
    merged_df['vote_count'].fillna(0, inplace=True)
    
    # Extract relevant information from nested JSON strings
    import json
    
    def parse_json(json_string):
        try:
            list_of_dicts = json.loads(json_string)
            return [item['name'] for item in list_of_dicts]
        except (json.JSONDecodeError, TypeError):
            return []
    
    merged_df['genres'] = merged_df['genres'].apply(parse_json)
    merged_df['keywords'] = merged_df['keywords'].apply(parse_json)
    merged_df['cast'] = merged_df['cast'].apply(parse_json)
    merged_df['crew'] = merged_df['crew'].apply(parse_json)
    
    # Keep only the director from the crew list
    def get_director(crew_list):
        for item in crew_list:
            if item == 'Director':
                return item
        return None
    
    merged_df['director'] = merged_df['crew'].apply(lambda x: [i for i in x if i in ['Director']])
    merged_df['director'] = merged_df['director'].apply(lambda x: x[0] if x else None)
    merged_df.drop('crew', axis=1, inplace=True)
    
    
    # Convert lists of strings into space-separated strings for easier processing
    def list_to_string(lst):
        return ' '.join([str(i).replace(" ","") for i in lst])
    
    for feature in ['genres', 'keywords', 'cast']:
        merged_df[feature] = merged_df[feature].apply(list_to_string)
    
    merged_df['director'] = merged_df['director'].apply(lambda x: str(x).replace(" ","") if x else '')
    
    # Create a 'soup' of combined features for TF-IDF or Count Vectorizer
    merged_df['soup'] = merged_df['title'] + merged_df['overview'].fillna('') + ' ' + merged_df['genres'] + ' ' + merged_df['keywords'] + ' ' + merged_df['cast'] + ' ' + merged_df['director'] + merged_df['release_date']
    
    
    merged_df.head(-5)
    
    print("\nInfo after feature engineering:")
    merged_df.info()

    
    # @title perform sentiment analysis
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import nltk
    nltk.download('vader_lexicon')
    
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Function to get sentiment score
    def get_sentiment_score(text):
        if pd.isna(text):
            return 0 # Return 0 for missing overviews
        return analyzer.polarity_scores(str(text))['compound'] # Use compound score as a single metric
    
    # Apply the function to the 'overview' column and create a new column for sentiment score
    merged_df['overview_sentiment_score'] = merged_df['overview'].apply(get_sentiment_score)
    
    print("\nDataFrame with Sentiment Scores:")
    print(merged_df[['title', 'overview', 'overview_sentiment_score']].head())
    
    # Optional: Analyze the distribution of sentiment scores
    plt.figure(figsize=(8, 6))
    sns.histplot(merged_df['overview_sentiment_score'], bins=20, kde=True)
    plt.title('Distribution of Overview Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()
    
    # @title The average sentiment score by genre.
    
    import pandas as pd
    import matplotlib.pyplot as plt
    # Average sentiment score by genre
    # We need to first "explode" the genres list so that each movie's sentiment score is associated with each of its genres
    genre_sentiment = merged_df[['genres', 'overview_sentiment_score']].copy()
    genre_sentiment['genres'] = genre_sentiment['genres'].str.split()
    genre_sentiment = genre_sentiment.explode('genres')
    
    # Now calculate the average sentiment score for each genre
    avg_sentiment_by_genre = genre_sentiment.groupby('genres')['overview_sentiment_score'].mean().sort_values(ascending=False)
    
    print("\nAverage Sentiment Score by Genre:")
    print(avg_sentiment_by_genre.head())
    
    # Visualize average sentiment by genre (top N)
    plt.figure(figsize=(12, 6))
    avg_sentiment_by_genre.head(10).plot(kind='bar')
    plt.title('Average Overview Sentiment Score by Genre (Top 10)')
    plt.xlabel('Genre')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # @title Average sentiment score by year
    merged_df['release_date'] = pd.to_datetime(merged_df['release_date'], errors='coerce')
    
    # Extract the year
    merged_df['release_year'] = merged_df['release_date'].dt.year
    
    # Drop rows with missing or invalid release years
    sentiment_by_year_df = merged_df.dropna(subset=['release_year', 'overview_sentiment_score'])
    
    # Group by year and calculate the mean sentiment score
    avg_sentiment_by_year = sentiment_by_year_df.groupby('release_year')['overview_sentiment_score'].mean().sort_index()
    
    print("\nAverage Sentiment Score by Year:")
    print(avg_sentiment_by_year.head())
    
    # Visualize average sentiment by year
    plt.figure(figsize=(15, 6))
    avg_sentiment_by_year.plot(kind='line')
    plt.title('Average Overview Sentiment Score by Release Year')
    plt.xlabel('Release Year')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # @title Perform recommendations on sentiment score
    
    import pandas as pd
    
    # Function to get movie recommendations based on sentiment score
    def recommend_by_sentiment(title, df, num_recommendations=5):
        # Find the index of the movie with the given title
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        if title not in indices:
            print(f"Movie '{title}' not found in the dataset.")
            return pd.DataFrame() # Return an empty DataFrame if movie not found
    
        idx = indices[title]
    
        # Get the sentiment score of the input movie
        input_sentiment_score = df.loc[idx, 'overview_sentiment_score']
    
        # Calculate the absolute difference in sentiment scores between the input movie and all other movies
        df['sentiment_difference'] = abs(df['overview_sentiment_score'] - input_sentiment_score)
    
        # Sort movies based on the absolute difference in sentiment scores (closest to the input movie's score)
        # Exclude the input movie itself
        recommended_movies = df.sort_values(by='sentiment_difference').head(num_recommendations + 1)
    
        # Filter out the input movie
        recommended_movies = recommended_movies[recommended_movies['title'] != title]
    
        # Return the top recommendations
        return recommended_movies[['title', 'overview_sentiment_score', 'sentiment_difference']]
    
    # @title Get recommendations for a movie based on its sentiment score
    movie_title = 'Avatar' # movie title to generate recommendations
    recommendations = recommend_by_sentiment(movie_title, merged_df)
    
    print(f"\nRecommendations based on sentiment similarity for '{movie_title}':")
    recommendations
    
    movie_title = 'Liar Liar' # movie title to generate recommendations
    recommendations = recommend_by_sentiment(movie_title, merged_df)
    
    print(f"\nRecommendations based on sentiment similarity for '{movie_title}':")
    recommendations
    
    # @title Generate recommendation with plot overview keywords based on the sentiment score
    
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    
    # Initialize the TfidfVectorizer
    # Use the 'soup' column which contains combined text features (overview, genres, keywords, cast, director)
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=86621,
    )
    
    # Construct the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(merged_df['soup'].fillna(''))
    
    print("\nShape of TF-IDF matrix:", tfidf_matrix.shape)
    
    # Calculate the cosine similarity matrix
    # This measures the similarity between movie 'soups'
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    print("Shape of Cosine Similarity matrix:", cosine_sim.shape)
    
    # Create a reverse mapping of movie titles to their indices
    indices = pd.Series(merged_df.index, index=merged_df['title']).drop_duplicates()
    
    # Function to get recommendations based on cosine similarity of the 'soup'
    def get_content_based_recommendations(title, df, cosine_sim=cosine_sim, num_recommendations=10):
        # Get the index of the movie that matches the title
        if title not in indices:
            print(f"Movie '{title}' not found in the dataset for content-based recommendations.")
            return pd.DataFrame()
    
        idx = indices[title]
    
        # Get the pairwise similarity scores for all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
    
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
        # Get the scores of the num_recommendations most similar movies
        # Skip the first element as it is the movie itself
        sim_scores = sim_scores[1:num_recommendations+1]
    
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
    
        # Return the top num_recommendations most similar movies
        return df[['title', 'genres', 'keywords', 'overview_sentiment_score']].iloc[movie_indices]
    
    # Function to generate combined recommendations considering both sentiment and content
    def get_combined_recommendations(title, df, cosine_sim=cosine_sim, num_recommendations=5, sentiment_weight=0.5, content_weight=0.5):
        if title not in indices:
            print(f"Movie '{title}' not found in the dataset.")
            return pd.DataFrame()
    
        idx = indices[title]
        input_sentiment_score = df.loc[idx, 'overview_sentiment_score']
    
        # Get sentiment similarity scores (closer to 0 difference is better)
        # We need to invert this difference to get a similarity score (higher is better)
        # A simple inversion could be 1 - abs_difference, but scaling might be needed
        # For now, let's use the inverse of the rank based on absolute difference
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
        return recommended_movies[['title', 'genres', 'keywords', 'overview_sentiment_score', 'combined_score']].reset_index(drop=True)
    
    # @title Get combined recommendations based on content similarity and sentiment score
    movie_title_for_combined = 'Avatar'  #@param {type:"string"}
    sentiment_weight = 0.5  #@param {type:"slider", min:0.0, max:1.0, step:0.1}
    content_weight = 0.7  #@param {type:"slider", min:0.0, max:1.0, step:0.1}
    num_recommendations_combined = 5  #@param {type:"slider", min:1, max:20, step:1}
    
    
    combined_recommendations = get_combined_recommendations(
        movie_title_for_combined,
        merged_df,
        cosine_sim=cosine_sim,
        num_recommendations=num_recommendations_combined,
        sentiment_weight=sentiment_weight,
        content_weight=content_weight
    )
    
    print(f"\nCombined recommendations (Sentiment weight: {sentiment_weight}, Content weight: {content_weight}) for '{movie_title_for_combined}':")
    combined_recommendations
    
    # @title genres vs keywords
    
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    plt.subplots(figsize=(8, 8))
    df_2dhist = pd.DataFrame({
        x_label: grp['keywords'].value_counts()
        for x_label, grp in combined_recommendations.groupby('genres')
    })
    sns.heatmap(df_2dhist, cmap='viridis')
    plt.xlabel('genres')
    _ = plt.ylabel('keywords')
    
    # @title Plot overview_sentiment_score vs combined_score
    
    from matplotlib import pyplot as plt
    import seaborn as sns
    def _plot_series(series, series_name, series_index=0):
      palette = list(sns.palettes.mpl_palette('Dark2'))
      xs = series['overview_sentiment_score']
      ys = series['combined_score']
    
      plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])
    
    fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
    df_sorted = combined_recommendations.sort_values('overview_sentiment_score', ascending=True)
    for i, (series_name, series) in enumerate(df_sorted.groupby('genres')):
      _plot_series(series, series_name, i)
      fig.legend(title='genres', bbox_to_anchor=(1, 1), loc='upper left')
    sns.despine(fig=fig, ax=ax)
    plt.xlabel('overview_sentiment_score')
    _ = plt.ylabel('combined_score')
    
    # Generate recommendations with reasons by movie name based on the sentiment score and print it as table
    
    def generate_recommendations_with_reasons(title, df, cosine_sim, num_recommendations=5, sentiment_weight=0.5, content_weight=0.5):
        """
        Generates movie recommendations based on combined sentiment and content similarity,
        providing reasons for each recommendation.
    
        Args:
            title (str): The title of the input movie.
            df (pd.DataFrame): The DataFrame containing movie information (merged_df).
            cosine_sim (np.array): The cosine similarity matrix based on the 'soup'.
            num_recommendations (int, optional): The number of recommendations to generate. Defaults to 5.
            sentiment_weight (float, optional): Weight for sentiment similarity. Defaults to 0.5.
            content_weight (float, optional): Weight for content similarity. Defaults to 0.5.
    
        Returns:
            pd.DataFrame: A DataFrame containing the recommended movies and reasons.
        """
        if title not in indices:
            print(f"Movie '{title}' not found in the dataset.")
            return pd.DataFrame()
    
        idx = indices[title]
        input_sentiment_score = df.loc[idx, 'overview_sentiment_score']
        input_genres = df.loc[idx, 'genres']
        input_keywords = df.loc[idx, 'keywords']
    
    
        # Get sentiment similarity scores
        df_temp = df.copy()
        df_temp['sentiment_difference'] = abs(df_temp['overview_sentiment_score'] - input_sentiment_score)
        df_temp['sentiment_rank'] = df_temp['sentiment_difference'].rank(method='min', ascending=True)
        df_temp['normalized_sentiment_sim'] = 1 / df_temp['sentiment_rank']
        df_temp['normalized_sentiment_sim'] = df_temp['normalized_sentiment_sim'] / df_temp['normalized_sentiment_sim'].max()
    
    
        # Get content similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        content_sim_series = pd.Series([score for index, score in sim_scores])
        df_temp['content_sim'] = content_sim_series
        df_temp['normalized_content_sim'] = df_temp['content_sim'] / df_temp['content_sim'].max()
    
    
        # Combine scores using weights
        df_temp['combined_score'] = (df_temp['normalized_sentiment_sim'] * sentiment_weight) + (df_temp['normalized_content_sim'] * content_weight)
    
        # Sort movies based on the combined score
        recommended_movies = df_temp.sort_values(by='combined_score', ascending=False).head(num_recommendations + 1)
    
        # Filter out the input movie
        recommended_movies = recommended_movies[recommended_movies['title'] != title].reset_index(drop=True)
    
        # Generate reasons for recommendation
        recommendations_with_reasons = []
        for i, row in recommended_movies.iterrows():
            reason = f"Recommended because it has a similar sentiment score ({row['overview_sentiment_score']:.2f} vs {input_sentiment_score:.2f})"
    
            # Add reasons based on content similarity (genres, keywords)
            rec_genres = row['genres']
            rec_keywords = row['keywords']
    
            shared_genres = set(input_genres.split()) & set(rec_genres.split())
            shared_keywords = set(input_keywords.split()) & set(rec_keywords.split())
    
            if shared_genres:
                reason += f" and shares genres like {', '.join(list(shared_genres)[:3])}" # Show up to 3 shared genres
            if shared_keywords:
                 reason += f" and keywords such as {', '.join(list(shared_keywords)[:3])}" # Show up to 3 shared keywords
    
    
            recommendations_with_reasons.append({
                'Recommended Movie': row['title'],
                'Reason': reason,
                'Sentiment Score': row['overview_sentiment_score'],
                'Combined Score': row['combined_score']
            })
    
        return pd.DataFrame(recommendations_with_reasons)
    
    # @title Generate recommendations with reasons for a specific movie title
    
    movie_title_for_reasons = 'Avatar'  #@param {type:"string"}
    sentiment_weight_reasons = 0.5  #@param {type:"slider", min:0.0, max:1.0, step:0.1}
    content_weight_reasons = 0.5  #@param {type:"slider", min:0.0, max:1.0, step:0.1}
    num_recommendations_reasons = 5  #@param {type:"slider", min:1, max:20, step:1}
    
    
    recommendations_table = generate_recommendations_with_reasons(
        movie_title_for_reasons,
        merged_df,
        cosine_sim,
        num_recommendations=num_recommendations_reasons,
        sentiment_weight=sentiment_weight_reasons,
        content_weight=content_weight_reasons
    )
    
    print(f"\nRecommendations and Reasons for '{movie_title_for_reasons}':")
    from IPython.display import display
    display(recommendations_table)
    
    # @title Recommended Movie vs Reason
    
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    plt.subplots(figsize=(8, 8))
    df_2dhist = pd.DataFrame({
        x_label: grp['Reason'].value_counts()
        for x_label, grp in recommendations_table.groupby('Recommended Movie')
    })
    sns.heatmap(df_2dhist, cmap='viridis')
    plt.xlabel('Recommended Movie')
    _ = plt.ylabel('Reason')
    
    # generate movie recommendation with keywords for movie title or plot overview  and also print the confidence score as well
    
    def generate_recommendations_by_keyword(keyword, df, cosine_sim=cosine_sim, num_recommendations=10):
        # Use the TF-IDF vectorizer to transform the keyword into a vector
        # We need to fit the vectorizer first if it hasn't been fitted on the entire corpus
    
        # Transform the input keyword/query
        keyword_vec = tfidf.transform([keyword])
    
        # Calculate cosine similarity between the keyword vector and all movie soup vectors
        keyword_sim_scores = linear_kernel(keyword_vec, tfidf_matrix).flatten()
    
        # Get the pairwise similarity scores as a list of (index, score) tuples
        sim_scores = list(enumerate(keyword_sim_scores))
    
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
        # Get the scores of the num_recommendations most similar movies
        # We take from the beginning since the input is not a movie itself
        sim_scores = sim_scores[:num_recommendations]
    
        # Get the movie indices and their confidence scores (similarity score)
        movie_indices = [(i[0], i[1]) for i in sim_scores]
    
        # Create a list of recommended movies and their confidence scores
        recommendations_list = []
        for idx, confidence in movie_indices:
            recommendations_list.append({
                'title': df['title'].iloc[idx],
                'overview': df['overview'].iloc[idx],
                'genres': df['genres'].iloc[idx],
                'keywords': df['keywords'].iloc[idx],
                'confidence_score': confidence # Confidence score is the cosine similarity
            })
    
        return pd.DataFrame(recommendations_list)
    
    # @title Generate recommendations by keyword and print confidence score
    search_keyword = 'chocolate'  #@param {type:"string"}
    num_recommendations_keyword = 10  #@param {type:"slider", min:1, max:20, step:1}
    
    keyword_recommendations = generate_recommendations_by_keyword(
        search_keyword,
        merged_df,
        cosine_sim=cosine_sim,
        num_recommendations=num_recommendations_keyword
    )
    
    print(f"\nRecommendations based on the keyword '{search_keyword}':")
    keyword_recommendations
    
    
    # Movie recommendations for any of the below condition matches and display the reason with confidence score
    # 1. movie title or partial movie name
    # 2. movie keyword
    # 3. plot overview
    # 4. actor name or partial actor name
    # 5. release year
    # 6. country
    # 7. language
    
    import pandas as pd
    import numpy as np
    def generate_recommendations(query, df, cosine_sim, num_recommendations=5, sentiment_weight=0.5, content_weight=0.5):
        """
        Generates movie recommendations based on various criteria (title, keyword, plot, actor, year, country, language).
    
        Args:
            query (str or int): The input query (movie title, keyword, year, etc.).
            df (pd.DataFrame): The DataFrame containing movie information (merged_df).
            cosine_sim (np.array): The cosine similarity matrix based on the 'soup'.
            num_recommendations (int, optional): The number of recommendations to generate. Defaults to 5.
            sentiment_weight (float, optional): Weight for sentiment similarity (used for title/overview match). Defaults to 0.5.
            content_weight (float, optional): Weight for content similarity (used for title/overview match). Defaults to 0.5.
    
        Returns:
            pd.DataFrame: A DataFrame containing the recommended movies, reason, and confidence score.
                          Returns an empty DataFrame if no matches are found.
        """
        results = []
    
        # --- 1. Match by Movie Title (Partial or Full) ---
        # Find movies where the title contains the query (case-insensitive)
        title_matches = df[df['title'].str.contains(str(query), case=False, na=False)]
        if not title_matches.empty:
            # If an exact match is found, use content-based/sentiment recommendations
            exact_match = title_matches[title_matches['title'].str.lower() == str(query).lower()]
            if not exact_match.empty:
                movie_title = exact_match['title'].iloc[0]
                idx = indices[movie_title]
                input_sentiment_score = df.loc[idx, 'overview_sentiment_score']
                input_genres = df.loc[idx, 'genres']
                input_keywords = df.loc[idx, 'keywords']
    
                # Calculate combined scores as done in get_combined_recommendations
                df_temp = df.copy()
                df_temp['sentiment_difference'] = abs(df_temp['overview_sentiment_score'] - input_sentiment_score)
                df_temp['sentiment_rank'] = df_temp['sentiment_difference'].rank(method='min', ascending=True)
                df_temp['normalized_sentiment_sim'] = 1 / df_temp['sentiment_rank']
                df_temp['normalized_sentiment_sim'] = df_temp['normalized_sentiment_sim'] / df_temp['normalized_sentiment_sim'].max()
    
                sim_scores_list = list(enumerate(cosine_sim[idx]))
                content_sim_series = pd.Series([score for index, score in sim_scores_list])
                df_temp['content_sim'] = content_sim_series
                df_temp['normalized_content_sim'] = df_temp['normalized_content_sim'] / df_temp['normalized_content_sim'].max()
    
                df_temp['combined_score'] = (df_temp['normalized_sentiment_sim'] * sentiment_weight) + (df_temp['normalized_content_sim'] * content_weight)
    
                recommended_movies = df_temp.sort_values(by='combined_score', ascending=False).head(num_recommendations + 1)
                recommended_movies = recommended_movies[recommended_movies['title'] != movie_title].reset_index(drop=True)
    
                for i, row in recommended_movies.iterrows():
                     reason = f"Recommended because it is similar to '{movie_title}' based on its content (genres, keywords, cast, director)"
                     # Add sentiment similarity to the reason if sentiment weight is significant
                     if sentiment_weight > 0.1:
                          reason += f" and similar overview sentiment ({row['overview_sentiment_score']:.2f} vs {input_sentiment_score:.2f})"
    
                     results.append({
                         'Recommended Movie': row['title'],
                         'Reason': reason,
                         'Confidence Score': row['combined_score'] # Use combined score as confidence
                     })
                return pd.DataFrame(results) # Return recommendations for exact title match
    
    
            # If only partial matches are found, list them as potential recommendations
            for i, row in title_matches.iterrows():
                results.append({
                    'Recommended Movie': row['title'],
                    'Reason': f"Title contains '{query}'",
                    'Confidence Score': 1.0 # Assign high confidence for direct title match
                })
            # If we find title matches, maybe stop here or prioritize these?
            # Let's add them and continue checking other conditions
            # To avoid overwhelming, let's limit title match results if many are found
            results = results[:num_recommendations * 2] # Show a bit more than the requested recs
    
    
        # --- 2. Match by Keyword or Plot Overview (using TF-IDF and Cosine Similarity) ---
        # This covers both keyword and plot overview conditions
        # We'll use the existing generate_recommendations_by_keyword function, which works on the 'soup'
        # The confidence score from this function is the cosine similarity.
    
        # Only perform keyword/plot search if no strong title match recommendations were generated
        if not results:
            keyword_recommendations_df = generate_recommendations_by_keyword(
                str(query),
                df,
                cosine_sim=cosine_sim,
                num_recommendations=num_recommendations
            )
            for i, row in keyword_recommendations_df.iterrows():
                 # Determine the reason based on which part of the soup contributed most (complex to do precisely)
                 # For simplicity, state it's based on overall content similarity
                 reason = f"Recommended based on content similarity (keywords, plot, genres, cast, director)"
                 results.append({
                     'Recommended Movie': row['title'],
                     'Reason': reason,
                     'Confidence Score': row['confidence_score'] # Cosine similarity
                 })
    
        # --- 4. Match by Actor Name (Partial or Full) ---
        # Check if the actor name (partial or full) is in the 'cast' string
        actor_matches = df[df['cast'].str.contains(str(query), case=False, na=False)]
        if not actor_matches.empty:
            for i, row in actor_matches.iterrows():
                # Check if this movie is already in results to avoid duplicates, or prioritize
                if row['title'] not in [r['Recommended Movie'] for r in results]:
                     results.append({
                        'Recommended Movie': row['title'],
                        'Reason': f"Features actor '{query}'",
                        'Confidence Score': 0.9 # Assign high confidence for actor match
                    })
    
        # --- 5. Match by Release Year ---
        try:
            query_year = int(query)
            year_matches = df[df['release_year'] == query_year]
            if not year_matches.empty:
                for i, row in year_matches.iterrows():
                     if row['title'] not in [r['Recommended Movie'] for r in results]:
                        results.append({
                            'Recommended Movie': row['title'],
                            'Reason': f"Released in the year {query_year}",
                            'Confidence Score': 0.8 # Assign good confidence for year match
                        })
        except ValueError:
            pass # Query is not a valid year, ignore this condition
    
        pass # Currently cannot match by language
    
    
        # Sort results by Confidence Score in descending order
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values(by='Confidence Score', ascending=False).drop_duplicates(subset=['Recommended Movie']).head(num_recommendations).reset_index(drop=True)
        else:
            print(f"No recommendations found for query '{query}'.")
    
    
        return results_df
    
    # @title Generate Recommendations based on various criteria
    
    search_query = 'sci-fi'  #@param {type:"string"}
    num_recommendations_general = 10  #@param {type:"slider", min:1, max:20, step:1}
    
    general_recommendations = generate_recommendations(
        search_query,
        merged_df,
        cosine_sim,
        num_recommendations=num_recommendations_general
    )
    
    print(f"\nRecommendations for query '{search_query}':")
    display(general_recommendations)
    
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

