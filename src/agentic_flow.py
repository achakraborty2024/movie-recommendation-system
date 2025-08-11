def run_agentic_flow():
    """
    Ensure you have the necessary libraries installed
    !pip install pandas scikit-learn joblib sentence-transformers keras tensorflow transformers
    !pip install langchain openai langchain_community
    """
    
    import os
    import pandas as pd
    import json
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np # Import numpy
    
    
    os.environ['OPENAI_API_KEY'] = "<YOUR_OPENAI_API_KEY>" # Set the OpenAI API key here
    
    def extract_all_genres(movies):
        all_genres = []
        for g in movies['genres']:
            # Safely attempt to load JSON and extract names
            if isinstance(g, str) and g: # Check if it's a non-empty string
                try:
                    all_genres += [d['name'] for d in json.loads(g)]
                except json.JSONDecodeError:
                    # Handle cases with invalid JSON or non-JSON strings
                    continue # Skip this entry if it's not valid JSON
        return sorted(set(all_genres))
    
    def llm_genre_extractor(user_input, all_genres, llm):
        genres_string = ', '.join(all_genres)
        # Adding a constraint to only list genres from the provided list
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a movie recommendation assistant. Extract up to 3 movie genres from the user's input. Only list genres that are present in the following comma-separated list, exactly as they appear in the list: " + genres_string + ". Respond with a comma-separated list of the extracted genres, or 'None' if no matching genres are found."),
            ("human", "{user_input}")
        ])
        chain = prompt | llm
        response = chain.invoke({"user_input": user_input})
        # Filter to ensure only valid genres from the list are returned
        extracted_genres = [g.strip() for g in response.content.split(",") if g.strip() in all_genres]
        return extracted_genres if extracted_genres else [] # Return empty list if no valid genres found
    
    def content_score_agent(movies, fav_genres):
        def genre_score(genres_json, fav_genres):
            if not fav_genres: return 0 # Return 0 if no favorite genres
            if isinstance(genres_json, str) and genres_json: # Check if it's a non-empty string
                try:
                    genres = [g['name'] for g in json.loads(genres_json)]
                    # Calculate score based on intersection
                    return len(set(genres).intersection(fav_genres)) / len(fav_genres) # Score based on proportion of fav_genres found
                except json.JSONDecodeError:
                     return 0 # Return 0 for invalid JSON
            return 0 # Return 0 for non-string or empty values
    
        movies = movies.copy()
        movies['content_score'] = movies['genres'].apply(lambda x: genre_score(x, fav_genres))
        return movies
    
    def popularity_agent(movies):
        # Handle potential non-numeric or missing vote_count values before scaling
        movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce').fillna(0)
        scaler = MinMaxScaler()
        movies = movies.copy()
        # Reshape the data for the scaler
        movies['collab_score'] = scaler.fit_transform(movies[['vote_count']])
        return movies
    
    def blending_agent(movies, w_content=0.6, w_collab=0.4):
        movies = movies.copy()
        # Ensure content_score and collab_score exist and handle potential NaNs
        movies['content_score'] = movies['content_score'].fillna(0)
        movies['collab_score'] = movies['collab_score'].fillna(0)
        movies['hybrid_score'] = w_content * movies['content_score'] + w_collab * movies['collab_score']
        return movies
    
    def recommendation_agent(movies, top_n=5):
        # Ensure hybrid_score exists before sorting
        if 'hybrid_score' not in movies.columns:
            print("Error: 'hybrid_score' column not found. Cannot generate recommendations.")
            return pd.DataFrame() # Return empty DataFrame
    
        # Sort by hybrid_score and return top_n
        return movies.sort_values('hybrid_score', ascending=False).head(top_n)
    
    def llm_reason_generator(movie, fav_genres, llm):
        genre_list = ', '.join(fav_genres)
        # Safely extract movie genres, handling potential errors
        movie_genres_list = []
        if isinstance(movie['genres'], str) and movie['genres']:
            try:
                movie_genres_list = [g['name'] for g in json.loads(movie['genres'])]
            except json.JSONDecodeError:
                pass # If decoding fails, movie_genres_list remains empty
    
        movie_genres_str = ', '.join(movie_genres_list)
    
        # Adjust prompt to handle cases where no movie genres were extracted
        prompt_template = """You are an assistant explaining movie recommendations to a user. Give a brief (1-2 sentence) reason why the following movie is recommended. Focus on how its genres or popularity match the user's preferences.
    User's favorite genres: {genre_list}.
    Movie: {movie_title}.
    Movie Genres: {movie_genres}.
    Hybrid score: {hybrid_score:.2f}.
    """
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", f"Explain the recommendation for the movie '{movie_title}'.") # More direct human message
        ])
    
        chain = prompt | llm
        # Prepare input variables for the prompt
        input_variables = {
            "genre_list": genre_list,
            "movie_title": movie['title'],
            "movie_genres": movie_genres_str if movie_genres_str else "No genre information available.", # Provide a default if no genres extracted
            "hybrid_score": movie['hybrid_score']
        }
        return chain.invoke(input_variables).content
    
    def explain_agent(top_movies, fav_genres, llm):
        reasons = []
        if top_movies.empty:
            print("No top movies to explain.")
            return top_movies # Return empty if no movies were recommended
    
        for index, movie in top_movies.iterrows(): # Use iterrows to iterate over rows
            try:
                reason = llm_reason_generator(movie, fav_genres, llm)
                reasons.append(reason)
            except Exception as e:
                print(f"Error generating reason for movie '{movie.get('title', 'Unknown')}': {e}")
                reasons.append("Could not generate explanation.") # Add a fallback reason
        top_movies = top_movies.copy()
        top_movies['reason'] = reasons
        # Select and rename columns for the final output
        output_cols = ['title', 'content_score', 'collab_score', 'hybrid_score', 'reason']
        # Filter output_cols to only include columns that actually exist in top_movies
        existing_output_cols = [col for col in output_cols if col in top_movies.columns]
        return top_movies[existing_output_cols].rename(columns={'title': 'Recommended Movie'})
    
    
    def agentic_movie_recommender(movies_df_input, user_input):
        # Ensure movies_df_input is a DataFrame and not empty
        if not isinstance(movies_df_input, pd.DataFrame) or movies_df_input.empty:
            print("Error: Input movie DataFrame is invalid or empty.")
            return pd.DataFrame()
    
        # Ensure essential columns exist in the input DataFrame
        required_initial_cols = ['genres', 'vote_count', 'title']
        if not all(col in movies_df_input.columns for col in required_initial_cols):
             print(f"Error: Input movie DataFrame is missing required columns: {required_initial_cols}")
             return pd.DataFrame()
    
        all_genres = extract_all_genres(movies_df_input)
        # Initialize LLM - ensure OPENAI_API_KEY is set in environment
        llm = ChatOpenAI(temperature=0, model="gpt-4o") # Added model parameter
    
        fav_genres = llm_genre_extractor(user_input, all_genres, llm)
        print(f"Extracted favorite genres: {fav_genres}")
    
        # Pass the input DataFrame through the agents
        movies_scored_content = content_score_agent(movies_df_input, fav_genres)
        movies_scored_popularity = popularity_agent(movies_scored_content) # Chain the outputs
        movies_blended = blending_agent(movies_scored_popularity, w_content=0.6, w_collab=0.4)
    
        top_movies = recommendation_agent(movies_blended, top_n=5)
    
        # Only proceed to explain if recommendations were found
        if not top_movies.empty:
            explained = explain_agent(top_movies, fav_genres, llm)
            return explained
        else:
            print("No recommendations found.")
            return pd.DataFrame() # Return empty DataFrame if no recommendations
    
    user_input = "I love sci-fi and adventure but don't like horror or romance."
    result = agentic_movie_recommender(merged_df, user_input)
    print(result)
    
    user_input = "I love horror and adventure but don't like sci-fi or romance."
    result = agentic_movie_recommender(merged_df, user_input)
    print(result)

