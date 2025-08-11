def run_model_optimization():
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
    
    # Experiment with TF-IDF Vectorizer hyperparameters
    
    print("## Experimenting with TF-IDF Vectorizer Hyperparameters\n")
    
    # Original TF-IDF (already fitted)
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=86621)
    tfidf_matrix = tfidf.fit_transform(merged_df['soup'].fillna(''))
    
    # Experiment with different TF-IDF hyperparameters
    tfidf_experiments = {
        'Original': {'max_features': None, 'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0},
        'Max_Features_5000': {'max_features': 5000, 'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0},
        'Ngram_Range_1_2': {'max_features': None, 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 1.0},
        'Min_DF_5': {'max_features': None, 'ngram_range': (1, 1), 'min_df': 5, 'max_df': 1.0},
        'Max_DF_0_9': {'max_features': None, 'ngram_range': (1, 1), 'min_df': 1, 'max_df': 0.9}
    }
    
    sample_movie_for_tfidf = "Pirates of the Caribbean: At World\'s End"
    num_recommendations_tfidf_exp = 5
    
    original_tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=86621)
    original_tfidf_matrix = original_tfidf.fit_transform(merged_df['soup'].fillna(''))
    original_cosine_sim = linear_kernel(original_tfidf_matrix, original_tfidf_matrix)
    
    # Use the existing get_content_based_recommendations function
    
    print(f"Generating recommendations for '{sample_movie_for_tfidf}' using different TF-IDF settings:")
    optimized_tfidf=None # save to use later for model deployment
    for name, params in tfidf_experiments.items():
        print(f"\n--- TF-IDF Experiment: {name} ---")
        try:
            # Create and fit a new TF-IDF vectorizer with the experimental parameters
            optimized_tfidf = TfidfVectorizer(stop_words='english', **params)
            current_tfidf_matrix = optimized_tfidf.fit_transform(merged_df['soup'].fillna(''))
            current_cosine_sim = linear_kernel(current_tfidf_matrix, current_tfidf_matrix)
    
            # Generate recommendations using the new cosine similarity matrix
            current_recommendations = get_content_based_recommendations(
                sample_movie_for_tfidf,
                merged_df,
                cosine_sim=current_cosine_sim,
                num_recommendations=num_recommendations_tfidf_exp
            )
    
            print(f"Recommendations for '{sample_movie_for_tfidf}' with {name} TF-IDF:")
            display(current_recommendations)
    
            # Qualitative assessment (manual observation of displayed recommendations)
            print(f"Qualitative Assessment for {name}: [Observe the recommendations above and note changes compared to Original]")
    
        except Exception as e:
            print(f"Error running TF-IDF experiment {name}: {e}")
    
    
    # @title genres vs keywords
    
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    plt.subplots(figsize=(8, 8))
    df_2dhist = pd.DataFrame({
        x_label: grp['keywords'].value_counts()
        for x_label, grp in current_recommendations.groupby('genres')
    })
    sns.heatmap(df_2dhist, cmap='viridis')
    plt.xlabel('genres')
    _ = plt.ylabel('keywords')
    
    # @title overview_sentiment_score
    
    from matplotlib import pyplot as plt
    current_recommendations['overview_sentiment_score'].plot(kind='hist', bins=20, title='overview_sentiment_score')
    plt.gca().spines[['top', 'right',]].set_visible(False)
    
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

