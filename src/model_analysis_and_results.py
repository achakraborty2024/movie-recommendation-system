def run_model_analysis_and_results():
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

