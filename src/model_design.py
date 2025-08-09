"""
Model Design and Building Module for Movie Recommendation System
==============================================================

This module contains classes and functions to design and build various 
recommendation models including content-based filtering, collaborative filtering,
hybrid models, and deep learning approaches.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
import pickle
import joblib
import warnings

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Autoencoder models will not be available.")

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Sentiment analysis will not be available.")


class ContentBasedRecommender:
    """
    Content-based recommendation system using TF-IDF and cosine similarity.
    """
    
    def __init__(self, stop_words='english', ngram_range=(1, 2), max_features=10000):
        """
        Initialize the content-based recommender.
        
        Args:
            stop_words (str): Stop words to use in TF-IDF
            ngram_range (tuple): N-gram range for TF-IDF
            max_features (int): Maximum number of features for TF-IDF
        """
        self.tfidf = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_features=max_features
        )
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.data = None
    
    def fit(self, data, content_column='soup'):
        """
        Fit the content-based model.
        
        Args:
            data (pd.DataFrame): Movie data
            content_column (str): Column containing text content
        """
        self.data = data.copy()
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.tfidf.fit_transform(data[content_column].fillna(''))
        
        # Calculate cosine similarity
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create indices mapping
        self.indices = pd.Series(data.index, index=data['title']).drop_duplicates()
        
        print(f"Content-based model fitted. TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def get_recommendations(self, title, num_recommendations=10):
        """
        Get recommendations based on movie title.
        
        Args:
            title (str): Movie title
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if title not in self.indices:
            return pd.DataFrame()
        
        idx = self.indices[title]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        
        movie_indices = [i[0] for i in sim_scores]
        
        # Return recommended movies with scores
        recommendations = self.data.iloc[movie_indices].copy()
        recommendations['similarity_score'] = [score[1] for score in sim_scores]
        
        return recommendations[['title', 'genres', 'overview', 'similarity_score']]
    
    def get_recommendations_by_query(self, query, num_recommendations=10):
        """
        Get recommendations based on text query.
        
        Args:
            query (str): Text query
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        # Transform query
        query_vec = self.tfidf.transform([query])
        
        # Calculate similarities
        similarities = linear_kernel(query_vec, self.tfidf_matrix).flatten()
        
        # Get top recommendations
        top_indices = similarities.argsort()[-num_recommendations:][::-1]
        
        recommendations = self.data.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations[['title', 'genres', 'overview', 'similarity_score']]


class KNNRecommender:
    """
    K-Nearest Neighbors recommendation system.
    """
    
    def __init__(self, n_neighbors=10, metric='cosine'):
        """
        Initialize KNN recommender.
        
        Args:
            n_neighbors (int): Number of neighbors
            metric (str): Distance metric to use
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)
        self.feature_matrix = None
        self.data = None
        self.indices = None
    
    def fit(self, data, content_column='soup'):
        """
        Fit the KNN model.
        
        Args:
            data (pd.DataFrame): Movie data
            content_column (str): Column containing text content
        """
        self.data = data.copy()
        
        # Create feature matrix
        self.feature_matrix = self.tfidf.fit_transform(data[content_column].fillna(''))
        
        # Fit KNN model
        self.model.fit(self.feature_matrix)
        
        # Create indices mapping
        self.indices = pd.Series(data.index, index=data['title']).drop_duplicates()
        
        print(f"KNN model fitted. Feature matrix shape: {self.feature_matrix.shape}")
    
    def get_recommendations(self, title, num_recommendations=10):
        """
        Get recommendations based on movie title using KNN.
        
        Args:
            title (str): Movie title
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if title not in self.indices:
            return pd.DataFrame()
        
        idx = self.indices[title]
        
        # Find neighbors
        distances, indices = self.model.kneighbors(
            self.feature_matrix[idx],
            n_neighbors=num_recommendations + 1
        )
        
        # Skip first result (same movie)
        neighbor_indices = indices[0][1:]
        neighbor_distances = distances[0][1:]
        
        # Convert distances to similarities
        similarities = 1 - neighbor_distances
        
        recommendations = self.data.iloc[neighbor_indices].copy()
        recommendations['similarity_score'] = similarities
        
        return recommendations[['title', 'genres', 'overview', 'similarity_score']]
    
    def get_recommendations_by_query(self, query, num_recommendations=10):
        """
        Get recommendations based on text query using KNN.
        
        Args:
            query (str): Text query
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        # Transform query
        query_vec = self.tfidf.transform([query])
        
        # Find neighbors
        distances, indices = self.model.kneighbors(query_vec, n_neighbors=num_recommendations)
        
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]
        similarities = 1 - neighbor_distances
        
        recommendations = self.data.iloc[neighbor_indices].copy()
        recommendations['similarity_score'] = similarities
        
        return recommendations[['title', 'genres', 'overview', 'similarity_score']]


class AutoencoderRecommender:
    """
    Deep learning-based recommendation system using autoencoders.
    """
    
    def __init__(self, encoding_dim=128, epochs=50, batch_size=256):
        """
        Initialize autoencoder recommender.
        
        Args:
            encoding_dim (int): Dimension of encoding layer
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for AutoencoderRecommender")
        
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)
        self.scaler = MinMaxScaler()
        self.autoencoder = None
        self.encoder = None
        self.feature_matrix = None
        self.latent_features = None
        self.data = None
        self.indices = None
    
    def _build_autoencoder(self, input_dim):
        """
        Build the autoencoder model.
        
        Args:
            input_dim (int): Input dimension
        """
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(self.encoding_dim, activation='relu', kernel_regularizer=l2(1e-5))(input_layer)
        
        # Decoder
        decoded = Dense(input_dim, activation='sigmoid', kernel_regularizer=l2(1e-5))(encoded)
        
        # Models
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        # Compile
        self.autoencoder.compile(optimizer='adam', loss='mse')
    
    def fit(self, data, content_column='soup'):
        """
        Fit the autoencoder model.
        
        Args:
            data (pd.DataFrame): Movie data
            content_column (str): Column containing text content
        """
        self.data = data.copy()
        
        # Create TF-IDF features
        tfidf_matrix = self.tfidf.fit_transform(data[content_column].fillna(''))
        
        # Scale features
        self.feature_matrix = self.scaler.fit_transform(tfidf_matrix.toarray())
        
        # Build autoencoder
        self._build_autoencoder(self.feature_matrix.shape[1])
        
        # Train autoencoder
        print("Training autoencoder...")
        history = self.autoencoder.fit(
            self.feature_matrix, self.feature_matrix,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        
        # Generate latent features
        self.latent_features = self.encoder.predict(self.feature_matrix, verbose=0)
        
        # Create indices mapping
        self.indices = pd.Series(data.index, index=data['title']).drop_duplicates()
        
        print(f"Autoencoder model fitted. Latent features shape: {self.latent_features.shape}")
        
        return history
    
    def get_recommendations(self, title, num_recommendations=10):
        """
        Get recommendations based on latent features.
        
        Args:
            title (str): Movie title
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if title not in self.indices:
            return pd.DataFrame()
        
        idx = self.indices[title]
        
        # Calculate similarities in latent space
        similarities = cosine_similarity([self.latent_features[idx]], self.latent_features)[0]
        
        # Get top recommendations
        top_indices = similarities.argsort()[-num_recommendations-1:-1][::-1]
        
        recommendations = self.data.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations[['title', 'genres', 'overview', 'similarity_score']]


class ClusteringRecommender:
    """
    Clustering-based recommendation system.
    """
    
    def __init__(self, n_clusters=20, random_state=42):
        """
        Initialize clustering recommender.
        
        Args:
            n_clusters (int): Number of clusters
            random_state (int): Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)
        self.feature_matrix = None
        self.data = None
        self.indices = None
    
    def fit(self, data, content_column='soup'):
        """
        Fit the clustering model.
        
        Args:
            data (pd.DataFrame): Movie data
            content_column (str): Column containing text content
        """
        self.data = data.copy()
        
        # Create features
        self.feature_matrix = self.tfidf.fit_transform(data[content_column].fillna(''))
        
        # Fit clustering model
        cluster_labels = self.kmeans.fit_predict(self.feature_matrix.toarray())
        self.data['cluster'] = cluster_labels
        
        # Create indices mapping
        self.indices = pd.Series(data.index, index=data['title']).drop_duplicates()
        
        print(f"Clustering model fitted. {self.n_clusters} clusters created.")
        print(f"Cluster distribution: {pd.Series(cluster_labels).value_counts().sort_index().to_dict()}")
    
    def get_recommendations(self, title, num_recommendations=10):
        """
        Get recommendations from same cluster.
        
        Args:
            title (str): Movie title
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if title not in self.indices:
            return pd.DataFrame()
        
        idx = self.indices[title]
        cluster_id = self.data.loc[idx, 'cluster']
        
        # Get movies in same cluster
        cluster_movies = self.data[self.data['cluster'] == cluster_id]
        cluster_movies = cluster_movies[cluster_movies.index != idx]
        
        # Sample recommendations if too many
        if len(cluster_movies) > num_recommendations:
            recommendations = cluster_movies.sample(n=num_recommendations, random_state=self.random_state)
        else:
            recommendations = cluster_movies
        
        recommendations = recommendations.copy()
        recommendations['cluster_id'] = cluster_id
        
        return recommendations[['title', 'genres', 'overview', 'cluster_id']]


class SentimentBasedRecommender:
    """
    Sentiment-based recommendation system.
    """
    
    def __init__(self):
        """
        Initialize sentiment-based recommender.
        """
        if not NLTK_AVAILABLE:
            print("NLTK not available. Using simple sentiment scores if available in data.")
            self.analyzer = None
        else:
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')
            self.analyzer = SentimentIntensityAnalyzer()
        
        self.data = None
    
    def fit(self, data, sentiment_column=None, text_column='overview'):
        """
        Fit the sentiment-based model.
        
        Args:
            data (pd.DataFrame): Movie data
            sentiment_column (str): Pre-computed sentiment column
            text_column (str): Text column to analyze sentiment
        """
        self.data = data.copy()
        
        # Use existing sentiment scores or compute new ones
        if sentiment_column and sentiment_column in data.columns:
            self.data['sentiment_score'] = data[sentiment_column]
        elif self.analyzer is not None:
            print("Computing sentiment scores...")
            self.data['sentiment_score'] = self.data[text_column].fillna('').apply(
                lambda x: self.analyzer.polarity_scores(str(x))['compound']
            )
        else:
            raise ValueError("No sentiment data available and NLTK not installed")
        
        print("Sentiment-based model fitted.")
    
    def get_recommendations(self, title, num_recommendations=10, sentiment_tolerance=0.1):
        """
        Get recommendations based on sentiment similarity.
        
        Args:
            title (str): Movie title
            num_recommendations (int): Number of recommendations to return
            sentiment_tolerance (float): Tolerance for sentiment matching
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if title not in self.data['title'].values:
            return pd.DataFrame()
        
        # Get sentiment of input movie
        input_sentiment = self.data[self.data['title'] == title]['sentiment_score'].iloc[0]
        
        # Find movies with similar sentiment
        sentiment_diff = abs(self.data['sentiment_score'] - input_sentiment)
        similar_movies = self.data[sentiment_diff <= sentiment_tolerance]
        similar_movies = similar_movies[similar_movies['title'] != title]
        
        # Sort by similarity and return top recommendations
        similar_movies = similar_movies.copy()
        similar_movies['sentiment_difference'] = sentiment_diff[similar_movies.index]
        similar_movies = similar_movies.sort_values('sentiment_difference')
        
        recommendations = similar_movies.head(num_recommendations)
        
        return recommendations[['title', 'genres', 'overview', 'sentiment_score', 'sentiment_difference']]


class HybridRecommender:
    """
    Hybrid recommendation system combining multiple approaches.
    """
    
    def __init__(self, models, weights=None):
        """
        Initialize hybrid recommender.
        
        Args:
            models (dict): Dictionary of model names and instances
            weights (dict): Weights for each model
        """
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models.keys()}
        self.data = None
    
    def fit(self, data, content_column='soup'):
        """
        Fit all component models.
        
        Args:
            data (pd.DataFrame): Movie data
            content_column (str): Column containing text content
        """
        self.data = data.copy()
        
        print("Fitting hybrid model components...")
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            if hasattr(model, 'fit'):
                model.fit(data, content_column)
        
        print("Hybrid model fitted.")
    
    def get_recommendations(self, title, num_recommendations=10):
        """
        Get hybrid recommendations by combining multiple models.
        
        Args:
            title (str): Movie title
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies with combined scores
        """
        all_recommendations = {}
        
        # Get recommendations from each model
        for name, model in self.models.items():
            try:
                recs = model.get_recommendations(title, num_recommendations * 2)  # Get more to combine
                if not recs.empty:
                    score_col = 'similarity_score' if 'similarity_score' in recs.columns else 'sentiment_difference'
                    for idx, row in recs.iterrows():
                        movie_title = row['title']
                        if movie_title not in all_recommendations:
                            all_recommendations[movie_title] = {'scores': {}, 'data': row}
                        
                        # Invert sentiment difference for consistent scoring
                        score = row[score_col]
                        if score_col == 'sentiment_difference':
                            score = 1 / (1 + score)  # Convert difference to similarity
                        
                        all_recommendations[movie_title]['scores'][name] = score
            except Exception as e:
                print(f"Error getting recommendations from {name}: {e}")
        
        # Calculate combined scores
        combined_recommendations = []
        for movie_title, info in all_recommendations.items():
            combined_score = 0
            total_weight = 0
            
            for model_name, weight in self.weights.items():
                if model_name in info['scores']:
                    combined_score += info['scores'][model_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                combined_score /= total_weight
                
                rec_data = info['data'].copy()
                rec_data['combined_score'] = combined_score
                combined_recommendations.append(rec_data)
        
        # Convert to DataFrame and sort
        if combined_recommendations:
            result_df = pd.DataFrame(combined_recommendations)
            result_df = result_df.sort_values('combined_score', ascending=False)
            result_df = result_df.head(num_recommendations)
            
            return result_df[['title', 'genres', 'overview', 'combined_score']]
        else:
            return pd.DataFrame()


def create_recommendation_pipeline(model_type='content', **kwargs):
    """
    Factory function to create recommendation models.
    
    Args:
        model_type (str): Type of model to create
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Recommendation model instance
    """
    if model_type == 'content':
        return ContentBasedRecommender(**kwargs)
    elif model_type == 'knn':
        return KNNRecommender(**kwargs)
    elif model_type == 'autoencoder':
        return AutoencoderRecommender(**kwargs)
    elif model_type == 'clustering':
        return ClusteringRecommender(**kwargs)
    elif model_type == 'sentiment':
        return SentimentBasedRecommender(**kwargs)
    elif model_type == 'hybrid':
        # Create default hybrid model
        models = {
            'content': ContentBasedRecommender(),
            'knn': KNNRecommender(),
            'clustering': ClusteringRecommender()
        }
        return HybridRecommender(models, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_model(model, filepath):
    """
    Save a recommendation model to file.
    
    Args:
        model: Recommendation model instance
        filepath (str): Path to save the model
    """
    if hasattr(model, 'autoencoder'):
        # Save autoencoder components separately
        model_data = {
            'type': type(model).__name__,
            'params': {k: v for k, v in model.__dict__.items() if k not in ['autoencoder', 'encoder']}
        }
        joblib.dump(model_data, filepath + '_params.joblib')
        if model.autoencoder:
            model.autoencoder.save(filepath + '_autoencoder.keras')
        if model.encoder:
            model.encoder.save(filepath + '_encoder.keras')
    else:
        joblib.dump(model, filepath)
    
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a recommendation model from file.
    
    Args:
        filepath (str): Path to load the model from
        
    Returns:
        Recommendation model instance
    """
    try:
        # Try loading as regular joblib file
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    except:
        # Try loading autoencoder model
        try:
            model_data = joblib.load(filepath + '_params.joblib')
            model_class = globals()[model_data['type']]
            model = model_class()
            
            # Restore parameters
            for key, value in model_data['params'].items():
                setattr(model, key, value)
            
            # Load neural network components
            if TENSORFLOW_AVAILABLE:
                try:
                    model.autoencoder = tf.keras.models.load_model(filepath + '_autoencoder.keras')
                    model.encoder = tf.keras.models.load_model(filepath + '_encoder.keras')
                except:
                    pass
            
            print(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            raise ValueError(f"Could not load model from {filepath}: {e}")


def main():
    """
    Example usage of the recommendation models.
    """
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'title': [f'Movie {i}' for i in range(100)],
        'genres': ['Action Adventure'] * 25 + ['Drama'] * 25 + ['Comedy'] * 25 + ['Horror'] * 25,
        'overview': [f'This is a great movie about {i}' for i in range(100)],
        'soup': [f'movie {i} action adventure great story' for i in range(100)]
    })
    
    print("Creating and testing recommendation models...")
    
    # Test content-based recommender
    print("\n1. Content-based Recommender:")
    content_model = ContentBasedRecommender()
    content_model.fit(sample_data)
    recs = content_model.get_recommendations('Movie 0', 5)
    print(recs)
    
    # Test KNN recommender
    print("\n2. KNN Recommender:")
    knn_model = KNNRecommender(n_neighbors=5)
    knn_model.fit(sample_data)
    recs = knn_model.get_recommendations('Movie 0', 5)
    print(recs)
    
    # Test clustering recommender
    print("\n3. Clustering Recommender:")
    cluster_model = ClusteringRecommender(n_clusters=5)
    cluster_model.fit(sample_data)
    recs = cluster_model.get_recommendations('Movie 0', 5)
    print(recs)
    
    # Test hybrid recommender
    print("\n4. Hybrid Recommender:")
    models = {
        'content': ContentBasedRecommender(),
        'knn': KNNRecommender(n_neighbors=5),
        'clustering': ClusteringRecommender(n_clusters=5)
    }
    hybrid_model = HybridRecommender(models)
    hybrid_model.fit(sample_data)
    recs = hybrid_model.get_recommendations('Movie 0', 5)
    print(recs)
    
    print("\nModel testing completed!")


if __name__ == "__main__":
    main()
