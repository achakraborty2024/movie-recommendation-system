"""
Model Training Module for Movie Recommendation System
===================================================

This module handles the training of various recommendation models,
including hyperparameter tuning, cross-validation, and model evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
import joblib
import time
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class ModelTrainer:
    """
    Class to handle training of recommendation models with hyperparameter tuning.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model trainer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.trained_models = {}
        self.training_history = {}
        self.evaluation_results = {}
    
    def prepare_training_data(self, data, test_size=0.2):
        """
        Prepare data for training and testing.
        
        Args:
            data (pd.DataFrame): Complete dataset
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: Training and testing datasets
        """
        train_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=data['genres'] if 'genres' in data.columns else None
        )
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")
        
        return train_data, test_data
    
    def train_content_based_model(self, train_data, model_params=None):
        """
        Train content-based recommendation model with hyperparameter tuning.
        
        Args:
            train_data (pd.DataFrame): Training dataset
            model_params (dict): Model parameters to tune
            
        Returns:
            Best model and parameters
        """
        from .model_design import ContentBasedRecommender
        
        if model_params is None:
            model_params = {
                'max_features': [5000, 10000, 20000],
                'ngram_range': [(1, 1), (1, 2), (1, 3)],
                'stop_words': ['english']
            }
        
        best_model = None
        best_score = -1
        best_params = None
        
        print("Training Content-Based Model with hyperparameter tuning...")
        
        # Grid search over parameters
        for max_features in model_params['max_features']:
            for ngram_range in model_params['ngram_range']:
                for stop_words in model_params['stop_words']:
                    
                    params = {
                        'max_features': max_features,
                        'ngram_range': ngram_range,
                        'stop_words': stop_words
                    }
                    
                    try:
                        # Train model
                        model = ContentBasedRecommender(**params)
                        start_time = time.time()
                        model.fit(train_data)
                        training_time = time.time() - start_time
                        
                        # Simple evaluation based on TF-IDF matrix density
                        score = self._evaluate_content_model(model)
                        
                        print(f"Params: {params}, Score: {score:.4f}, Time: {training_time:.2f}s")
                        
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_params = params
                            
                    except Exception as e:
                        print(f"Error training with params {params}: {e}")
        
        self.trained_models['content_based'] = best_model
        self.training_history['content_based'] = {
            'best_params': best_params,
            'best_score': best_score
        }
        
        print(f"Best Content-Based Model - Score: {best_score:.4f}, Params: {best_params}")
        return best_model, best_params
    
    def train_knn_model(self, train_data, model_params=None):
        """
        Train KNN recommendation model with hyperparameter tuning.
        
        Args:
            train_data (pd.DataFrame): Training dataset
            model_params (dict): Model parameters to tune
            
        Returns:
            Best model and parameters
        """
        from .model_design import KNNRecommender
        
        if model_params is None:
            model_params = {
                'n_neighbors': [5, 10, 15, 20, 25],
                'metric': ['cosine', 'euclidean']
            }
        
        best_model = None
        best_score = -1
        best_params = None
        
        print("Training KNN Model with hyperparameter tuning...")
        
        for n_neighbors in model_params['n_neighbors']:
            for metric in model_params['metric']:
                
                params = {
                    'n_neighbors': n_neighbors,
                    'metric': metric
                }
                
                try:
                    model = KNNRecommender(**params)
                    start_time = time.time()
                    model.fit(train_data)
                    training_time = time.time() - start_time
                    
                    # Evaluate model
                    score = self._evaluate_knn_model(model, train_data)
                    
                    print(f"Params: {params}, Score: {score:.4f}, Time: {training_time:.2f}s")
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                        
                except Exception as e:
                    print(f"Error training with params {params}: {e}")
        
        self.trained_models['knn'] = best_model
        self.training_history['knn'] = {
            'best_params': best_params,
            'best_score': best_score
        }
        
        print(f"Best KNN Model - Score: {best_score:.4f}, Params: {best_params}")
        return best_model, best_params
    
    def train_autoencoder_model(self, train_data, model_params=None):
        """
        Train Autoencoder recommendation model with hyperparameter tuning.
        
        Args:
            train_data (pd.DataFrame): Training dataset
            model_params (dict): Model parameters to tune
            
        Returns:
            Best model and parameters
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping autoencoder training.")
            return None, None
        
        from .model_design import AutoencoderRecommender
        
        if model_params is None:
            model_params = {
                'encoding_dim': [64, 128, 256],
                'epochs': [20, 50],
                'batch_size': [128, 256]
            }
        
        best_model = None
        best_score = float('inf')  # Lower is better for autoencoder loss
        best_params = None
        
        print("Training Autoencoder Model with hyperparameter tuning...")
        
        for encoding_dim in model_params['encoding_dim']:
            for epochs in model_params['epochs']:
                for batch_size in model_params['batch_size']:
                    
                    params = {
                        'encoding_dim': encoding_dim,
                        'epochs': epochs,
                        'batch_size': batch_size
                    }
                    
                    try:
                        model = AutoencoderRecommender(**params)
                        start_time = time.time()
                        history = model.fit(train_data)
                        training_time = time.time() - start_time
                        
                        # Use final validation loss as score
                        score = min(history.history['val_loss']) if 'val_loss' in history.history else min(history.history['loss'])
                        
                        print(f"Params: {params}, Val Loss: {score:.4f}, Time: {training_time:.2f}s")
                        
                        if score < best_score:
                            best_score = score
                            best_model = model
                            best_params = params
                            
                    except Exception as e:
                        print(f"Error training with params {params}: {e}")
        
        self.trained_models['autoencoder'] = best_model
        self.training_history['autoencoder'] = {
            'best_params': best_params,
            'best_score': best_score
        }
        
        print(f"Best Autoencoder Model - Val Loss: {best_score:.4f}, Params: {best_params}")
        return best_model, best_params
    
    def train_clustering_model(self, train_data, model_params=None):
        """
        Train Clustering recommendation model with hyperparameter tuning.
        
        Args:
            train_data (pd.DataFrame): Training dataset
            model_params (dict): Model parameters to tune
            
        Returns:
            Best model and parameters
        """
        from .model_design import ClusteringRecommender
        
        if model_params is None:
            model_params = {
                'n_clusters': [10, 15, 20, 25, 30]
            }
        
        best_model = None
        best_score = -1
        best_params = None
        
        print("Training Clustering Model with hyperparameter tuning...")
        
        for n_clusters in model_params['n_clusters']:
            
            params = {
                'n_clusters': n_clusters,
                'random_state': self.random_state
            }
            
            try:
                model = ClusteringRecommender(**params)
                start_time = time.time()
                model.fit(train_data)
                training_time = time.time() - start_time
                
                # Use silhouette score for evaluation
                score = self._evaluate_clustering_model(model)
                
                print(f"Params: {params}, Silhouette Score: {score:.4f}, Time: {training_time:.2f}s")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params
                    
            except Exception as e:
                print(f"Error training with params {params}: {e}")
        
        self.trained_models['clustering'] = best_model
        self.training_history['clustering'] = {
            'best_params': best_params,
            'best_score': best_score
        }
        
        print(f"Best Clustering Model - Silhouette Score: {best_score:.4f}, Params: {best_params}")
        return best_model, best_params
    
    def train_sentiment_model(self, train_data):
        """
        Train Sentiment-based recommendation model.
        
        Args:
            train_data (pd.DataFrame): Training dataset
            
        Returns:
            Trained model
        """
        from .model_design import SentimentBasedRecommender
        
        print("Training Sentiment-Based Model...")
        
        try:
            model = SentimentBasedRecommender()
            start_time = time.time()
            
            # Check if sentiment scores already exist
            if 'overview_sentiment_score' in train_data.columns:
                model.fit(train_data, sentiment_column='overview_sentiment_score')
            else:
                model.fit(train_data, text_column='overview')
            
            training_time = time.time() - start_time
            
            self.trained_models['sentiment'] = model
            self.training_history['sentiment'] = {
                'training_time': training_time
            }
            
            print(f"Sentiment Model trained in {training_time:.2f}s")
            return model
            
        except Exception as e:
            print(f"Error training sentiment model: {e}")
            return None
    
    def train_hybrid_model(self, train_data, component_models=None, weights=None):
        """
        Train Hybrid recommendation model.
        
        Args:
            train_data (pd.DataFrame): Training dataset
            component_models (dict): Pre-trained component models
            weights (dict): Weights for combining models
            
        Returns:
            Hybrid model
        """
        from .model_design import HybridRecommender
        
        if component_models is None:
            component_models = self.trained_models.copy()
        
        if not component_models:
            print("No component models available for hybrid model.")
            return None
        
        print("Training Hybrid Model...")
        
        try:
            model = HybridRecommender(component_models, weights)
            start_time = time.time()
            model.fit(train_data)
            training_time = time.time() - start_time
            
            self.trained_models['hybrid'] = model
            self.training_history['hybrid'] = {
                'component_models': list(component_models.keys()),
                'weights': weights,
                'training_time': training_time
            }
            
            print(f"Hybrid Model trained in {training_time:.2f}s")
            return model
            
        except Exception as e:
            print(f"Error training hybrid model: {e}")
            return None
    
    def _evaluate_content_model(self, model):
        """
        Evaluate content-based model using TF-IDF matrix properties.
        
        Args:
            model: Content-based model
            
        Returns:
            float: Evaluation score
        """
        if model.tfidf_matrix is None:
            return 0.0
        
        # Use sparsity and feature utilization as proxy metrics
        sparsity = 1.0 - (model.tfidf_matrix.nnz / (model.tfidf_matrix.shape[0] * model.tfidf_matrix.shape[1]))
        feature_utilization = len(model.tfidf.vocabulary_) / model.tfidf.max_features if model.tfidf.max_features else 1.0
        
        # Combine metrics (higher sparsity and good feature utilization are generally better)
        score = (sparsity * 0.7 + feature_utilization * 0.3)
        return score
    
    def _evaluate_knn_model(self, model, data):
        """
        Evaluate KNN model using cross-validation on sample recommendations.
        
        Args:
            model: KNN model
            data (pd.DataFrame): Data for evaluation
            
        Returns:
            float: Evaluation score
        """
        try:
            # Sample some movies for evaluation
            sample_size = min(20, len(data))
            sample_movies = data.sample(n=sample_size, random_state=self.random_state)
            
            scores = []
            for _, row in sample_movies.iterrows():
                try:
                    recs = model.get_recommendations(row['title'], 5)
                    # Score based on average similarity score
                    if not recs.empty and 'similarity_score' in recs.columns:
                        scores.append(recs['similarity_score'].mean())
                except:
                    continue
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            print(f"Error evaluating KNN model: {e}")
            return 0.0
    
    def _evaluate_clustering_model(self, model):
        """
        Evaluate clustering model using silhouette score.
        
        Args:
            model: Clustering model
            
        Returns:
            float: Silhouette score
        """
        try:
            from sklearn.metrics import silhouette_score
            
            # Calculate silhouette score
            if hasattr(model, 'data') and 'cluster' in model.data.columns:
                features = model.feature_matrix.toarray()
                labels = model.data['cluster'].values
                score = silhouette_score(features, labels)
                return score
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating silhouette score: {e}")
            return 0.0
    
    def evaluate_models(self, test_data, metrics=['precision', 'diversity', 'coverage']):
        """
        Evaluate all trained models on test data.
        
        Args:
            test_data (pd.DataFrame): Test dataset
            metrics (list): Metrics to calculate
            
        Returns:
            dict: Evaluation results for all models
        """
        print("Evaluating trained models...")
        
        results = {}
        
        for model_name, model in self.trained_models.items():
            print(f"\nEvaluating {model_name} model...")
            
            model_results = {}
            
            try:
                # Sample test movies
                sample_size = min(50, len(test_data))
                test_sample = test_data.sample(n=sample_size, random_state=self.random_state)
                
                all_recommendations = []
                successful_recs = 0
                
                for _, row in test_sample.iterrows():
                    try:
                        recs = model.get_recommendations(row['title'], 10)
                        if not recs.empty:
                            all_recommendations.append(recs)
                            successful_recs += 1
                    except:
                        continue
                
                # Calculate metrics
                if all_recommendations:
                    combined_recs = pd.concat(all_recommendations, ignore_index=True)
                    
                    if 'precision' in metrics:
                        model_results['precision'] = self._calculate_precision(combined_recs, test_data)
                    
                    if 'diversity' in metrics:
                        model_results['diversity'] = self._calculate_diversity(combined_recs)
                    
                    if 'coverage' in metrics:
                        model_results['coverage'] = self._calculate_coverage(combined_recs, test_data)
                    
                    model_results['success_rate'] = successful_recs / len(test_sample)
                    
                else:
                    model_results = {metric: 0.0 for metric in metrics}
                    model_results['success_rate'] = 0.0
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                model_results = {metric: 0.0 for metric in metrics}
                model_results['success_rate'] = 0.0
            
            results[model_name] = model_results
            print(f"{model_name} results: {model_results}")
        
        self.evaluation_results = results
        return results
    
    def _calculate_precision(self, recommendations, test_data):
        """
        Calculate precision based on genre matching.
        
        Args:
            recommendations (pd.DataFrame): Recommended movies
            test_data (pd.DataFrame): Test dataset
            
        Returns:
            float: Precision score
        """
        # Simple precision based on genre similarity
        # This is a proxy metric since we don't have explicit ratings
        if 'genres' not in recommendations.columns:
            return 0.0
        
        genre_matches = 0
        total_recs = len(recommendations)
        
        for _, rec in recommendations.iterrows():
            if rec['genres'] in test_data['genres'].values:
                genre_matches += 1
        
        return genre_matches / total_recs if total_recs > 0 else 0.0
    
    def _calculate_diversity(self, recommendations):
        """
        Calculate diversity of recommendations.
        
        Args:
            recommendations (pd.DataFrame): Recommended movies
            
        Returns:
            float: Diversity score
        """
        if 'genres' not in recommendations.columns:
            return 0.0
        
        unique_genres = len(set(recommendations['genres'].values))
        total_genres = len(recommendations)
        
        return unique_genres / total_genres if total_genres > 0 else 0.0
    
    def _calculate_coverage(self, recommendations, test_data):
        """
        Calculate coverage of recommendations.
        
        Args:
            recommendations (pd.DataFrame): Recommended movies
            test_data (pd.DataFrame): Test dataset
            
        Returns:
            float: Coverage score
        """
        unique_recommendations = len(set(recommendations['title'].values))
        total_movies = len(test_data)
        
        return unique_recommendations / total_movies if total_movies > 0 else 0.0
    
    def cross_validate_model(self, model, data, cv_folds=5):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Recommendation model
            data (pd.DataFrame): Dataset
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: Cross-validation results
        """
        from sklearn.model_selection import KFold
        
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            print(f"Training fold {fold + 1}/{cv_folds}...")
            
            train_fold = data.iloc[train_idx]
            val_fold = data.iloc[val_idx]
            
            try:
                # Train model on fold
                model.fit(train_fold)
                
                # Evaluate on validation fold
                fold_result = self._evaluate_fold(model, val_fold)
                fold_results.append(fold_result)
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {e}")
                fold_results.append({'score': 0.0})
        
        # Aggregate results
        cv_results = {
            'mean_score': np.mean([result['score'] for result in fold_results]),
            'std_score': np.std([result['score'] for result in fold_results]),
            'fold_results': fold_results
        }
        
        print(f"CV Results - Mean: {cv_results['mean_score']:.4f}, Std: {cv_results['std_score']:.4f}")
        return cv_results
    
    def _evaluate_fold(self, model, val_data):
        """
        Evaluate model performance on a validation fold.
        
        Args:
            model: Recommendation model
            val_data (pd.DataFrame): Validation data
            
        Returns:
            dict: Fold evaluation results
        """
        sample_size = min(20, len(val_data))
        sample_movies = val_data.sample(n=sample_size, random_state=self.random_state)
        
        scores = []
        for _, row in sample_movies.iterrows():
            try:
                recs = model.get_recommendations(row['title'], 5)
                if not recs.empty:
                    # Use average similarity score or other relevant metric
                    if 'similarity_score' in recs.columns:
                        scores.append(recs['similarity_score'].mean())
                    else:
                        scores.append(1.0)  # Default score if no similarity column
            except:
                continue
        
        return {'score': np.mean(scores) if scores else 0.0}
    
    def save_trained_models(self, save_dir='./trained_models/'):
        """
        Save all trained models to disk.
        
        Args:
            save_dir (str): Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            try:
                filepath = os.path.join(save_dir, f'{model_name}_model')
                
                if hasattr(model, 'autoencoder'):
                    # Save autoencoder models separately
                    model_data = {
                        'type': type(model).__name__,
                        'params': {k: v for k, v in model.__dict__.items() 
                                  if k not in ['autoencoder', 'encoder']},
                        'training_history': self.training_history.get(model_name, {})
                    }
                    joblib.dump(model_data, filepath + '_params.joblib')
                    
                    if hasattr(model, 'autoencoder') and model.autoencoder:
                        model.autoencoder.save(filepath + '_autoencoder.keras')
                    if hasattr(model, 'encoder') and model.encoder:
                        model.encoder.save(filepath + '_encoder.keras')
                else:
                    # Save regular models
                    model_data = {
                        'model': model,
                        'training_history': self.training_history.get(model_name, {})
                    }
                    joblib.dump(model_data, filepath + '.joblib')
                
                print(f"Saved {model_name} model to {filepath}")
                
            except Exception as e:
                print(f"Error saving {model_name} model: {e}")
        
        # Save training history and evaluation results
        history_filepath = os.path.join(save_dir, 'training_history.joblib')
        joblib.dump({
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results
        }, history_filepath)
        
        print(f"Saved training history to {history_filepath}")
    
    def print_training_summary(self):
        """
        Print a summary of all training results.
        """
        print("="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        for model_name, history in self.training_history.items():
            print(f"\n{model_name.upper()} MODEL:")
            print(f"  Trained: Yes")
            
            if 'best_params' in history:
                print(f"  Best Parameters: {history['best_params']}")
            if 'best_score' in history:
                print(f"  Best Score: {history['best_score']:.4f}")
            if 'training_time' in history:
                print(f"  Training Time: {history['training_time']:.2f}s")
        
        print("\n" + "="*60)
        
        if self.evaluation_results:
            print("EVALUATION RESULTS")
            print("="*60)
            
            for model_name, results in self.evaluation_results.items():
                print(f"\n{model_name.upper()} MODEL:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.4f}")


def main():
    """
    Example usage of the ModelTrainer class.
    """
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'title': [f'Movie {i}' for i in range(200)],
        'genres': (['Action'] * 50 + ['Drama'] * 50 + 
                  ['Comedy'] * 50 + ['Horror'] * 50),
        'overview': [f'This is a great movie about topic {i}' for i in range(200)],
        'soup': [f'movie {i} action adventure great story topic {i}' for i in range(200)],
        'overview_sentiment_score': np.random.normal(0.1, 0.3, 200)
    })
    
    print("Initializing Model Trainer...")
    trainer = ModelTrainer()
    
    # Prepare data
    train_data, test_data = trainer.prepare_training_data(sample_data)
    
    # Train models
    print("\nTraining models...")
    
    # Content-based model
    trainer.train_content_based_model(train_data)
    
    # KNN model
    trainer.train_knn_model(train_data)
    
    # Clustering model
    trainer.train_clustering_model(train_data)
    
    # Sentiment model
    trainer.train_sentiment_model(train_data)
    
    # Hybrid model
    trainer.train_hybrid_model(train_data)
    
    # Evaluate models
    trainer.evaluate_models(test_data)
    
    # Print summary
    trainer.print_training_summary()
    
    # Save models
    trainer.save_trained_models()
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
