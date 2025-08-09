"""
Model Optimization Module for Movie Recommendation System
=======================================================

This module provides advanced optimization techniques for recommendation models,
including hyperparameter tuning, ensemble methods, and performance optimization.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, BayesSearchCV
from sklearn.metrics import make_scorer
import optuna
import joblib
import time
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class ModelOptimizer:
    """
    Advanced model optimization class using various hyperparameter tuning techniques.
    """
    
    def __init__(self, random_state=42, n_jobs=-1):
        """
        Initialize the model optimizer.
        
        Args:
            random_state (int): Random state for reproducibility
            n_jobs (int): Number of parallel jobs
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.optimization_results = {}
        self.best_models = {}
        self.optimization_history = defaultdict(list)
    
    def optimize_content_based_model(self, train_data, optimization_method='grid', n_trials=100):
        """
        Optimize content-based recommendation model.
        
        Args:
            train_data (pd.DataFrame): Training data
            optimization_method (str): Optimization method ('grid', 'random', 'bayes', 'optuna')
            n_trials (int): Number of trials for optuna optimization
            
        Returns:
            dict: Optimization results
        """
        print(f"Optimizing Content-Based Model using {optimization_method} search...")
        
        # Define parameter space
        if optimization_method == 'optuna':
            return self._optimize_content_optuna(train_data, n_trials)
        else:
            param_grid = {
                'max_features': [1000, 5000, 10000, 20000, None],
                'ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],
                'min_df': [1, 2, 5],
                'max_df': [0.8, 0.9, 1.0]
            }
            
            return self._optimize_sklearn_model(
                'content_based', 
                train_data, 
                param_grid, 
                optimization_method
            )
    
    def optimize_knn_model(self, train_data, optimization_method='grid', n_trials=50):
        """
        Optimize KNN recommendation model.
        
        Args:
            train_data (pd.DataFrame): Training data
            optimization_method (str): Optimization method
            n_trials (int): Number of trials for optuna optimization
            
        Returns:
            dict: Optimization results
        """
        print(f"Optimizing KNN Model using {optimization_method} search...")
        
        if optimization_method == 'optuna':
            return self._optimize_knn_optuna(train_data, n_trials)
        else:
            param_grid = {
                'n_neighbors': [3, 5, 10, 15, 20, 25, 30],
                'metric': ['cosine', 'euclidean', 'manhattan'],
                'max_features': [1000, 5000, 10000]
            }
            
            return self._optimize_sklearn_model(
                'knn', 
                train_data, 
                param_grid, 
                optimization_method
            )
    
    def optimize_clustering_model(self, train_data, optimization_method='grid', n_trials=30):
        """
        Optimize clustering recommendation model.
        
        Args:
            train_data (pd.DataFrame): Training data
            optimization_method (str): Optimization method
            n_trials (int): Number of trials for optuna optimization
            
        Returns:
            dict: Optimization results
        """
        print(f"Optimizing Clustering Model using {optimization_method} search...")
        
        if optimization_method == 'optuna':
            return self._optimize_clustering_optuna(train_data, n_trials)
        else:
            param_grid = {
                'n_clusters': [5, 10, 15, 20, 25, 30, 40],
                'max_features': [1000, 5000, 10000],
                'n_init': [10, 20]
            }
            
            return self._optimize_sklearn_model(
                'clustering', 
                train_data, 
                param_grid, 
                optimization_method
            )
    
    def optimize_autoencoder_model(self, train_data, n_trials=20):
        """
        Optimize autoencoder model using Optuna.
        
        Args:
            train_data (pd.DataFrame): Training data
            n_trials (int): Number of optimization trials
            
        Returns:
            dict: Optimization results
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Skipping autoencoder optimization.")
            return None
        
        print(f"Optimizing Autoencoder Model using Optuna with {n_trials} trials...")
        
        def objective(trial):
            # Suggest hyperparameters
            encoding_dim = trial.suggest_categorical('encoding_dim', [32, 64, 128, 256])
            epochs = trial.suggest_int('epochs', 10, 100)
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
            
            try:
                from .model_design import AutoencoderRecommender
                
                # Create and train model
                model = AutoencoderRecommender(
                    encoding_dim=encoding_dim,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                # Custom training with suggested parameters
                history = model.fit(train_data)
                
                # Return validation loss
                val_loss = min(history.history.get('val_loss', history.history['loss']))
                return val_loss
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Store results
        best_params = study.best_params
        best_score = study.best_value
        
        self.optimization_results['autoencoder'] = {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
        
        print(f"Best Autoencoder Parameters: {best_params}")
        print(f"Best Validation Loss: {best_score:.4f}")
        
        return self.optimization_results['autoencoder']
    
    def _optimize_content_optuna(self, train_data, n_trials):
        """
        Optimize content-based model using Optuna.
        """
        def objective(trial):
            # Suggest hyperparameters
            max_features = trial.suggest_categorical('max_features', [1000, 5000, 10000, 20000])
            ngram_range = trial.suggest_categorical('ngram_range', [(1, 1), (1, 2), (1, 3)])
            min_df = trial.suggest_int('min_df', 1, 10)
            max_df = trial.suggest_float('max_df', 0.8, 1.0)
            
            try:
                from .model_design import ContentBasedRecommender
                
                model = ContentBasedRecommender(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words='english'
                )
                model.tfidf.min_df = min_df
                model.tfidf.max_df = max_df
                
                model.fit(train_data)
                
                # Evaluate using custom metric
                score = self._evaluate_content_model(model, train_data)
                return score
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        result = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
        
        self.optimization_results['content_based'] = result
        print(f"Best Content-Based Parameters: {study.best_params}")
        print(f"Best Score: {study.best_value:.4f}")
        
        return result
    
    def _optimize_knn_optuna(self, train_data, n_trials):
        """
        Optimize KNN model using Optuna.
        """
        def objective(trial):
            n_neighbors = trial.suggest_int('n_neighbors', 3, 50)
            metric = trial.suggest_categorical('metric', ['cosine', 'euclidean'])
            max_features = trial.suggest_categorical('max_features', [1000, 5000, 10000])
            
            try:
                from .model_design import KNNRecommender
                
                model = KNNRecommender(n_neighbors=n_neighbors, metric=metric)
                model.tfidf.max_features = max_features
                model.fit(train_data)
                
                score = self._evaluate_knn_model(model, train_data)
                return score
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        result = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
        
        self.optimization_results['knn'] = result
        print(f"Best KNN Parameters: {study.best_params}")
        print(f"Best Score: {study.best_value:.4f}")
        
        return result
    
    def _optimize_clustering_optuna(self, train_data, n_trials):
        """
        Optimize clustering model using Optuna.
        """
        def objective(trial):
            n_clusters = trial.suggest_int('n_clusters', 5, 50)
            max_features = trial.suggest_categorical('max_features', [1000, 5000, 10000])
            
            try:
                from .model_design import ClusteringRecommender
                
                model = ClusteringRecommender(
                    n_clusters=n_clusters, 
                    random_state=self.random_state
                )
                model.tfidf.max_features = max_features
                model.fit(train_data)
                
                score = self._evaluate_clustering_model(model)
                return score
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        result = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
        
        self.optimization_results['clustering'] = result
        print(f"Best Clustering Parameters: {study.best_params}")
        print(f"Best Score: {study.best_value:.4f}")
        
        return result
    
    def _optimize_sklearn_model(self, model_name, train_data, param_grid, method):
        """
        Optimize model using scikit-learn optimization methods.
        """
        from .model_design import create_recommendation_pipeline
        
        # Create base model
        base_model = create_recommendation_pipeline(model_name)
        
        # Create scoring function
        def custom_scorer(estimator, X, y=None):
            try:
                if model_name == 'content_based':
                    return self._evaluate_content_model(estimator, train_data)
                elif model_name == 'knn':
                    return self._evaluate_knn_model(estimator, train_data)
                elif model_name == 'clustering':
                    return self._evaluate_clustering_model(estimator)
                else:
                    return 0.0
            except:
                return 0.0
        
        scorer = make_scorer(custom_scorer, greater_is_better=True)
        
        # Choose optimization method
        if method == 'grid':
            optimizer = GridSearchCV(
                base_model, 
                param_grid, 
                scoring=scorer,
                cv=3,
                n_jobs=self.n_jobs,
                verbose=1
            )
        elif method == 'random':
            optimizer = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=20,
                scoring=scorer,
                cv=3,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        elif method == 'bayes' and SKOPT_AVAILABLE:
            optimizer = BayesSearchCV(
                base_model,
                param_grid,
                n_iter=20,
                scoring=scorer,
                cv=3,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )
        else:
            print(f"Method {method} not available. Using grid search.")
            optimizer = GridSearchCV(
                base_model, 
                param_grid, 
                scoring=scorer,
                cv=3,
                n_jobs=self.n_jobs
            )
        
        # Fit optimizer
        X_dummy = train_data[['title']].copy()  # Dummy features for sklearn compatibility
        optimizer.fit(X_dummy)
        
        result = {
            'best_params': optimizer.best_params_,
            'best_score': optimizer.best_score_,
            'cv_results': optimizer.cv_results_
        }
        
        self.optimization_results[model_name] = result
        print(f"Best {model_name} Parameters: {optimizer.best_params_}")
        print(f"Best Score: {optimizer.best_score_:.4f}")
        
        return result
    
    def optimize_hybrid_weights(self, train_data, component_models, n_trials=50):
        """
        Optimize weights for hybrid model using Optuna.
        
        Args:
            train_data (pd.DataFrame): Training data
            component_models (dict): Component models
            n_trials (int): Number of optimization trials
            
        Returns:
            dict: Optimization results
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Using equal weights.")
            return {'best_weights': {name: 1.0/len(component_models) for name in component_models.keys()}}
        
        print(f"Optimizing hybrid model weights with {n_trials} trials...")
        
        def objective(trial):
            # Suggest weights for each component model
            weights = {}
            weight_sum = 0
            
            for model_name in component_models.keys():
                weight = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
                weights[model_name] = weight
                weight_sum += weight
            
            # Normalize weights
            if weight_sum > 0:
                weights = {name: weight/weight_sum for name, weight in weights.items()}
            else:
                weights = {name: 1.0/len(component_models) for name in component_models.keys()}
            
            try:
                from .model_design import HybridRecommender
                
                # Create and fit hybrid model
                hybrid_model = HybridRecommender(component_models, weights)
                hybrid_model.fit(train_data)
                
                # Evaluate hybrid model
                score = self._evaluate_hybrid_model(hybrid_model, train_data)
                return score
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        result = {
            'best_weights': {k.replace('weight_', ''): v for k, v in study.best_params.items()},
            'best_score': study.best_value,
            'study': study
        }
        
        self.optimization_results['hybrid_weights'] = result
        print(f"Best Hybrid Weights: {result['best_weights']}")
        print(f"Best Score: {study.best_value:.4f}")
        
        return result
    
    def create_ensemble_model(self, models, ensemble_method='voting', weights=None):
        """
        Create an ensemble of recommendation models.
        
        Args:
            models (dict): Dictionary of trained models
            ensemble_method (str): Ensemble method ('voting', 'weighted', 'stacking')
            weights (dict): Weights for weighted ensemble
            
        Returns:
            EnsembleRecommender: Ensemble model
        """
        print(f"Creating ensemble model using {ensemble_method} method...")
        
        if ensemble_method == 'voting':
            return VotingEnsemble(models)
        elif ensemble_method == 'weighted':
            return WeightedEnsemble(models, weights)
        elif ensemble_method == 'stacking':
            return StackingEnsemble(models)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    def _evaluate_content_model(self, model, data):
        """Evaluate content-based model."""
        try:
            if hasattr(model, 'tfidf_matrix') and model.tfidf_matrix is not None:
                sparsity = 1.0 - (model.tfidf_matrix.nnz / (model.tfidf_matrix.shape[0] * model.tfidf_matrix.shape[1]))
                return sparsity
            return 0.0
        except:
            return 0.0
    
    def _evaluate_knn_model(self, model, data):
        """Evaluate KNN model."""
        try:
            sample_size = min(10, len(data))
            sample_movies = data.sample(n=sample_size, random_state=self.random_state)
            
            scores = []
            for _, row in sample_movies.iterrows():
                try:
                    recs = model.get_recommendations(row['title'], 5)
                    if not recs.empty and 'similarity_score' in recs.columns:
                        scores.append(recs['similarity_score'].mean())
                except:
                    continue
            
            return np.mean(scores) if scores else 0.0
        except:
            return 0.0
    
    def _evaluate_clustering_model(self, model):
        """Evaluate clustering model."""
        try:
            from sklearn.metrics import silhouette_score
            
            if hasattr(model, 'data') and 'cluster' in model.data.columns:
                features = model.feature_matrix.toarray()
                labels = model.data['cluster'].values
                score = silhouette_score(features, labels)
                return score
            return 0.0
        except:
            return 0.0
    
    def _evaluate_hybrid_model(self, model, data):
        """Evaluate hybrid model."""
        try:
            sample_size = min(10, len(data))
            sample_movies = data.sample(n=sample_size, random_state=self.random_state)
            
            scores = []
            for _, row in sample_movies.iterrows():
                try:
                    recs = model.get_recommendations(row['title'], 5)
                    if not recs.empty and 'combined_score' in recs.columns:
                        scores.append(recs['combined_score'].mean())
                except:
                    continue
            
            return np.mean(scores) if scores else 0.0
        except:
            return 0.0
    
    def compare_optimization_methods(self, train_data, model_type='content_based'):
        """
        Compare different optimization methods for a model.
        
        Args:
            train_data (pd.DataFrame): Training data
            model_type (str): Type of model to optimize
            
        Returns:
            dict: Comparison results
        """
        print(f"Comparing optimization methods for {model_type} model...")
        
        methods = ['grid', 'random']
        if SKOPT_AVAILABLE:
            methods.append('bayes')
        if OPTUNA_AVAILABLE:
            methods.append('optuna')
        
        results = {}
        
        for method in methods:
            print(f"\nTesting {method} search...")
            start_time = time.time()
            
            try:
                if model_type == 'content_based':
                    result = self.optimize_content_based_model(train_data, method, n_trials=20)
                elif model_type == 'knn':
                    result = self.optimize_knn_model(train_data, method, n_trials=20)
                elif model_type == 'clustering':
                    result = self.optimize_clustering_model(train_data, method, n_trials=20)
                else:
                    continue
                
                optimization_time = time.time() - start_time
                
                results[method] = {
                    'best_score': result['best_score'],
                    'best_params': result['best_params'],
                    'optimization_time': optimization_time
                }
                
                print(f"{method} - Score: {result['best_score']:.4f}, Time: {optimization_time:.2f}s")
                
            except Exception as e:
                print(f"Error with {method}: {e}")
                results[method] = {'error': str(e)}
        
        # Find best method
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_method = max(valid_results.keys(), key=lambda k: valid_results[k]['best_score'])
            results['best_method'] = best_method
            print(f"\nBest optimization method: {best_method}")
        
        return results
    
    def save_optimization_results(self, filepath='optimization_results.joblib'):
        """
        Save optimization results to file.
        
        Args:
            filepath (str): Path to save results
        """
        results_to_save = {}
        
        for model_name, result in self.optimization_results.items():
            # Remove study objects which can't be pickled easily
            clean_result = result.copy()
            if 'study' in clean_result:
                clean_result['study_best_params'] = clean_result['study'].best_params
                clean_result['study_best_value'] = clean_result['study'].best_value
                del clean_result['study']
            
            results_to_save[model_name] = clean_result
        
        joblib.dump(results_to_save, filepath)
        print(f"Optimization results saved to {filepath}")
    
    def print_optimization_summary(self):
        """
        Print a summary of optimization results.
        """
        print("="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        for model_name, result in self.optimization_results.items():
            print(f"\n{model_name.upper()} MODEL:")
            print(f"  Best Score: {result.get('best_score', 'N/A')}")
            print(f"  Best Parameters: {result.get('best_params', 'N/A')}")
            
            if 'optimization_time' in result:
                print(f"  Optimization Time: {result['optimization_time']:.2f}s")


class VotingEnsemble:
    """
    Voting ensemble for recommendation models.
    """
    
    def __init__(self, models):
        """
        Initialize voting ensemble.
        
        Args:
            models (dict): Dictionary of models
        """
        self.models = models
        self.data = None
    
    def fit(self, data, content_column='soup'):
        """
        Fit all models in ensemble.
        
        Args:
            data (pd.DataFrame): Training data
            content_column (str): Content column name
        """
        self.data = data
        for name, model in self.models.items():
            print(f"Fitting {name} model...")
            model.fit(data, content_column)
    
    def get_recommendations(self, title, num_recommendations=10):
        """
        Get recommendations using voting ensemble.
        
        Args:
            title (str): Movie title
            num_recommendations (int): Number of recommendations
            
        Returns:
            pd.DataFrame: Ensemble recommendations
        """
        all_recommendations = defaultdict(int)
        
        # Get recommendations from each model
        for name, model in self.models.items():
            try:
                recs = model.get_recommendations(title, num_recommendations * 2)
                if not recs.empty:
                    for _, row in recs.iterrows():
                        all_recommendations[row['title']] += 1
            except:
                continue
        
        # Sort by votes
        sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Create result DataFrame
        if sorted_recs:
            top_recs = sorted_recs[:num_recommendations]
            result_data = []
            
            for movie_title, votes in top_recs:
                movie_info = self.data[self.data['title'] == movie_title]
                if not movie_info.empty:
                    movie_row = movie_info.iloc[0]
                    result_data.append({
                        'title': movie_title,
                        'votes': votes,
                        'genres': movie_row.get('genres', ''),
                        'overview': movie_row.get('overview', '')
                    })
            
            return pd.DataFrame(result_data)
        else:
            return pd.DataFrame()


class WeightedEnsemble:
    """
    Weighted ensemble for recommendation models.
    """
    
    def __init__(self, models, weights=None):
        """
        Initialize weighted ensemble.
        
        Args:
            models (dict): Dictionary of models
            weights (dict): Model weights
        """
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models.keys()}
        self.data = None
    
    def fit(self, data, content_column='soup'):
        """
        Fit all models in ensemble.
        
        Args:
            data (pd.DataFrame): Training data
            content_column (str): Content column name
        """
        self.data = data
        for name, model in self.models.items():
            print(f"Fitting {name} model...")
            model.fit(data, content_column)
    
    def get_recommendations(self, title, num_recommendations=10):
        """
        Get recommendations using weighted ensemble.
        
        Args:
            title (str): Movie title
            num_recommendations (int): Number of recommendations
            
        Returns:
            pd.DataFrame: Ensemble recommendations
        """
        all_recommendations = defaultdict(float)
        
        # Get weighted recommendations from each model
        for name, model in self.models.items():
            try:
                recs = model.get_recommendations(title, num_recommendations * 2)
                weight = self.weights.get(name, 0.0)
                
                if not recs.empty:
                    for _, row in recs.iterrows():
                        score = row.get('similarity_score', 1.0)
                        all_recommendations[row['title']] += score * weight
            except:
                continue
        
        # Sort by weighted scores
        sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Create result DataFrame
        if sorted_recs:
            top_recs = sorted_recs[:num_recommendations]
            result_data = []
            
            for movie_title, weighted_score in top_recs:
                movie_info = self.data[self.data['title'] == movie_title]
                if not movie_info.empty:
                    movie_row = movie_info.iloc[0]
                    result_data.append({
                        'title': movie_title,
                        'weighted_score': weighted_score,
                        'genres': movie_row.get('genres', ''),
                        'overview': movie_row.get('overview', '')
                    })
            
            return pd.DataFrame(result_data)
        else:
            return pd.DataFrame()


class StackingEnsemble:
    """
    Stacking ensemble for recommendation models.
    """
    
    def __init__(self, models):
        """
        Initialize stacking ensemble.
        
        Args:
            models (dict): Dictionary of models
        """
        self.models = models
        self.meta_learner = None
        self.data = None
    
    def fit(self, data, content_column='soup'):
        """
        Fit stacking ensemble.
        
        Args:
            data (pd.DataFrame): Training data
            content_column (str): Content column name
        """
        self.data = data
        
        # Fit base models
        for name, model in self.models.items():
            print(f"Fitting {name} model...")
            model.fit(data, content_column)
        
        # Train meta-learner (simplified for this example)
        print("Training meta-learner...")
        # In practice, this would use cross-validation predictions from base models
        # For simplicity, we'll use equal weights
        self.meta_learner = {name: 1.0/len(self.models) for name in self.models.keys()}
    
    def get_recommendations(self, title, num_recommendations=10):
        """
        Get recommendations using stacking ensemble.
        
        Args:
            title (str): Movie title
            num_recommendations (int): Number of recommendations
            
        Returns:
            pd.DataFrame: Ensemble recommendations
        """
        # For simplicity, this acts like a weighted ensemble
        # In practice, the meta-learner would make the final decision
        all_recommendations = defaultdict(float)
        
        for name, model in self.models.items():
            try:
                recs = model.get_recommendations(title, num_recommendations * 2)
                weight = self.meta_learner.get(name, 0.0)
                
                if not recs.empty:
                    for _, row in recs.iterrows():
                        score = row.get('similarity_score', 1.0)
                        all_recommendations[row['title']] += score * weight
            except:
                continue
        
        # Sort and return results
        sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_recs:
            top_recs = sorted_recs[:num_recommendations]
            result_data = []
            
            for movie_title, score in top_recs:
                movie_info = self.data[self.data['title'] == movie_title]
                if not movie_info.empty:
                    movie_row = movie_info.iloc[0]
                    result_data.append({
                        'title': movie_title,
                        'ensemble_score': score,
                        'genres': movie_row.get('genres', ''),
                        'overview': movie_row.get('overview', '')
                    })
            
            return pd.DataFrame(result_data)
        else:
            return pd.DataFrame()


def main():
    """
    Example usage of the ModelOptimizer class.
    """
    # Create sample data
    sample_data = pd.DataFrame({
        'title': [f'Movie {i}' for i in range(100)],
        'genres': (['Action'] * 25 + ['Drama'] * 25 + 
                  ['Comedy'] * 25 + ['Horror'] * 25),
        'overview': [f'This is a great movie about topic {i}' for i in range(100)],
        'soup': [f'movie {i} action adventure great story topic {i}' for i in range(100)]
    })
    
    print("Initializing Model Optimizer...")
    optimizer = ModelOptimizer()
    
    # Optimize models
    print("\nOptimizing models...")
    
    # Content-based optimization
    optimizer.optimize_content_based_model(sample_data, 'optuna', n_trials=10)
    
    # KNN optimization
    optimizer.optimize_knn_model(sample_data, 'optuna', n_trials=10)
    
    # Clustering optimization
    optimizer.optimize_clustering_model(sample_data, 'optuna', n_trials=10)
    
    # Print summary
    optimizer.print_optimization_summary()
    
    # Save results
    optimizer.save_optimization_results()
    
    print("\nOptimization completed!")


if __name__ == "__main__":
    main()
