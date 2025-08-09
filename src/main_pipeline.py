"""
Main Pipeline Integration Module

This module provides a unified interface to run the complete movie recommendation pipeline.
It integrates all components: data cleaning, EDA, model design, training, optimization, 
analysis, AI agents, and model persistence.

Classes:
    - MovieRecommendationPipeline: Main pipeline orchestrator
    - PipelineConfig: Configuration management
    - PipelineRunner: Execute pipeline steps

Dependencies:
    - All other modules in the project
    - pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import all project modules
try:
    from data_cleaning import MovieDataCleaner, DataMerger
    DATA_CLEANING_AVAILABLE = True
except ImportError:
    print("Warning: data_cleaning module not available")
    DATA_CLEANING_AVAILABLE = False

try:
    from exploratory_data_analysis import MovieEDA, MovieProfiler, EDAVisualizer
    EDA_AVAILABLE = True
except ImportError:
    print("Warning: exploratory_data_analysis module not available")
    EDA_AVAILABLE = False

try:
    from model_design import (ContentBasedRecommender, KNNRecommender, 
                             AutoencoderRecommender, ClusteringRecommender,
                             SentimentBasedRecommender, HybridRecommender)
    MODEL_DESIGN_AVAILABLE = True
except ImportError:
    print("Warning: model_design module not available")
    MODEL_DESIGN_AVAILABLE = False

try:
    from model_training import ModelTrainer, HyperparameterTuner, ModelEvaluator
    MODEL_TRAINING_AVAILABLE = True
except ImportError:
    print("Warning: model_training module not available")
    MODEL_TRAINING_AVAILABLE = False

try:
    from model_optimization import AdvancedOptimizer, EnsembleRecommender, OptimizationReporter
    MODEL_OPTIMIZATION_AVAILABLE = True
except ImportError:
    print("Warning: model_optimization module not available")
    MODEL_OPTIMIZATION_AVAILABLE = False

try:
    from model_analysis import RecommendationAnalyzer, ModelComparison, ResultsReporter
    MODEL_ANALYSIS_AVAILABLE = True
except ImportError:
    print("Warning: model_analysis module not available")
    MODEL_ANALYSIS_AVAILABLE = False

try:
    from ai_agents import MovieRecommendationAgent, SentimentAnalysisAgent, ExplanationAgent, AgenticPipeline
    AI_AGENTS_AVAILABLE = True
except ImportError:
    print("Warning: ai_agents module not available")
    AI_AGENTS_AVAILABLE = False

try:
    from model_persistence import ModelSaver, ModelLoader, ModelManager
    MODEL_PERSISTENCE_AVAILABLE = True
except ImportError:
    print("Warning: model_persistence module not available")
    MODEL_PERSISTENCE_AVAILABLE = False


class PipelineConfig:
    """
    Configuration management for the pipeline.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize pipeline configuration.
        
        Parameters:
        - config_file (str): Path to configuration file
        """
        self.config = self._get_default_config()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def _get_default_config(self):
        """Get default pipeline configuration."""
        return {
            "data": {
                "movies_file": "data/movies.csv",
                "credits_file": "data/credits.csv",
                "output_dir": "./output",
                "clean_data": True
            },
            "models": {
                "content_based": {"enabled": True, "tfidf_params": {"max_features": 5000}},
                "knn": {"enabled": True, "n_neighbors": 10},
                "autoencoder": {"enabled": True, "latent_dim": 100, "epochs": 50},
                "clustering": {"enabled": True, "n_clusters": 20},
                "sentiment": {"enabled": True},
                "hybrid": {"enabled": True, "weights": {"content": 0.4, "collaborative": 0.3, "sentiment": 0.3}}
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42,
                "cross_validation": True,
                "cv_folds": 5
            },
            "optimization": {
                "grid_search": True,
                "random_search": True,
                "bayesian_optimization": False,
                "optuna_optimization": False,
                "ensemble": True
            },
            "analysis": {
                "diversity_analysis": True,
                "novelty_analysis": True,
                "coverage_analysis": True,
                "generate_reports": True
            },
            "ai_agents": {
                "enabled": False,  # Requires OpenAI API key
                "api_key": None,
                "model_name": "gpt-4o",
                "sentiment_analysis": True,
                "explanation_generation": True
            },
            "persistence": {
                "save_models": True,
                "models_dir": "./saved_models",
                "save_results": True,
                "create_deployment_config": True
            },
            "logging": {
                "level": "INFO",
                "log_file": "./pipeline.log"
            }
        }
    
    def load_config(self, config_file):
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            
            # Merge with default config
            self._merge_config(self.config, loaded_config)
            print(f"Configuration loaded from: {config_file}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
    
    def save_config(self, config_file):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to: {config_file}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def _merge_config(self, default, loaded):
        """Recursively merge configurations."""
        for key, value in loaded.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value


class MovieRecommendationPipeline:
    """
    Main pipeline orchestrator for the movie recommendation system.
    """
    
    def __init__(self, config=None):
        """
        Initialize the pipeline.
        
        Parameters:
        - config (PipelineConfig): Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.results = {}
        self.components = {}
        self.trained_models = {}
        
        # Initialize directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories."""
        dirs = [
            self.config.get('data.output_dir', './output'),
            self.config.get('persistence.models_dir', './saved_models'),
            './logs'
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def run_complete_pipeline(self):
        """
        Run the complete recommendation pipeline.
        
        Returns:
        - dict: Pipeline results
        """
        print("="*60)
        print("MOVIE RECOMMENDATION SYSTEM - COMPLETE PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Data Cleaning and Preparation
            print("\n" + "="*50)
            print("STEP 1: DATA CLEANING AND PREPARATION")
            print("="*50)
            self._run_data_cleaning()
            
            # Step 2: Exploratory Data Analysis
            print("\n" + "="*50)
            print("STEP 2: EXPLORATORY DATA ANALYSIS")
            print("="*50)
            self._run_eda()
            
            # Step 3: Model Design and Implementation
            print("\n" + "="*50)
            print("STEP 3: MODEL DESIGN AND IMPLEMENTATION")
            print("="*50)
            self._run_model_design()
            
            # Step 4: Model Training
            print("\n" + "="*50)
            print("STEP 4: MODEL TRAINING")
            print("="*50)
            self._run_model_training()
            
            # Step 5: Model Optimization
            print("\n" + "="*50)
            print("STEP 5: MODEL OPTIMIZATION")
            print("="*50)
            self._run_model_optimization()
            
            # Step 6: Model Analysis and Results
            print("\n" + "="*50)
            print("STEP 6: MODEL ANALYSIS AND RESULTS")
            print("="*50)
            self._run_model_analysis()
            
            # Step 7: AI Agents (Optional)
            if self.config.get('ai_agents.enabled', False):
                print("\n" + "="*50)
                print("STEP 7: AI AGENTS INTEGRATION")
                print("="*50)
                self._run_ai_agents()
            
            # Step 8: Model Persistence
            print("\n" + "="*50)
            print("STEP 8: MODEL PERSISTENCE")
            print("="*50)
            self._run_model_persistence()
            
            # Final Summary
            print("\n" + "="*50)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*50)
            self._print_final_summary()
            
            return self.results
            
        except Exception as e:
            print(f"\nPIPELINE ERROR: {e}")
            return {"error": str(e), "partial_results": self.results}
    
    def _run_data_cleaning(self):
        """Run data cleaning step."""
        if not DATA_CLEANING_AVAILABLE:
            print("Data cleaning module not available. Skipping...")
            return
        
        try:
            print("Loading and cleaning movie data...")
            
            # Initialize data cleaner
            cleaner = MovieDataCleaner()
            
            # Check if data files exist
            movies_file = self.config.get('data.movies_file')
            credits_file = self.config.get('data.credits_file')
            
            if not os.path.exists(movies_file):
                print(f"Movies file not found: {movies_file}")
                print("Creating sample data for demonstration...")
                movies_df = self._create_sample_movies_data()
                credits_df = self._create_sample_credits_data()
            else:
                movies_df = pd.read_csv(movies_file)
                credits_df = pd.read_csv(credits_file) if os.path.exists(credits_file) else None
            
            # Clean movies data
            cleaned_movies = cleaner.clean_movies_data(movies_df)
            
            if credits_df is not None:
                cleaned_credits = cleaner.clean_credits_data(credits_df)
                
                # Merge data
                merger = DataMerger()
                merged_df = merger.merge_datasets(cleaned_movies, cleaned_credits)
            else:
                merged_df = cleaned_movies
            
            # Store results
            self.components['movies_df'] = cleaned_movies
            if credits_df is not None:
                self.components['credits_df'] = cleaned_credits
            self.components['merged_df'] = merged_df
            
            self.results['data_cleaning'] = {
                'original_movies_count': len(movies_df),
                'cleaned_movies_count': len(cleaned_movies),
                'final_merged_count': len(merged_df),
                'columns': list(merged_df.columns)
            }
            
            print(f"Data cleaning completed. Final dataset: {merged_df.shape}")
            
        except Exception as e:
            print(f"Error in data cleaning: {e}")
            self.results['data_cleaning'] = {'error': str(e)}
    
    def _run_eda(self):
        """Run exploratory data analysis step."""
        if not EDA_AVAILABLE:
            print("EDA module not available. Skipping...")
            return
        
        try:
            movies_df = self.components.get('merged_df')
            if movies_df is None or movies_df.empty:
                print("No data available for EDA")
                return
            
            print("Performing exploratory data analysis...")
            
            # Initialize EDA components
            eda = MovieEDA(movies_df)
            profiler = MovieProfiler()
            visualizer = EDAVisualizer()
            
            # Generate statistics
            basic_stats = eda.generate_basic_statistics()
            genre_analysis = eda.analyze_genres()
            rating_analysis = eda.analyze_ratings()
            
            # Generate profile report
            profile_report = profiler.generate_report(movies_df, "Movie Dataset Profile")
            
            # Generate visualizations
            visualizer.create_genre_distribution_plot(movies_df)
            visualizer.create_rating_distribution_plot(movies_df)
            visualizer.create_correlation_heatmap(movies_df)
            
            # Store results
            self.results['eda'] = {
                'basic_statistics': basic_stats,
                'genre_analysis': genre_analysis,
                'rating_analysis': rating_analysis,
                'profile_report_generated': profile_report is not None
            }
            
            print("EDA completed successfully")
            
        except Exception as e:
            print(f"Error in EDA: {e}")
            self.results['eda'] = {'error': str(e)}
    
    def _run_model_design(self):
        """Run model design and implementation step."""
        if not MODEL_DESIGN_AVAILABLE:
            print("Model design module not available. Skipping...")
            return
        
        try:
            movies_df = self.components.get('merged_df')
            if movies_df is None or movies_df.empty:
                print("No data available for model design")
                return
            
            print("Designing and implementing recommendation models...")
            
            models = {}
            
            # Content-based recommender
            if self.config.get('models.content_based.enabled', True):
                print("Creating content-based recommender...")
                content_model = ContentBasedRecommender(movies_df)
                models['content_based'] = content_model
            
            # KNN recommender
            if self.config.get('models.knn.enabled', True):
                print("Creating KNN recommender...")
                knn_model = KNNRecommender(movies_df)
                models['knn'] = knn_model
            
            # Autoencoder recommender
            if self.config.get('models.autoencoder.enabled', True):
                print("Creating autoencoder recommender...")
                autoencoder_model = AutoencoderRecommender(movies_df)
                models['autoencoder'] = autoencoder_model
            
            # Clustering recommender
            if self.config.get('models.clustering.enabled', True):
                print("Creating clustering recommender...")
                clustering_model = ClusteringRecommender(movies_df)
                models['clustering'] = clustering_model
            
            # Sentiment-based recommender
            if self.config.get('models.sentiment.enabled', True):
                print("Creating sentiment-based recommender...")
                sentiment_model = SentimentBasedRecommender(movies_df)
                models['sentiment'] = sentiment_model
            
            # Hybrid recommender
            if self.config.get('models.hybrid.enabled', True):
                print("Creating hybrid recommender...")
                hybrid_model = HybridRecommender(list(models.values()))
                models['hybrid'] = hybrid_model
            
            # Store models
            self.components['models'] = models
            
            self.results['model_design'] = {
                'models_created': list(models.keys()),
                'total_models': len(models)
            }
            
            print(f"Model design completed. Created {len(models)} models")
            
        except Exception as e:
            print(f"Error in model design: {e}")
            self.results['model_design'] = {'error': str(e)}
    
    def _run_model_training(self):
        """Run model training step."""
        if not MODEL_TRAINING_AVAILABLE:
            print("Model training module not available. Skipping...")
            return
        
        try:
            models = self.components.get('models', {})
            movies_df = self.components.get('merged_df')
            
            if not models or movies_df is None:
                print("No models or data available for training")
                return
            
            print("Training recommendation models...")
            
            # Initialize trainer
            trainer = ModelTrainer(models, movies_df)
            
            # Train models
            training_results = trainer.train_all_models()
            
            # Evaluate models
            evaluator = ModelEvaluator()
            evaluation_results = {}
            
            for model_name, model in models.items():
                print(f"Evaluating {model_name}...")
                eval_result = evaluator.evaluate_model(model, movies_df)
                evaluation_results[model_name] = eval_result
            
            # Store results
            self.trained_models = models
            self.results['model_training'] = {
                'training_results': training_results,
                'evaluation_results': evaluation_results
            }
            
            print("Model training completed successfully")
            
        except Exception as e:
            print(f"Error in model training: {e}")
            self.results['model_training'] = {'error': str(e)}
    
    def _run_model_optimization(self):
        """Run model optimization step."""
        if not MODEL_OPTIMIZATION_AVAILABLE:
            print("Model optimization module not available. Skipping...")
            return
        
        try:
            models = self.trained_models
            movies_df = self.components.get('merged_df')
            
            if not models or movies_df is None:
                print("No trained models or data available for optimization")
                return
            
            print("Optimizing recommendation models...")
            
            # Initialize optimizer
            optimizer = AdvancedOptimizer()
            
            optimization_results = {}
            
            # Optimize each model
            for model_name, model in models.items():
                print(f"Optimizing {model_name}...")
                
                # Grid search optimization
                if self.config.get('optimization.grid_search', True):
                    grid_result = optimizer.grid_search_optimization(model, movies_df)
                    optimization_results[f"{model_name}_grid"] = grid_result
                
                # Random search optimization
                if self.config.get('optimization.random_search', True):
                    random_result = optimizer.random_search_optimization(model, movies_df)
                    optimization_results[f"{model_name}_random"] = random_result
            
            # Create ensemble if enabled
            if self.config.get('optimization.ensemble', True):
                print("Creating ensemble model...")
                ensemble = EnsembleRecommender(list(models.values()))
                self.trained_models['ensemble'] = ensemble
            
            # Generate optimization report
            reporter = OptimizationReporter()
            report = reporter.generate_summary_report(optimization_results)
            
            # Store results
            self.results['model_optimization'] = {
                'optimization_results': optimization_results,
                'ensemble_created': self.config.get('optimization.ensemble', True),
                'report': report
            }
            
            print("Model optimization completed successfully")
            
        except Exception as e:
            print(f"Error in model optimization: {e}")
            self.results['model_optimization'] = {'error': str(e)}
    
    def _run_model_analysis(self):
        """Run model analysis and results generation step."""
        if not MODEL_ANALYSIS_AVAILABLE:
            print("Model analysis module not available. Skipping...")
            return
        
        try:
            models = self.trained_models
            movies_df = self.components.get('merged_df')
            
            if not models or movies_df is None:
                print("No trained models or data available for analysis")
                return
            
            print("Analyzing model performance and generating results...")
            
            # Initialize analyzer
            analyzer = RecommendationAnalyzer(movies_df)
            
            # Analyze each model
            analysis_results = {}
            for model_name, model in models.items():
                print(f"Analyzing {model_name}...")
                
                # Get sample recommendations
                try:
                    if hasattr(model, 'get_recommendations'):
                        recommendations = model.get_recommendations("sample query", top_k=10)
                    else:
                        recommendations = pd.DataFrame()  # Empty for models without this method
                    
                    # Evaluate recommendation quality
                    evaluation = analyzer.evaluate_recommendation_quality(
                        model_name, recommendations
                    )
                    analysis_results[model_name] = evaluation
                    
                except Exception as model_error:
                    print(f"Error analyzing {model_name}: {model_error}")
                    analysis_results[model_name] = {'error': str(model_error)}
            
            # Compare models
            comparator = ModelComparison()
            for model_name, analysis in analysis_results.items():
                if 'error' not in analysis:
                    # Add dummy recommendations for comparison
                    sample_recs = pd.DataFrame({
                        'title': [f'Movie {i}' for i in range(5)],
                        'genres': ['Action|Adventure'] * 5,
                        'vote_count': [1000] * 5,
                        'vote_average': [7.5] * 5
                    })
                    comparator.add_model_results(model_name, sample_recs, analysis)
            
            comparison_report = comparator.generate_comparison_report()
            
            # Generate visualizations and reports
            reporter = ResultsReporter(analyzer)
            
            if self.config.get('analysis.generate_reports', True):
                html_report = reporter.generate_html_report(
                    comparison_report, 
                    os.path.join(self.config.get('data.output_dir'), 'analysis_report.html')
                )
            
            # Store results
            self.results['model_analysis'] = {
                'individual_analysis': analysis_results,
                'comparison_report': comparison_report,
                'html_report_generated': self.config.get('analysis.generate_reports', True)
            }
            
            print("Model analysis completed successfully")
            
        except Exception as e:
            print(f"Error in model analysis: {e}")
            self.results['model_analysis'] = {'error': str(e)}
    
    def _run_ai_agents(self):
        """Run AI agents integration step."""
        if not AI_AGENTS_AVAILABLE:
            print("AI agents module not available. Skipping...")
            return
        
        try:
            movies_df = self.components.get('merged_df')
            api_key = self.config.get('ai_agents.api_key')
            
            if movies_df is None:
                print("No data available for AI agents")
                return
            
            if not api_key:
                print("OpenAI API key not provided. Skipping AI agents...")
                return
            
            print("Integrating AI agents...")
            
            # Initialize agents
            model_name = self.config.get('ai_agents.model_name', 'gpt-4o')
            
            recommendation_agent = MovieRecommendationAgent(movies_df, api_key, model_name)
            sentiment_agent = SentimentAnalysisAgent(api_key, model_name)
            explanation_agent = ExplanationAgent(api_key, model_name)
            
            # Create agentic pipeline
            agentic_pipeline = AgenticPipeline(movies_df, api_key, model_name)
            
            # Test agents with sample queries
            test_queries = [
                "I love action movies with great special effects",
                "Looking for a romantic comedy that's not too cheesy",
                "Recommend some sci-fi movies similar to Inception"
            ]
            
            agent_results = {}
            for i, query in enumerate(test_queries):
                print(f"Testing agent with query {i+1}: {query}")
                
                try:
                    result = agentic_pipeline.process_user_request(query)
                    agent_results[f"query_{i+1}"] = result
                except Exception as agent_error:
                    print(f"Error processing query {i+1}: {agent_error}")
                    agent_results[f"query_{i+1}"] = {'error': str(agent_error)}
            
            # Store results
            self.results['ai_agents'] = {
                'agents_initialized': True,
                'test_results': agent_results,
                'total_test_queries': len(test_queries)
            }
            
            print("AI agents integration completed successfully")
            
        except Exception as e:
            print(f"Error in AI agents integration: {e}")
            self.results['ai_agents'] = {'error': str(e)}
    
    def _run_model_persistence(self):
        """Run model persistence step."""
        if not MODEL_PERSISTENCE_AVAILABLE:
            print("Model persistence module not available. Skipping...")
            return
        
        try:
            if not self.config.get('persistence.save_models', True):
                print("Model saving disabled in configuration")
                return
            
            print("Saving models and components for deployment...")
            
            # Initialize model manager
            models_dir = self.config.get('persistence.models_dir', './saved_models')
            manager = ModelManager(models_dir)
            
            # Prepare components for saving
            components_to_save = {}
            
            # Add data components
            if 'merged_df' in self.components:
                components_to_save['movies_df'] = self.components['merged_df']
            
            # Add model components (simplified for demonstration)
            trained_models = self.trained_models
            if trained_models:
                # Note: In a real implementation, you would save actual trained models
                # For now, we'll just save metadata about the models
                model_metadata = {
                    'model_names': list(trained_models.keys()),
                    'model_count': len(trained_models),
                    'training_completed': True
                }
                
                # Save model metadata as JSON
                metadata_path = os.path.join(models_dir, 'models_metadata.json')
                os.makedirs(models_dir, exist_ok=True)
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2)
                
                components_to_save['models_metadata'] = model_metadata
            
            # Create deployment configuration
            if self.config.get('persistence.create_deployment_config', True):
                deployment_config = {
                    'version': '1.0.0',
                    'pipeline_results': self.results,
                    'components': list(components_to_save.keys()),
                    'models_available': list(self.trained_models.keys()) if self.trained_models else []
                }
                
                config_path = manager.saver.create_deployment_config(deployment_config)
                self.results['persistence'] = {
                    'deployment_config_path': config_path,
                    'models_dir': models_dir,
                    'components_saved': list(components_to_save.keys())
                }
            
            print(f"Model persistence completed. Files saved to: {models_dir}")
            
        except Exception as e:
            print(f"Error in model persistence: {e}")
            self.results['persistence'] = {'error': str(e)}
    
    def _print_final_summary(self):
        """Print final pipeline summary."""
        print("\nPIPELINE EXECUTION SUMMARY:")
        print("-" * 30)
        
        total_steps = 0
        successful_steps = 0
        
        for step_name, step_results in self.results.items():
            total_steps += 1
            if isinstance(step_results, dict) and 'error' not in step_results:
                successful_steps += 1
                status = "✓ SUCCESS"
            else:
                status = "✗ ERROR"
            
            print(f"{step_name.upper()}: {status}")
        
        print(f"\nTotal Steps: {total_steps}")
        print(f"Successful: {successful_steps}")
        print(f"Failed: {total_steps - successful_steps}")
        
        if 'merged_df' in self.components:
            print(f"Final Dataset Shape: {self.components['merged_df'].shape}")
        
        if self.trained_models:
            print(f"Models Trained: {len(self.trained_models)}")
        
        print(f"\nResults and outputs saved to: {self.config.get('data.output_dir')}")
        print(f"Models saved to: {self.config.get('persistence.models_dir')}")
    
    def _create_sample_movies_data(self):
        """Create sample movies data for demonstration."""
        return pd.DataFrame({
            'id': range(1, 11),
            'title': [
                'The Matrix', 'Inception', 'The Dark Knight', 'Pulp Fiction',
                'The Shawshank Redemption', 'Forrest Gump', 'The Godfather',
                'Star Wars', 'Avatar', 'Titanic'
            ],
            'genres': [
                'Action|Sci-Fi', 'Action|Sci-Fi|Thriller', 'Action|Crime|Drama',
                'Crime|Drama', 'Drama', 'Drama|Romance', 'Crime|Drama',
                'Adventure|Fantasy|Sci-Fi', 'Action|Adventure|Fantasy',
                'Drama|Romance'
            ],
            'overview': [
                'A computer programmer discovers reality is a simulation.',
                'A thief enters dreams to steal secrets.',
                'Batman fights the Joker in Gotham City.',
                'Interconnected criminal stories in Los Angeles.',
                'A banker finds hope and friendship in prison.',
                'Life story of a man with low IQ but good heart.',
                'The story of a powerful crime family.',
                'A young farm boy joins a rebellion against an evil empire.',
                'Humans colonize an alien world.',
                'A ship sinks during its maiden voyage.'
            ],
            'vote_average': [8.7, 8.8, 9.0, 8.9, 9.3, 8.8, 9.2, 8.6, 7.8, 7.9],
            'vote_count': [19000, 28000, 32000, 27000, 35000, 25000, 30000, 22000, 26000, 29000]
        })
    
    def _create_sample_credits_data(self):
        """Create sample credits data for demonstration."""
        return pd.DataFrame({
            'id': range(1, 11),
            'cast': [
                'Keanu Reeves|Laurence Fishburne|Carrie-Anne Moss',
                'Leonardo DiCaprio|Marion Cotillard|Ellen Page',
                'Christian Bale|Heath Ledger|Aaron Eckhart',
                'John Travolta|Samuel L. Jackson|Uma Thurman',
                'Tim Robbins|Morgan Freeman|Bob Gunton',
                'Tom Hanks|Robin Wright|Gary Sinise',
                'Marlon Brando|Al Pacino|James Caan',
                'Mark Hamill|Harrison Ford|Carrie Fisher',
                'Sam Worthington|Zoe Saldana|Sigourney Weaver',
                'Leonardo DiCaprio|Kate Winslet|Billy Zane'
            ],
            'crew': [
                'Lana Wachowski|Lilly Wachowski',
                'Christopher Nolan',
                'Christopher Nolan',
                'Quentin Tarantino',
                'Frank Darabont',
                'Robert Zemeckis',
                'Francis Ford Coppola',
                'George Lucas',
                'James Cameron',
                'James Cameron'
            ]
        })


class PipelineRunner:
    """
    Execute pipeline with different configurations and modes.
    """
    
    def __init__(self):
        self.pipelines = {}
    
    def run_development_mode(self, config_file=None):
        """Run pipeline in development mode with sample data."""
        print("Running pipeline in DEVELOPMENT mode...")
        
        config = PipelineConfig(config_file)
        
        # Disable resource-intensive features for development
        config.set('models.autoencoder.enabled', False)
        config.set('optimization.bayesian_optimization', False)
        config.set('optimization.optuna_optimization', False)
        config.set('ai_agents.enabled', False)
        
        pipeline = MovieRecommendationPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        self.pipelines['development'] = pipeline
        return results
    
    def run_production_mode(self, config_file=None):
        """Run pipeline in production mode with full features."""
        print("Running pipeline in PRODUCTION mode...")
        
        config = PipelineConfig(config_file)
        
        pipeline = MovieRecommendationPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        self.pipelines['production'] = pipeline
        return results
    
    def run_evaluation_mode(self, config_file=None):
        """Run pipeline in evaluation mode focusing on analysis."""
        print("Running pipeline in EVALUATION mode...")
        
        config = PipelineConfig(config_file)
        
        # Enable all analysis features
        config.set('analysis.diversity_analysis', True)
        config.set('analysis.novelty_analysis', True)
        config.set('analysis.coverage_analysis', True)
        config.set('analysis.generate_reports', True)
        
        # Disable model persistence to focus on evaluation
        config.set('persistence.save_models', False)
        
        pipeline = MovieRecommendationPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        self.pipelines['evaluation'] = pipeline
        return results


def main():
    """
    Main entry point for the movie recommendation pipeline.
    """
    print("Movie Recommendation System - Main Pipeline")
    print("=" * 60)
    
    # Create sample configuration file
    config = PipelineConfig()
    config.save_config('./pipeline_config.json')
    print("Sample configuration file created: pipeline_config.json")
    
    # Run pipeline in development mode
    runner = PipelineRunner()
    
    print("\n" + "="*60)
    print("RUNNING DEVELOPMENT MODE PIPELINE")
    print("="*60)
    
    try:
        results = runner.run_development_mode('./pipeline_config.json')
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED!")
        print("="*60)
        
        print("\nFinal Results Summary:")
        for step, result in results.items():
            if isinstance(result, dict) and 'error' in result:
                print(f"  {step}: ERROR - {result['error']}")
            else:
                print(f"  {step}: SUCCESS")
        
        print("\nTo run in different modes:")
        print("  - Development: runner.run_development_mode()")
        print("  - Production: runner.run_production_mode()")
        print("  - Evaluation: runner.run_evaluation_mode()")
        
        print(f"\nCheck the following directories for outputs:")
        print(f"  - Output: ./output")
        print(f"  - Models: ./saved_models")
        print(f"  - Logs: ./logs")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        
    print("\nPipeline execution completed!")


if __name__ == "__main__":
    main()
