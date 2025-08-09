# Movie Recommendation System - Extracted Modules

This project contains modularized Python code extracted from the Jupyter notebook `AAI_Final_Project__Movie_Recommendations_And_Sentiment_Analysis.ipynb`. The code has been refactored into individual modules for better organization, reusability, and deployment.

## 📁 Project Structure

```
/Users/arupchakraborty/Documents/AAI-590/FinalProject/
├── notebook/
│   └── AAI_Final_Project__Movie_Recommendations_And_Sentiment_Analysis.ipynb
├── src/
│   ├── data_cleaning.py              # Data preprocessing and cleaning
│   ├── exploratory_data_analysis.py  # EDA, visualizations, and profiling
│   ├── model_design.py              # Recommendation model classes
│   ├── model_training.py            # Model training and evaluation
│   ├── model_optimization.py        # Advanced optimization techniques
│   ├── model_analysis.py            # Model analysis and results reporting
│   ├── ai_agents.py                 # LangChain/OpenAI agentic workflows
│   ├── model_persistence.py         # Model saving and loading utilities
│   ├── main_pipeline.py             # Complete pipeline integration
│   └── README.md                    # This file
├── data/                            # (Optional) Data files directory
├── output/                          # Pipeline output directory
├── saved_models/                    # Saved model artifacts
└── logs/                           # Pipeline execution logs
```

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
# Core dependencies (required)
pip install pandas numpy matplotlib scikit-learn scipy

# Optional dependencies for full functionality
pip install seaborn plotly tensorflow sentence-transformers
pip install langchain langchain-openai openai
pip install optuna scikit-optimize ydata-profiling
pip install joblib
```

### Basic Usage

1. **Run Individual Modules:**

```python
# Data Cleaning
from src.data_cleaning import MovieDataCleaner, DataMerger

cleaner = MovieDataCleaner()
cleaned_data = cleaner.clean_movies_data(movies_df)

# Exploratory Data Analysis
from src.exploratory_data_analysis import MovieEDA, EDAVisualizer

eda = MovieEDA(movies_df)
stats = eda.generate_basic_statistics()

# Model Design and Training
from src.model_design import ContentBasedRecommender
from src.model_training import ModelTrainer

model = ContentBasedRecommender(movies_df)
trainer = ModelTrainer({'content_model': model}, movies_df)
trainer.train_all_models()
```

2. **Run Complete Pipeline:**

```python
from src.main_pipeline import PipelineRunner

runner = PipelineRunner()
results = runner.run_development_mode()
```

## 📋 Module Documentation

### 1. Data Cleaning (`data_cleaning.py`)

**Classes:**
- `MovieDataCleaner`: Handles data preprocessing and cleaning
- `DataMerger`: Merges multiple datasets
- `FeatureEngineer`: Creates new features and transformations

**Key Features:**
- Missing value handling
- Data type conversions
- Feature engineering (e.g., 'soup' creation for content-based filtering)
- Data validation and quality checks

**Usage:**
```python
from src.data_cleaning import MovieDataCleaner

cleaner = MovieDataCleaner()
cleaned_movies = cleaner.clean_movies_data(movies_df)
```

### 2. Exploratory Data Analysis (`exploratory_data_analysis.py`)

**Classes:**
- `MovieEDA`: Statistical analysis and insights
- `MovieProfiler`: Automated profiling reports
- `EDAVisualizer`: Visualization creation

**Key Features:**
- Comprehensive statistical analysis
- Genre and rating distribution analysis
- Automated profiling reports
- Interactive visualizations

**Usage:**
```python
from src.exploratory_data_analysis import MovieEDA, EDAVisualizer

eda = MovieEDA(movies_df)
basic_stats = eda.generate_basic_statistics()

visualizer = EDAVisualizer()
visualizer.create_genre_distribution_plot(movies_df)
```

### 3. Model Design (`model_design.py`)

**Classes:**
- `ContentBasedRecommender`: TF-IDF and cosine similarity
- `KNNRecommender`: K-Nearest Neighbors approach
- `AutoencoderRecommender`: Deep learning with autoencoders
- `ClusteringRecommender`: K-Means clustering approach
- `SentimentBasedRecommender`: Sentiment-aware recommendations
- `HybridRecommender`: Combines multiple approaches

**Key Features:**
- Multiple recommendation algorithms
- Modular and extensible design
- Common interface for all recommenders
- Configurable parameters

**Usage:**
```python
from src.model_design import ContentBasedRecommender, HybridRecommender

# Content-based model
content_model = ContentBasedRecommender(movies_df)
recommendations = content_model.get_recommendations("action adventure", top_k=10)

# Hybrid model
models = [content_model]  # Add more models
hybrid_model = HybridRecommender(models)
```

### 4. Model Training (`model_training.py`)

**Classes:**
- `ModelTrainer`: Coordinates model training
- `HyperparameterTuner`: Parameter optimization
- `ModelEvaluator`: Performance evaluation

**Key Features:**
- Automated training pipelines
- Cross-validation support
- Hyperparameter tuning
- Performance metrics calculation

**Usage:**
```python
from src.model_training import ModelTrainer, HyperparameterTuner

trainer = ModelTrainer(models_dict, movies_df)
results = trainer.train_all_models()

tuner = HyperparameterTuner()
best_params = tuner.tune_content_based_model(content_model, movies_df)
```

### 5. Model Optimization (`model_optimization.py`)

**Classes:**
- `AdvancedOptimizer`: Grid search, random search, Bayesian optimization
- `EnsembleRecommender`: Model ensemble techniques
- `OptimizationReporter`: Optimization results reporting

**Key Features:**
- Multiple optimization strategies
- Ensemble methods
- Detailed reporting and analysis
- Performance comparison

**Usage:**
```python
from src.model_optimization import AdvancedOptimizer, EnsembleRecommender

optimizer = AdvancedOptimizer()
grid_results = optimizer.grid_search_optimization(model, movies_df)

ensemble = EnsembleRecommender(models_list)
ensemble_recommendations = ensemble.get_recommendations("query")
```

### 6. Model Analysis (`model_analysis.py`)

**Classes:**
- `RecommendationAnalyzer`: Comprehensive analysis metrics
- `ModelComparison`: Multi-model comparison
- `ResultsReporter`: Report generation and visualization

**Key Features:**
- Diversity and novelty metrics
- Coverage analysis
- Model comparison tools
- HTML report generation

**Usage:**
```python
from src.model_analysis import RecommendationAnalyzer, ModelComparison

analyzer = RecommendationAnalyzer(movies_df)
evaluation = analyzer.evaluate_recommendation_quality("model_name", recommendations_df)

comparator = ModelComparison()
comparator.add_model_results("model1", recommendations1)
report = comparator.generate_comparison_report()
```

### 7. AI Agents (`ai_agents.py`)

**Classes:**
- `MovieRecommendationAgent`: LangChain-powered recommendation agent
- `SentimentAnalysisAgent`: Sentiment analysis for preferences
- `ExplanationAgent`: Generate explanations for recommendations
- `AgenticPipeline`: Complete agentic workflow

**Key Features:**
- LangChain integration
- OpenAI API integration
- Natural language processing
- Agentic recommendation workflows

**Requirements:**
- OpenAI API key
- LangChain packages

**Usage:**
```python
from src.ai_agents import AgenticPipeline

# Requires OpenAI API key
api_key = "your-openai-api-key"
pipeline = AgenticPipeline(movies_df, api_key)
result = pipeline.process_user_request("I want exciting sci-fi movies")
```

### 8. Model Persistence (`model_persistence.py`)

**Classes:**
- `ModelSaver`: Save trained models and components
- `ModelLoader`: Load saved models and components
- `ModelManager`: Manage model lifecycle

**Key Features:**
- Support for multiple model types
- Metadata management
- Deployment configuration
- Version control

**Usage:**
```python
from src.model_persistence import ModelManager

manager = ModelManager("./saved_models")
components = {
    'tfidf_vectorizer': tfidf,
    'movies_df': movies_df
}
manager.save_complete_system(components)
```

### 9. Main Pipeline (`main_pipeline.py`)

**Classes:**
- `MovieRecommendationPipeline`: Complete pipeline orchestrator
- `PipelineConfig`: Configuration management
- `PipelineRunner`: Execute different pipeline modes

**Key Features:**
- End-to-end pipeline execution
- Configuration management
- Multiple execution modes (development, production, evaluation)
- Comprehensive logging and reporting

**Usage:**
```python
from src.main_pipeline import PipelineRunner

runner = PipelineRunner()

# Development mode (faster, limited features)
dev_results = runner.run_development_mode()

# Production mode (full features)
prod_results = runner.run_production_mode()

# Evaluation mode (focus on analysis)
eval_results = runner.run_evaluation_mode()
```

## ⚙️ Configuration

The pipeline uses a JSON configuration file for customization:

```json
{
  "data": {
    "movies_file": "data/movies.csv",
    "credits_file": "data/credits.csv",
    "output_dir": "./output",
    "clean_data": true
  },
  "models": {
    "content_based": {"enabled": true, "tfidf_params": {"max_features": 5000}},
    "knn": {"enabled": true, "n_neighbors": 10},
    "autoencoder": {"enabled": true, "latent_dim": 100, "epochs": 50},
    "clustering": {"enabled": true, "n_clusters": 20},
    "sentiment": {"enabled": true},
    "hybrid": {"enabled": true, "weights": {"content": 0.4, "collaborative": 0.3, "sentiment": 0.3}}
  },
  "training": {
    "test_size": 0.2,
    "random_state": 42,
    "cross_validation": true,
    "cv_folds": 5
  },
  "optimization": {
    "grid_search": true,
    "random_search": true,
    "bayesian_optimization": false,
    "optuna_optimization": false,
    "ensemble": true
  },
  "analysis": {
    "diversity_analysis": true,
    "novelty_analysis": true,
    "coverage_analysis": true,
    "generate_reports": true
  },
  "ai_agents": {
    "enabled": false,
    "api_key": null,
    "model_name": "gpt-4o",
    "sentiment_analysis": true,
    "explanation_generation": true
  },
  "persistence": {
    "save_models": true,
    "models_dir": "./saved_models",
    "save_results": true,
    "create_deployment_config": true
  }
}
```

## 🔧 Dependencies

### Required (Core Functionality)
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
scipy>=1.9.0
joblib>=1.2.0
```

### Optional (Extended Functionality)
```
seaborn>=0.11.0           # Enhanced visualizations
plotly>=5.10.0            # Interactive visualizations
tensorflow>=2.8.0         # Autoencoder models
sentence-transformers>=2.2.0  # Text embeddings
ydata-profiling>=4.0.0    # Automated profiling
optuna>=3.0.0             # Advanced optimization
scikit-optimize>=0.9.0    # Bayesian optimization
langchain>=0.1.0          # AI agents
langchain-openai>=0.1.0   # OpenAI integration
openai>=1.0.0             # OpenAI API
```

## 🚨 Known Issues and Limitations

1. **Missing Dependencies**: Some modules have optional dependencies that may not be installed. The code includes fallback mechanisms for most cases.

2. **OpenAI API**: AI agents functionality requires a valid OpenAI API key. Set the `OPENAI_API_KEY` environment variable or pass it as a parameter.

3. **Large Models**: Autoencoder and deep learning models may require significant computational resources and time to train.

4. **Data Requirements**: The system expects movie data in a specific format. Sample data generation is included for testing.

5. **Path Dependencies**: Some import statements may need adjustment depending on your Python path configuration.

## 🔄 Migration from Notebook

To use these modules with your existing notebook data:

1. **Extract Data Variables**: Copy your processed DataFrames from the notebook
2. **Initialize Models**: Create model instances with your data
3. **Run Pipeline**: Use the main pipeline or individual modules
4. **Save Results**: Use model persistence to save trained models

Example migration:
```python
# From notebook
# merged_df = ... (your processed data)
# cosine_sim = ... (your similarity matrix)

# To modular code
from src.model_design import ContentBasedRecommender
from src.model_persistence import ModelSaver

# Create recommender with your data
recommender = ContentBasedRecommender(merged_df)

# Save for deployment
saver = ModelSaver("./my_models")
saver.save_movie_data(merged_df)
saver.save_cosine_similarity_matrix(cosine_sim)
```

## 📊 Output and Results

The pipeline generates various outputs:

- **Data Processing Results**: Cleaned datasets, feature engineering outputs
- **Model Artifacts**: Trained models, similarity matrices, vectorizers
- **Analysis Reports**: HTML reports, visualizations, performance metrics
- **Deployment Configs**: Configuration files for production deployment
- **Logs**: Detailed execution logs and error reports

## 🤝 Contributing

To extend or modify the system:

1. **Add New Models**: Extend the base recommender classes in `model_design.py`
2. **Custom Analysis**: Add new metrics in `model_analysis.py`
3. **New Agents**: Create additional agents in `ai_agents.py`
4. **Pipeline Steps**: Extend the main pipeline with new processing steps

## 📝 License

This project follows the same license as the original notebook. Please refer to the notebook's license information.

## 📞 Support

For issues or questions:

1. Check the individual module documentation
2. Review the example usage in each file's `main()` function
3. Examine the comprehensive pipeline in `main_pipeline.py`
4. Refer to the original notebook for algorithm details

---

*This modular extraction was created to improve code organization, reusability, and deployment readiness while preserving all the functionality from the original Jupyter notebook.*
