"""
Model Analysis and Results Module

This module provides comprehensive analysis and evaluation capabilities for movie recommendation models.
It includes quantitative metrics calculation, qualitative analysis, results reporting, and visualization tools.

Classes:
    - RecommendationAnalyzer: Main analyzer for recommendation models
    - ModelComparison: Comparison tools for multiple models
    - ResultsReporter: Report generation and visualization

Dependencies:
    - pandas, numpy, matplotlib, seaborn, plotly
    - sklearn.metrics
    - scipy.stats
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')


class RecommendationAnalyzer:
    """
    Comprehensive analyzer for recommendation model performance and results.
    """
    
    def __init__(self, movies_df=None):
        """
        Initialize the analyzer with movie data.
        
        Parameters:
        - movies_df (pd.DataFrame): Movie dataset with metadata
        """
        self.movies_df = movies_df
        self.analysis_results = {}
        self.recommendations_cache = {}
        
    def calculate_diversity_metrics(self, recommendations_df, genre_column='genres'):
        """
        Calculate diversity metrics for recommendations.
        
        Parameters:
        - recommendations_df (pd.DataFrame): DataFrame with recommendations
        - genre_column (str): Column name containing genre information
        
        Returns:
        - dict: Dictionary with diversity metrics
        """
        try:
            if genre_column not in recommendations_df.columns:
                return {"error": f"Column '{genre_column}' not found"}
            
            # Extract unique genres from recommendations
            all_genres = []
            for genres_str in recommendations_df[genre_column].dropna():
                if isinstance(genres_str, str):
                    # Assuming genres are stored as strings like "Action|Adventure|Sci-Fi"
                    genres = [g.strip() for g in genres_str.replace('[', '').replace(']', '').replace("'", "").split(',')]
                    all_genres.extend(genres)
            
            unique_genres = set(all_genres)
            total_recommendations = len(recommendations_df)
            
            # Calculate diversity metrics
            genre_diversity = len(unique_genres) / total_recommendations if total_recommendations > 0 else 0
            
            # Calculate genre distribution
            genre_counts = pd.Series(all_genres).value_counts()
            genre_entropy = -sum((count/len(all_genres)) * np.log2(count/len(all_genres)) 
                                for count in genre_counts if count > 0)
            
            return {
                "unique_genres_count": len(unique_genres),
                "genre_diversity_ratio": genre_diversity,
                "genre_entropy": genre_entropy,
                "most_common_genre": genre_counts.index[0] if len(genre_counts) > 0 else None,
                "genre_distribution": genre_counts.to_dict()
            }
            
        except Exception as e:
            return {"error": f"Error calculating diversity metrics: {str(e)}"}
    
    def calculate_novelty_metrics(self, recommendations_df, popularity_column='vote_count'):
        """
        Calculate novelty metrics for recommendations.
        
        Parameters:
        - recommendations_df (pd.DataFrame): DataFrame with recommendations
        - popularity_column (str): Column name for popularity measure
        
        Returns:
        - dict: Dictionary with novelty metrics
        """
        try:
            if popularity_column not in recommendations_df.columns:
                return {"error": f"Column '{popularity_column}' not found"}
            
            popularity_scores = recommendations_df[popularity_column].dropna()
            
            if len(popularity_scores) == 0:
                return {"error": "No valid popularity scores found"}
            
            # Calculate novelty as inverse of popularity
            novelty_scores = 1 / (1 + popularity_scores)
            
            return {
                "average_novelty": novelty_scores.mean(),
                "novelty_std": novelty_scores.std(),
                "min_popularity": popularity_scores.min(),
                "max_popularity": popularity_scores.max(),
                "popularity_median": popularity_scores.median()
            }
            
        except Exception as e:
            return {"error": f"Error calculating novelty metrics: {str(e)}"}
    
    def calculate_coverage_metrics(self, all_recommendations, total_items=None):
        """
        Calculate catalog coverage metrics.
        
        Parameters:
        - all_recommendations (list): List of all recommended item IDs
        - total_items (int): Total number of items in catalog
        
        Returns:
        - dict: Dictionary with coverage metrics
        """
        try:
            unique_recommendations = set(all_recommendations)
            
            if total_items is None:
                total_items = len(self.movies_df) if self.movies_df is not None else len(unique_recommendations)
            
            catalog_coverage = len(unique_recommendations) / total_items if total_items > 0 else 0
            
            # Calculate recommendation frequency distribution
            recommendation_counts = pd.Series(all_recommendations).value_counts()
            
            return {
                "catalog_coverage": catalog_coverage,
                "unique_items_recommended": len(unique_recommendations),
                "total_catalog_items": total_items,
                "recommendation_distribution_gini": self._calculate_gini_coefficient(recommendation_counts.values)
            }
            
        except Exception as e:
            return {"error": f"Error calculating coverage metrics: {str(e)}"}
    
    def _calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for measuring inequality in distribution."""
        try:
            values = np.array(values)
            values = np.sort(values)
            n = len(values)
            cumulative = np.cumsum(values)
            return (2 * np.sum((np.arange(1, n + 1) * values))) / (n * cumulative[-1]) - (n + 1) / n
        except:
            return 0.0
    
    def evaluate_recommendation_quality(self, model_name, recommendations_df, user_preferences=None):
        """
        Comprehensive evaluation of recommendation quality.
        
        Parameters:
        - model_name (str): Name of the recommendation model
        - recommendations_df (pd.DataFrame): DataFrame with recommendations
        - user_preferences (dict): User preference information
        
        Returns:
        - dict: Comprehensive evaluation results
        """
        try:
            evaluation_results = {
                "model_name": model_name,
                "total_recommendations": len(recommendations_df),
                "timestamp": pd.Timestamp.now()
            }
            
            # Calculate diversity metrics
            diversity_metrics = self.calculate_diversity_metrics(recommendations_df)
            evaluation_results["diversity"] = diversity_metrics
            
            # Calculate novelty metrics
            novelty_metrics = self.calculate_novelty_metrics(recommendations_df)
            evaluation_results["novelty"] = novelty_metrics
            
            # Calculate basic statistics
            if 'vote_average' in recommendations_df.columns:
                vote_scores = recommendations_df['vote_average'].dropna()
                evaluation_results["quality_stats"] = {
                    "average_rating": vote_scores.mean(),
                    "rating_std": vote_scores.std(),
                    "min_rating": vote_scores.min(),
                    "max_rating": vote_scores.max()
                }
            
            # Store results
            self.analysis_results[model_name] = evaluation_results
            
            return evaluation_results
            
        except Exception as e:
            return {"error": f"Error in recommendation evaluation: {str(e)}"}


class ModelComparison:
    """
    Tools for comparing multiple recommendation models.
    """
    
    def __init__(self):
        self.comparison_results = {}
        self.models_data = {}
    
    def add_model_results(self, model_name, recommendations_df, metrics_dict=None):
        """
        Add model results for comparison.
        
        Parameters:
        - model_name (str): Name of the model
        - recommendations_df (pd.DataFrame): Recommendations from the model
        - metrics_dict (dict): Additional metrics for the model
        """
        self.models_data[model_name] = {
            "recommendations": recommendations_df,
            "metrics": metrics_dict or {}
        }
    
    def compare_diversity(self):
        """
        Compare diversity across models.
        
        Returns:
        - pd.DataFrame: Comparison results
        """
        try:
            analyzer = RecommendationAnalyzer()
            diversity_results = []
            
            for model_name, data in self.models_data.items():
                recommendations_df = data["recommendations"]
                diversity_metrics = analyzer.calculate_diversity_metrics(recommendations_df)
                
                if "error" not in diversity_metrics:
                    diversity_results.append({
                        "model": model_name,
                        "unique_genres": diversity_metrics.get("unique_genres_count", 0),
                        "diversity_ratio": diversity_metrics.get("genre_diversity_ratio", 0),
                        "genre_entropy": diversity_metrics.get("genre_entropy", 0)
                    })
            
            return pd.DataFrame(diversity_results)
            
        except Exception as e:
            print(f"Error in diversity comparison: {e}")
            return pd.DataFrame()
    
    def compare_novelty(self):
        """
        Compare novelty across models.
        
        Returns:
        - pd.DataFrame: Comparison results
        """
        try:
            analyzer = RecommendationAnalyzer()
            novelty_results = []
            
            for model_name, data in self.models_data.items():
                recommendations_df = data["recommendations"]
                novelty_metrics = analyzer.calculate_novelty_metrics(recommendations_df)
                
                if "error" not in novelty_metrics:
                    novelty_results.append({
                        "model": model_name,
                        "average_novelty": novelty_metrics.get("average_novelty", 0),
                        "popularity_median": novelty_metrics.get("popularity_median", 0),
                        "min_popularity": novelty_metrics.get("min_popularity", 0)
                    })
            
            return pd.DataFrame(novelty_results)
            
        except Exception as e:
            print(f"Error in novelty comparison: {e}")
            return pd.DataFrame()
    
    def generate_comparison_report(self):
        """
        Generate comprehensive comparison report.
        
        Returns:
        - dict: Comparison report
        """
        try:
            diversity_comparison = self.compare_diversity()
            novelty_comparison = self.compare_novelty()
            
            report = {
                "models_compared": list(self.models_data.keys()),
                "diversity_comparison": diversity_comparison.to_dict('records') if not diversity_comparison.empty else [],
                "novelty_comparison": novelty_comparison.to_dict('records') if not novelty_comparison.empty else [],
                "summary": {}
            }
            
            # Add summary statistics
            if not diversity_comparison.empty:
                report["summary"]["best_diversity"] = diversity_comparison.loc[diversity_comparison['diversity_ratio'].idxmax(), 'model']
                report["summary"]["best_genre_entropy"] = diversity_comparison.loc[diversity_comparison['genre_entropy'].idxmax(), 'model']
            
            if not novelty_comparison.empty:
                report["summary"]["most_novel"] = novelty_comparison.loc[novelty_comparison['average_novelty'].idxmax(), 'model']
            
            return report
            
        except Exception as e:
            return {"error": f"Error generating comparison report: {str(e)}"}


class ResultsReporter:
    """
    Generate reports and visualizations for recommendation analysis.
    """
    
    def __init__(self, analyzer=None):
        self.analyzer = analyzer or RecommendationAnalyzer()
    
    def create_diversity_plot(self, comparison_df):
        """
        Create visualization for diversity comparison.
        
        Parameters:
        - comparison_df (pd.DataFrame): Diversity comparison data
        
        Returns:
        - matplotlib figure or plotly figure
        """
        try:
            if comparison_df.empty:
                print("No data available for diversity plot")
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Diversity ratio plot
            if SEABORN_AVAILABLE:
                sns.barplot(data=comparison_df, x='model', y='diversity_ratio', ax=ax1)
            else:
                ax1.bar(comparison_df['model'], comparison_df['diversity_ratio'])
            ax1.set_title('Genre Diversity Ratio by Model')
            ax1.set_ylabel('Diversity Ratio')
            ax1.tick_params(axis='x', rotation=45)
            
            # Genre entropy plot
            if SEABORN_AVAILABLE:
                sns.barplot(data=comparison_df, x='model', y='genre_entropy', ax=ax2)
            else:
                ax2.bar(comparison_df['model'], comparison_df['genre_entropy'])
            ax2.set_title('Genre Entropy by Model')
            ax2.set_ylabel('Entropy')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating diversity plot: {e}")
            return None
    
    def create_novelty_plot(self, comparison_df):
        """
        Create visualization for novelty comparison.
        
        Parameters:
        - comparison_df (pd.DataFrame): Novelty comparison data
        
        Returns:
        - matplotlib figure or plotly figure
        """
        try:
            if comparison_df.empty:
                print("No data available for novelty plot")
                return None
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            if SEABORN_AVAILABLE:
                sns.barplot(data=comparison_df, x='model', y='average_novelty', ax=ax)
            else:
                ax.bar(comparison_df['model'], comparison_df['average_novelty'])
                
            ax.set_title('Average Novelty by Model')
            ax.set_ylabel('Average Novelty')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating novelty plot: {e}")
            return None
    
    def generate_html_report(self, comparison_results, output_path="recommendation_analysis_report.html"):
        """
        Generate HTML report with analysis results.
        
        Parameters:
        - comparison_results (dict): Comparison results from ModelComparison
        - output_path (str): Path to save the HTML report
        
        Returns:
        - str: HTML report content
        """
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Movie Recommendation Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                    .section {{ margin: 30px 0; }}
                    .metric {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007acc; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 8px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Movie Recommendation System Analysis Report</h1>
                    <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Models Analyzed</h2>
                    <ul>
            """
            
            for model in comparison_results.get("models_compared", []):
                html_content += f"<li>{model}</li>"
            
            html_content += """
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Diversity Analysis</h2>
                    <div class="metric">
            """
            
            diversity_data = comparison_results.get("diversity_comparison", [])
            if diversity_data:
                html_content += """
                        <table>
                            <tr>
                                <th>Model</th>
                                <th>Unique Genres</th>
                                <th>Diversity Ratio</th>
                                <th>Genre Entropy</th>
                            </tr>
                """
                
                for row in diversity_data:
                    html_content += f"""
                            <tr>
                                <td>{row.get('model', 'N/A')}</td>
                                <td>{row.get('unique_genres', 0)}</td>
                                <td>{row.get('diversity_ratio', 0):.4f}</td>
                                <td>{row.get('genre_entropy', 0):.4f}</td>
                            </tr>
                    """
                
                html_content += "</table>"
            else:
                html_content += "<p>No diversity data available</p>"
            
            html_content += """
                    </div>
                </div>
                
                <div class="section">
                    <h2>Novelty Analysis</h2>
                    <div class="metric">
            """
            
            novelty_data = comparison_results.get("novelty_comparison", [])
            if novelty_data:
                html_content += """
                        <table>
                            <tr>
                                <th>Model</th>
                                <th>Average Novelty</th>
                                <th>Median Popularity</th>
                                <th>Min Popularity</th>
                            </tr>
                """
                
                for row in novelty_data:
                    html_content += f"""
                            <tr>
                                <td>{row.get('model', 'N/A')}</td>
                                <td>{row.get('average_novelty', 0):.4f}</td>
                                <td>{row.get('popularity_median', 0)}</td>
                                <td>{row.get('min_popularity', 0)}</td>
                            </tr>
                    """
                
                html_content += "</table>"
            else:
                html_content += "<p>No novelty data available</p>"
            
            html_content += """
                    </div>
                </div>
                
                <div class="section summary">
                    <h2>Summary</h2>
            """
            
            summary = comparison_results.get("summary", {})
            if summary:
                for key, value in summary.items():
                    html_content += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
            else:
                html_content += "<p>No summary data available</p>"
            
            html_content += """
                </div>
            </body>
            </html>
            """
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"HTML report saved to: {output_path}")
            
            return html_content
            
        except Exception as e:
            error_msg = f"Error generating HTML report: {str(e)}"
            print(error_msg)
            return f"<html><body><h1>Error</h1><p>{error_msg}</p></body></html>"


def main():
    """
    Example usage of the model analysis module.
    """
    print("Model Analysis and Results Module")
    print("=" * 50)
    
    # Example data creation
    sample_recommendations = pd.DataFrame({
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'genres': ['Action|Adventure', 'Comedy|Romance', 'Horror|Thriller', 'Sci-Fi|Action', 'Drama|Romance'],
        'vote_count': [1000, 500, 2000, 300, 800],
        'vote_average': [7.5, 6.8, 8.2, 7.1, 6.9]
    })
    
    # Initialize analyzer
    analyzer = RecommendationAnalyzer()
    
    # Analyze recommendations
    print("\n1. Analyzing recommendation quality...")
    evaluation = analyzer.evaluate_recommendation_quality(
        model_name="sample_model",
        recommendations_df=sample_recommendations
    )
    
    print(f"Evaluation results: {evaluation}")
    
    # Initialize comparison
    print("\n2. Comparing multiple models...")
    comparator = ModelComparison()
    
    # Add sample models
    comparator.add_model_results("content_based", sample_recommendations)
    comparator.add_model_results("collaborative", sample_recommendations.sample(3))
    
    # Generate comparison
    comparison_report = comparator.generate_comparison_report()
    print(f"Comparison report: {comparison_report}")
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    reporter = ResultsReporter(analyzer)
    
    diversity_comparison = comparator.compare_diversity()
    if not diversity_comparison.empty:
        fig = reporter.create_diversity_plot(diversity_comparison)
        if fig:
            plt.show()
    
    # Generate HTML report
    print("\n4. Generating HTML report...")
    html_report = reporter.generate_html_report(comparison_report)
    print("HTML report generated successfully!")
    
    print("\nModel analysis example completed!")


if __name__ == "__main__":
    main()
