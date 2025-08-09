"""
Exploratory Data Analysis Module for Movie Recommendation System
===============================================================

This module provides comprehensive exploratory data analysis functionality
for movie datasets, including visualizations, statistical analysis, and
data profiling capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MovieEDA:
    """
    A class to perform exploratory data analysis on movie datasets.
    """
    
    def __init__(self, data=None):
        """
        Initialize the EDA class.
        
        Args:
            data (pd.DataFrame, optional): Movie dataset to analyze
        """
        self.data = data
        self.numerical_cols = None
        self.categorical_cols = None
        self.text_cols = None
        
        if data is not None:
            self._identify_column_types()
    
    def load_data(self, data_path=None, data=None):
        """
        Load data from file or accept DataFrame directly.
        
        Args:
            data_path (str, optional): Path to CSV file
            data (pd.DataFrame, optional): DataFrame to analyze
        """
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or data must be provided")
        
        self._identify_column_types()
        print(f"Data loaded successfully. Shape: {self.data.shape}")
    
    def _identify_column_types(self):
        """
        Identify different types of columns in the dataset.
        """
        if self.data is None:
            return
        
        # Numerical columns
        self.numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Categorical columns (excluding text columns)
        text_indicators = ['overview', 'soup', 'enhanced_soup', 'title', 'tagline']
        self.categorical_cols = []
        self.text_cols = []
        
        for col in self.data.select_dtypes(include=['object']).columns:
            if any(indicator in col.lower() for indicator in text_indicators) or \
               (col in self.data.columns and self.data[col].str.len().mean() > 50):
                self.text_cols.append(col)
            else:
                self.categorical_cols.append(col)
    
    def basic_info(self):
        """
        Display basic information about the dataset.
        """
        if self.data is None:
            print("No data loaded.")
            return
        
        print("="*60)
        print("BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\nColumn Types:")
        print(f"  Numerical columns ({len(self.numerical_cols)}): {self.numerical_cols}")
        print(f"  Categorical columns ({len(self.categorical_cols)}): {self.categorical_cols}")
        print(f"  Text columns ({len(self.text_cols)}): {self.text_cols}")
        
        print(f"\nData Types:")
        print(self.data.dtypes.value_counts())
        
        print(f"\nMissing Values:")
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing Percentage': missing_pct
        }).sort_values('Missing Count', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        
        print(f"\nDuplicate Rows: {self.data.duplicated().sum()}")
    
    def statistical_summary(self):
        """
        Generate comprehensive statistical summary.
        """
        if self.data is None or not self.numerical_cols:
            print("No numerical data to analyze.")
            return
        
        print("="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        
        # Basic descriptive statistics
        desc_stats = self.data[self.numerical_cols].describe()
        print("\nDescriptive Statistics:")
        print(desc_stats)
        
        # Additional statistics
        print("\nAdditional Statistics:")
        additional_stats = pd.DataFrame({
            'Skewness': self.data[self.numerical_cols].skew(),
            'Kurtosis': self.data[self.numerical_cols].kurtosis(),
            'Variance': self.data[self.numerical_cols].var()
        })
        print(additional_stats)
        
        # Outlier detection using IQR method
        print("\nOutlier Analysis (IQR Method):")
        for col in self.numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(self.data)*100:.1f}%)")
    
    def plot_distributions(self, figsize=(15, 10)):
        """
        Plot distributions of numerical variables.
        
        Args:
            figsize (tuple): Figure size for plots
        """
        if self.data is None or not self.numerical_cols:
            print("No numerical data to plot.")
            return
        
        n_cols = len(self.numerical_cols)
        n_rows = (n_cols + 2) // 3  # 3 plots per row
        
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(self.numerical_cols):
            if i < len(axes):
                # Histogram with KDE
                self.data[col].hist(bins=30, alpha=0.7, ax=axes[i], density=True)
                self.data[col].plot.kde(ax=axes[i], secondary_y=False)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Density')
        
        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    def plot_boxplots(self, figsize=(15, 8)):
        """
        Create box plots for numerical variables to identify outliers.
        
        Args:
            figsize (tuple): Figure size for plots
        """
        if self.data is None or not self.numerical_cols:
            print("No numerical data to plot.")
            return
        
        fig, axes = plt.subplots(1, min(3, len(self.numerical_cols)), figsize=figsize)
        if len(self.numerical_cols) == 1:
            axes = [axes]
        
        key_cols = ['budget', 'revenue', 'runtime'] if all(col in self.numerical_cols for col in ['budget', 'revenue', 'runtime']) else self.numerical_cols[:3]
        
        for i, col in enumerate(key_cols):
            if i < len(axes):
                sns.boxplot(y=self.data[col], ax=axes[i])
                axes[i].set_title(f'Box plot of {col}')
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self, figsize=(12, 10)):
        """
        Perform and visualize correlation analysis.
        
        Args:
            figsize (tuple): Figure size for correlation heatmap
        """
        if self.data is None or not self.numerical_cols:
            print("No numerical data for correlation analysis.")
            return
        
        print("="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Calculate correlation matrix
        corr_matrix = self.data[self.numerical_cols].corr()
        
        # Find highly correlated pairs
        print("\nHighly Correlated Pairs (|correlation| > 0.7):")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr:
            for col1, col2, corr_val in high_corr:
                print(f"  {col1} - {col2}: {corr_val:.3f}")
        else:
            print("  No highly correlated pairs found.")
        
        # Plot correlation heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                    center=0, square=True, fmt='.2f')
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    
    def plot_scatter_relationships(self, figsize=(15, 5)):
        """
        Plot scatter plots for key relationships.
        
        Args:
            figsize (tuple): Figure size for plots
        """
        if self.data is None:
            print("No data to plot.")
            return
        
        # Define key relationships to plot
        relationships = []
        if 'popularity' in self.data.columns:
            if 'revenue' in self.data.columns:
                relationships.append(('popularity', 'revenue'))
            if 'runtime' in self.data.columns:
                relationships.append(('popularity', 'runtime'))
            if 'budget' in self.data.columns:
                relationships.append(('popularity', 'budget'))
        
        if not relationships:
            print("No suitable columns found for scatter plots.")
            return
        
        fig, axes = plt.subplots(1, len(relationships), figsize=figsize)
        if len(relationships) == 1:
            axes = [axes]
        
        for i, (x_col, y_col) in enumerate(relationships):
            self.data.plot(kind='scatter', x=x_col, y=y_col, 
                          s=32, alpha=0.8, ax=axes[i])
            axes[i].set_title(f'{x_col.title()} vs {y_col.title()}')
            
            # Remove top and right spines
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_categorical_data(self):
        """
        Analyze categorical variables in the dataset.
        """
        if self.data is None:
            print("No data loaded.")
            return
        
        print("="*60)
        print("CATEGORICAL DATA ANALYSIS")
        print("="*60)
        
        # Analyze each categorical column
        for col in self.categorical_cols:
            if col in self.data.columns:
                print(f"\n{col.upper()}:")
                value_counts = self.data[col].value_counts()
                print(f"  Unique values: {self.data[col].nunique()}")
                print(f"  Most common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
                
                if len(value_counts) > 10:
                    print("  Top 10 values:")
                    print(value_counts.head(10))
                else:
                    print("  All values:")
                    print(value_counts)
    
    def analyze_text_features(self):
        """
        Analyze text columns for length, word count, etc.
        """
        if self.data is None or not self.text_cols:
            print("No text data to analyze.")
            return
        
        print("="*60)
        print("TEXT DATA ANALYSIS")
        print("="*60)
        
        for col in self.text_cols:
            if col in self.data.columns:
                print(f"\n{col.upper()}:")
                
                # Handle NaN values
                text_data = self.data[col].fillna('')
                
                # Basic statistics
                text_lengths = text_data.str.len()
                word_counts = text_data.str.split().str.len()
                
                print(f"  Non-empty entries: {(text_data != '').sum()}")
                print(f"  Average length: {text_lengths.mean():.1f} characters")
                print(f"  Average word count: {word_counts.mean():.1f} words")
                print(f"  Max length: {text_lengths.max()} characters")
                print(f"  Min length: {text_lengths.min()} characters")
    
    def genre_analysis(self):
        """
        Perform specific analysis on movie genres.
        """
        if self.data is None or 'genres' not in self.data.columns:
            print("No genre data available.")
            return
        
        print("="*60)
        print("GENRE ANALYSIS")
        print("="*60)
        
        # Extract all unique genres
        all_genres = []
        for genres_str in self.data['genres'].fillna(''):
            # Handle both JSON and space-separated formats
            if genres_str and isinstance(genres_str, str):
                if genres_str.startswith('['):
                    # JSON format
                    try:
                        genres_list = json.loads(genres_str)
                        all_genres.extend([g['name'] if isinstance(g, dict) else str(g) for g in genres_list])
                    except json.JSONDecodeError:
                        pass
                else:
                    # Space-separated format
                    all_genres.extend(genres_str.split())
        
        # Count genre frequencies
        genre_counts = pd.Series(all_genres).value_counts()
        
        print(f"Total unique genres: {len(genre_counts)}")
        print(f"\nTop 10 genres:")
        print(genre_counts.head(10))
        
        # Plot genre distribution
        plt.figure(figsize=(12, 6))
        genre_counts.head(15).plot(kind='bar')
        plt.title('Top 15 Movie Genres Distribution')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return genre_counts
    
    def sentiment_analysis_overview(self):
        """
        Analyze sentiment scores if available.
        """
        sentiment_cols = [col for col in self.data.columns if 'sentiment' in col.lower()]
        
        if not sentiment_cols:
            print("No sentiment data available.")
            return
        
        print("="*60)
        print("SENTIMENT ANALYSIS OVERVIEW")
        print("="*60)
        
        for col in sentiment_cols:
            print(f"\n{col.upper()}:")
            sentiment_data = self.data[col].dropna()
            
            print(f"  Mean sentiment: {sentiment_data.mean():.3f}")
            print(f"  Std sentiment: {sentiment_data.std():.3f}")
            print(f"  Min sentiment: {sentiment_data.min():.3f}")
            print(f"  Max sentiment: {sentiment_data.max():.3f}")
            
            # Plot sentiment distribution
            plt.figure(figsize=(10, 6))
            sentiment_data.hist(bins=20, alpha=0.7, density=True)
            sentiment_data.plot.kde()
            plt.title(f'Distribution of {col}')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def temporal_analysis(self):
        """
        Analyze temporal patterns in the data.
        """
        date_cols = [col for col in self.data.columns if 'date' in col.lower() or 'year' in col.lower()]
        
        if not date_cols:
            print("No temporal data available.")
            return
        
        print("="*60)
        print("TEMPORAL ANALYSIS")
        print("="*60)
        
        for col in date_cols:
            if col in self.data.columns:
                print(f"\n{col.upper()}:")
                
                # Convert to datetime if needed
                if 'date' in col.lower():
                    date_data = pd.to_datetime(self.data[col], errors='coerce')
                    if not date_data.isna().all():
                        year_data = date_data.dt.year
                        print(f"  Date range: {date_data.min()} to {date_data.max()}")
                        print(f"  Year range: {year_data.min()} to {year_data.max()}")
                        
                        # Plot movies per year
                        plt.figure(figsize=(12, 6))
                        year_counts = year_data.value_counts().sort_index()
                        year_counts.plot(kind='line')
                        plt.title(f'Movies per Year ({col})')
                        plt.xlabel('Year')
                        plt.ylabel('Number of Movies')
                        plt.grid(True, alpha=0.3)
                        plt.show()
                else:
                    # Assume it's already year data
                    year_data = pd.to_numeric(self.data[col], errors='coerce')
                    if not year_data.isna().all():
                        print(f"  Year range: {year_data.min()} to {year_data.max()}")
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive EDA report.
        """
        print("="*80)
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS REPORT")
        print("="*80)
        
        # Basic information
        self.basic_info()
        
        # Statistical summary
        self.statistical_summary()
        
        # Correlation analysis
        self.correlation_analysis()
        
        # Categorical analysis
        self.analyze_categorical_data()
        
        # Text analysis
        self.analyze_text_features()
        
        # Domain-specific analyses
        self.genre_analysis()
        self.sentiment_analysis_overview()
        self.temporal_analysis()
        
        # Visualizations
        print("\nGenerating visualizations...")
        self.plot_distributions()
        self.plot_boxplots()
        self.plot_scatter_relationships()
        
        print("\n" + "="*80)
        print("EDA REPORT COMPLETED")
        print("="*80)


def main():
    """
    Example usage of the MovieEDA class.
    """
    # Example: Load and analyze data
    try:
        # Initialize EDA object
        eda = MovieEDA()
        
        # Load data (update path as needed)
        data_path = "path/to/your/cleaned_movie_data.csv"
        eda.load_data(data_path=data_path)
        
        # Generate comprehensive report
        eda.generate_comprehensive_report()
        
    except FileNotFoundError:
        print("File not found. Please update the data path.")
        
        # Example with dummy data
        print("Creating example analysis with dummy data...")
        
        # Create dummy movie data for demonstration
        np.random.seed(42)
        dummy_data = pd.DataFrame({
            'budget': np.random.exponential(50000000, 1000),
            'revenue': np.random.exponential(100000000, 1000),
            'runtime': np.random.normal(110, 20, 1000),
            'vote_average': np.random.normal(6.5, 1.5, 1000),
            'vote_count': np.random.exponential(500, 1000),
            'popularity': np.random.exponential(10, 1000),
            'genres': ['Action Adventure', 'Drama', 'Comedy', 'Thriller', 'Romance'] * 200,
            'overview': ['A great movie about...' for _ in range(1000)],
            'title': [f'Movie {i}' for i in range(1000)],
            'overview_sentiment_score': np.random.normal(0.1, 0.3, 1000)
        })
        
        # Run analysis on dummy data
        eda = MovieEDA(dummy_data)
        eda.basic_info()
        eda.statistical_summary()
        
        print("Example analysis completed with dummy data.")
    
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
