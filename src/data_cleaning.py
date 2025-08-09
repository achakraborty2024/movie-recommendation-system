"""
Data Cleaning Module for Movie Recommendation System
====================================================

This module handles data loading, merging, and preprocessing for the movie recommendation system.
It combines movies.csv and credits.csv datasets and performs feature engineering to prepare
data for analysis and modeling.
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')


class MovieDataCleaner:
    """
    A class to handle data cleaning and preprocessing for movie datasets.
    """
    
    def __init__(self, movies_path=None, credits_path=None):
        """
        Initialize the data cleaner.
        
        Args:
            movies_path (str): Path to movies CSV file
            credits_path (str): Path to credits CSV file
        """
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.movies_df = None
        self.credits_df = None
        self.merged_df = None
    
    def load_data(self, movies_path=None, credits_path=None):
        """
        Load movies and credits data from CSV files.
        
        Args:
            movies_path (str, optional): Path to movies CSV file
            credits_path (str, optional): Path to credits CSV file
        """
        if movies_path:
            self.movies_path = movies_path
        if credits_path:
            self.credits_path = credits_path
            
        print("Loading data...")
        self.movies_df = pd.read_csv(self.movies_path)
        self.credits_df = pd.read_csv(self.credits_path)
        
        print(f"Movies data shape: {self.movies_df.shape}")
        print(f"Credits data shape: {self.credits_df.shape}")
        
        return self.movies_df, self.credits_df
    
    def get_data_info(self):
        """
        Get basic information about the loaded datasets.
        """
        if self.movies_df is None or self.credits_df is None:
            print("Data not loaded. Please load data first.")
            return
            
        print("\n=== MOVIES DATASET INFO ===")
        print(f"Shape: {self.movies_df.shape}")
        print(f"Columns: {list(self.movies_df.columns)}")
        print("\nMissing values:")
        print(self.movies_df.isnull().sum())
        print(f"\nDuplicates: {self.movies_df.duplicated().sum()}")
        
        print("\n=== CREDITS DATASET INFO ===")
        print(f"Shape: {self.credits_df.shape}")
        print(f"Columns: {list(self.credits_df.columns)}")
        print("\nMissing values:")
        print(self.credits_df.isnull().sum())
        print(f"\nDuplicates: {self.credits_df.duplicated().sum()}")
    
    def parse_json_column(self, json_string):
        """
        Parse JSON string and extract names.
        
        Args:
            json_string (str): JSON string to parse
            
        Returns:
            list: List of names from the JSON objects
        """
        try:
            list_of_dicts = json.loads(json_string)
            return [item['name'] for item in list_of_dicts]
        except (json.JSONDecodeError, TypeError):
            return []
    
    def extract_names_from_json(self, json_string):
        """
        Extract names from JSON-like strings for features like production companies.
        
        Args:
            json_string (str): JSON string to parse
            
        Returns:
            str: Space-separated names
        """
        if isinstance(json_string, str):
            try:
                list_of_dicts = json.loads(json_string)
                return ' '.join([d['name'].replace(" ", "") for d in list_of_dicts])
            except (json.JSONDecodeError, TypeError):
                return ''
        return ''
    
    def get_director(self, crew_list):
        """
        Extract director from crew list.
        
        Args:
            crew_list (list): List of crew members
            
        Returns:
            str or None: Director name if found
        """
        for item in crew_list:
            if item == 'Director':
                return item
        return None
    
    def list_to_string(self, lst):
        """
        Convert list to space-separated string with no spaces in individual items.
        
        Args:
            lst (list): List to convert
            
        Returns:
            str: Space-separated string
        """
        return ' '.join([str(i).replace(" ", "") for i in lst])
    
    def merge_datasets(self):
        """
        Merge movies and credits datasets and perform feature engineering.
        
        Returns:
            pd.DataFrame: Merged and processed dataframe
        """
        if self.movies_df is None or self.credits_df is None:
            print("Data not loaded. Please load data first.")
            return None
        
        print("Merging datasets...")
        
        # Ensure title columns are strings
        self.movies_df['title'] = self.movies_df['title'].astype(str)
        self.credits_df['title'] = self.credits_df['title'].astype(str)
        
        # Merge datasets on title
        self.merged_df = self.movies_df.merge(self.credits_df, on='title')
        
        print(f"Merged dataset shape: {self.merged_df.shape}")
        
        return self.merged_df
    
    def clean_data(self):
        """
        Perform comprehensive data cleaning and feature engineering.
        
        Returns:
            pd.DataFrame: Cleaned and processed dataframe
        """
        if self.merged_df is None:
            print("Data not merged. Please merge datasets first.")
            return None
        
        print("Cleaning and processing data...")
        
        # Drop unnecessary columns
        columns_to_drop = ['homepage', 'spoken_languages', 'tagline']
        existing_columns = [col for col in columns_to_drop if col in self.merged_df.columns]
        if existing_columns:
            self.merged_df.drop(existing_columns, axis=1, inplace=True)
        
        # Handle missing values
        self.merged_df['runtime'].fillna(self.merged_df['runtime'].mean(), inplace=True)
        self.merged_df['vote_average'].fillna(0, inplace=True)
        self.merged_df['vote_count'].fillna(0, inplace=True)
        
        # Process JSON columns
        json_columns = ['genres', 'keywords', 'cast', 'crew']
        for column in json_columns:
            if column in self.merged_df.columns:
                self.merged_df[column] = self.merged_df[column].apply(self.parse_json_column)
        
        # Extract director from crew
        if 'crew' in self.merged_df.columns:
            self.merged_df['director'] = self.merged_df['crew'].apply(
                lambda x: [i for i in x if i in ['Director']]
            )
            self.merged_df['director'] = self.merged_df['director'].apply(
                lambda x: x[0] if x else None
            )
            self.merged_df.drop('crew', axis=1, inplace=True)
        
        # Convert lists to strings for text processing
        list_columns = ['genres', 'keywords', 'cast']
        for column in list_columns:
            if column in self.merged_df.columns:
                self.merged_df[column] = self.merged_df[column].apply(self.list_to_string)
        
        # Clean director column
        if 'director' in self.merged_df.columns:
            self.merged_df['director'] = self.merged_df['director'].apply(
                lambda x: str(x).replace(" ", "") if x else ''
            )
        
        # Create enhanced features for production companies, countries
        if 'production_companies' in self.merged_df.columns:
            self.merged_df['production_companies_names'] = self.merged_df['production_companies'].apply(
                self.extract_names_from_json
            )
        
        if 'production_countries' in self.merged_df.columns:
            self.merged_df['production_countries_names'] = self.merged_df['production_countries'].apply(
                self.extract_names_from_json
            )
        
        if 'original_language' in self.merged_df.columns:
            self.merged_df['spoken_languages_names'] = self.merged_df['original_language'].apply(
                self.extract_names_from_json
            )
        
        # Create 'soup' column for content-based recommendations
        self.create_soup_column()
        
        print("Data cleaning completed.")
        return self.merged_df
    
    def create_soup_column(self):
        """
        Create a 'soup' column combining various text features for content-based filtering.
        """
        # Basic soup with core features
        soup_components = []
        
        if 'title' in self.merged_df.columns:
            soup_components.append(self.merged_df['title'].fillna(''))
        
        if 'overview' in self.merged_df.columns:
            soup_components.append(self.merged_df['overview'].fillna(''))
        
        if 'genres' in self.merged_df.columns:
            soup_components.append(self.merged_df['genres'].fillna(''))
        
        if 'keywords' in self.merged_df.columns:
            soup_components.append(self.merged_df['keywords'].fillna(''))
        
        if 'cast' in self.merged_df.columns:
            soup_components.append(self.merged_df['cast'].fillna(''))
        
        if 'director' in self.merged_df.columns:
            soup_components.append(self.merged_df['director'].fillna(''))
        
        if 'release_date' in self.merged_df.columns:
            soup_components.append(self.merged_df['release_date'].fillna(''))
        
        # Combine all components
        self.merged_df['soup'] = ' '.join([' '.join(component.astype(str)) for component in soup_components])
        
        # Create enhanced soup with additional features
        enhanced_components = soup_components.copy()
        
        if 'production_companies_names' in self.merged_df.columns:
            enhanced_components.append(self.merged_df['production_companies_names'].fillna(''))
        
        if 'production_countries_names' in self.merged_df.columns:
            enhanced_components.append(self.merged_df['production_countries_names'].fillna(''))
        
        if 'spoken_languages_names' in self.merged_df.columns:
            enhanced_components.append(self.merged_df['spoken_languages_names'].fillna(''))
        
        # Create enhanced soup
        self.merged_df['enhanced_soup'] = ' '.join([' '.join(component.astype(str)) for component in enhanced_components])
        
        print("Created 'soup' and 'enhanced_soup' columns for content-based filtering.")
    
    def save_cleaned_data(self, output_path):
        """
        Save cleaned data to CSV file.
        
        Args:
            output_path (str): Path to save the cleaned data
        """
        if self.merged_df is None:
            print("No cleaned data to save. Please run cleaning process first.")
            return
        
        self.merged_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")


def main():
    """
    Example usage of the MovieDataCleaner class.
    """
    # Initialize cleaner
    cleaner = MovieDataCleaner()
    
    # Example file paths (update these with your actual file paths)
    movies_path = "path/to/movies.csv"
    credits_path = "path/to/credits.csv"
    
    try:
        # Load data
        cleaner.load_data(movies_path, credits_path)
        
        # Get data info
        cleaner.get_data_info()
        
        # Merge datasets
        merged_data = cleaner.merge_datasets()
        
        # Clean data
        cleaned_data = cleaner.clean_data()
        
        # Save cleaned data
        cleaner.save_cleaned_data("cleaned_movie_data.csv")
        
        print("\nData cleaning completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please update the file paths in the main() function.")
    except Exception as e:
        print(f"Error during data cleaning: {e}")


if __name__ == "__main__":
    main()
