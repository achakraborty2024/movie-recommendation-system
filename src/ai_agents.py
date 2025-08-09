"""
AI Agents Module

This module implements AI agents for movie recommendations using LangChain and OpenAI.
It includes agentic workflows, LLM-powered recommendation explanations, and interactive agents.

Classes:
    - MovieRecommendationAgent: Main agentic recommendation system
    - SentimentAnalysisAgent: Sentiment detection for movie preferences
    - ExplanationAgent: Generate explanations for recommendations
    - AgenticPipeline: Complete agentic recommendation pipeline

Dependencies:
    - langchain, langchain-openai
    - openai
    - pandas, numpy
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.tools import tool
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain not available. Some agent features will be disabled.")
    LANGCHAIN_AVAILABLE = False

# OpenAI import for direct API usage
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI not available. Some features will be disabled.")
    OPENAI_AVAILABLE = False


class MovieRecommendationAgent:
    """
    Main agent class for movie recommendations using LangChain and LLM.
    """
    
    def __init__(self, movies_df=None, api_key=None, model_name="gpt-4o"):
        """
        Initialize the movie recommendation agent.
        
        Parameters:
        - movies_df (pd.DataFrame): Movie dataset
        - api_key (str): OpenAI API key
        - model_name (str): OpenAI model name to use
        """
        self.movies_df = movies_df
        self.model_name = model_name
        self.llm = None
        self.agent_executor = None
        
        # Set up API key
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        # Initialize LLM if available
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(temperature=0, model=model_name)
                print(f"Initialized {model_name} for agent")
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                self.llm = None
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Create agent if LLM is available
        if self.llm and self.tools:
            self._create_agent()
    
    def _create_tools(self):
        """Create LangChain tools for the agent."""
        tools = []
        
        if not LANGCHAIN_AVAILABLE:
            return tools
        
        @tool
        def get_movie_info(movie_title: str) -> str:
            """Get information about a specific movie from the database."""
            if self.movies_df is None or self.movies_df.empty:
                return "Movie database not available."
            
            try:
                # Search for the movie (case-insensitive)
                matches = self.movies_df[self.movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)]
                
                if matches.empty:
                    return f"Movie '{movie_title}' not found in database."
                
                movie = matches.iloc[0]
                info = f"Title: {movie.get('title', 'N/A')}\n"
                info += f"Genres: {movie.get('genres', 'N/A')}\n"
                info += f"Overview: {movie.get('overview', 'N/A')}\n"
                info += f"Rating: {movie.get('vote_average', 'N/A')}\n"
                info += f"Vote Count: {movie.get('vote_count', 'N/A')}"
                
                return info
                
            except Exception as e:
                return f"Error retrieving movie information: {str(e)}"
        
        @tool
        def search_movies_by_genre(genres: str) -> str:
            """Search for movies by genre(s). Provide genres as comma-separated values."""
            if self.movies_df is None or self.movies_df.empty:
                return "Movie database not available."
            
            try:
                genre_list = [g.strip().lower() for g in genres.split(',')]
                
                # Filter movies that contain any of the specified genres
                matching_movies = []
                
                for idx, row in self.movies_df.iterrows():
                    movie_genres = str(row.get('genres', '')).lower()
                    if any(genre in movie_genres for genre in genre_list):
                        matching_movies.append({
                            'title': row.get('title', 'N/A'),
                            'rating': row.get('vote_average', 0),
                            'genres': row.get('genres', 'N/A')
                        })
                
                if not matching_movies:
                    return f"No movies found for genres: {genres}"
                
                # Sort by rating and take top 10
                matching_movies.sort(key=lambda x: x['rating'], reverse=True)
                top_movies = matching_movies[:10]
                
                result = f"Top movies for genres '{genres}':\n"
                for i, movie in enumerate(top_movies, 1):
                    result += f"{i}. {movie['title']} (Rating: {movie['rating']})\n"
                
                return result
                
            except Exception as e:
                return f"Error searching movies by genre: {str(e)}"
        
        @tool
        def get_popular_movies(limit: int = 10) -> str:
            """Get the most popular movies based on vote count."""
            if self.movies_df is None or self.movies_df.empty:
                return "Movie database not available."
            
            try:
                # Sort by vote_count and get top movies
                popular_movies = self.movies_df.nlargest(limit, 'vote_count')
                
                result = f"Top {limit} popular movies:\n"
                for i, (_, movie) in enumerate(popular_movies.iterrows(), 1):
                    result += f"{i}. {movie.get('title', 'N/A')} "
                    result += f"(Votes: {movie.get('vote_count', 0)}, "
                    result += f"Rating: {movie.get('vote_average', 0)})\n"
                
                return result
                
            except Exception as e:
                return f"Error getting popular movies: {str(e)}"
        
        tools.extend([get_movie_info, search_movies_by_genre, get_popular_movies])
        
        return tools
    
    def _create_agent(self):
        """Create the LangChain agent with tools."""
        if not self.llm or not self.tools:
            return
        
        try:
            # Define agent prompt
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a movie recommendation assistant. Your goal is to help users find movies they might like.
                
                You have access to tools to:
                1. Get information about specific movies
                2. Search movies by genre
                3. Get popular movies
                
                When a user asks for recommendations:
                - Ask about their preferred genres if not specified
                - Use the search_movies_by_genre tool to find relevant movies
                - Provide detailed information about recommended movies
                - Explain why these movies match their preferences
                
                Always be helpful, informative, and personalized in your responses.
                """),
                ("human", "{input}"),
                ("placeholder", "{chat_history}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            # Create agent
            agent = create_tool_calling_agent(self.llm, self.tools, agent_prompt)
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )
            
            print("Agent created successfully")
            
        except Exception as e:
            print(f"Error creating agent: {e}")
            self.agent_executor = None
    
    def get_recommendations(self, user_input: str) -> str:
        """
        Get movie recommendations based on user input.
        
        Parameters:
        - user_input (str): User's request for recommendations
        
        Returns:
        - str: Agent's response with recommendations
        """
        if not self.agent_executor:
            return "Agent not available. Please check initialization."
        
        try:
            response = self.agent_executor.invoke({'input': user_input})
            return response.get('output', 'No response generated.')
            
        except Exception as e:
            return f"Error getting recommendations: {str(e)}"


class SentimentAnalysisAgent:
    """
    Agent specialized in sentiment analysis for movie preferences.
    """
    
    def __init__(self, api_key=None, model_name="gpt-4o"):
        """
        Initialize sentiment analysis agent.
        
        Parameters:
        - api_key (str): OpenAI API key
        - model_name (str): OpenAI model name
        """
        self.model_name = model_name
        self.llm = None
        
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(temperature=0, model=model_name)
            except Exception as e:
                print(f"Error initializing sentiment analysis LLM: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of user input.
        
        Parameters:
        - text (str): Text to analyze
        
        Returns:
        - dict: Sentiment analysis results
        """
        if not self.llm:
            return {"error": "LLM not available for sentiment analysis"}
        
        try:
            # Create sentiment analysis prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze the sentiment of the following text about movie preferences.
                
                Respond with a JSON object containing:
                - sentiment: 'positive', 'negative', or 'neutral'
                - confidence: float between 0 and 1
                - mood: one of 'happy', 'sad', 'excited', 'calm', 'anxious', 'nostalgic', 'adventurous'
                - intensity: 'low', 'medium', or 'high'
                
                Example response:
                {
                    "sentiment": "positive",
                    "confidence": 0.85,
                    "mood": "excited",
                    "intensity": "high"
                }
                """),
                ("human", "{text}")
            ])
            
            # Create chain
            chain = prompt | self.llm
            
            # Get response
            response = chain.invoke({"text": text})
            
            # Parse response
            import json
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                # Fallback to simple sentiment
                content = response.content.lower()
                if 'positive' in content:
                    sentiment = 'positive'
                elif 'negative' in content:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {
                    "sentiment": sentiment,
                    "confidence": 0.5,
                    "mood": "neutral",
                    "intensity": "medium"
                }
            
        except Exception as e:
            return {"error": f"Error in sentiment analysis: {str(e)}"}
    
    def extract_preferences(self, text: str) -> Dict[str, List[str]]:
        """
        Extract movie preferences from text.
        
        Parameters:
        - text (str): User input text
        
        Returns:
        - dict: Extracted preferences
        """
        if not self.llm:
            return {"error": "LLM not available"}
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract movie preferences from the user's text.
                
                Identify and return:
                - liked_genres: genres the user likes
                - disliked_genres: genres the user dislikes
                - liked_elements: specific elements they enjoy (e.g., "action scenes", "plot twists")
                - disliked_elements: specific elements they dislike
                - mood_preference: desired mood for recommendations
                
                Return as JSON format.
                
                Example:
                {
                    "liked_genres": ["action", "sci-fi"],
                    "disliked_genres": ["horror", "romance"],
                    "liked_elements": ["plot twists", "space battles"],
                    "disliked_elements": ["slow pacing"],
                    "mood_preference": "exciting"
                }
                """),
                ("human", "{text}")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({"text": text})
            
            # Parse response
            import json
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                return {
                    "liked_genres": [],
                    "disliked_genres": [],
                    "liked_elements": [],
                    "disliked_elements": [],
                    "mood_preference": "neutral"
                }
            
        except Exception as e:
            return {"error": f"Error extracting preferences: {str(e)}"}


class ExplanationAgent:
    """
    Agent that generates explanations for movie recommendations.
    """
    
    def __init__(self, api_key=None, model_name="gpt-4o"):
        """
        Initialize explanation agent.
        
        Parameters:
        - api_key (str): OpenAI API key
        - model_name (str): OpenAI model name
        """
        self.model_name = model_name
        self.llm = None
        
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(temperature=0.7, model=model_name)  # Slightly higher temp for creativity
            except Exception as e:
                print(f"Error initializing explanation LLM: {e}")
    
    def generate_explanation(self, movie_info: Dict[str, Any], user_preferences: Dict[str, Any]) -> str:
        """
        Generate explanation for why a movie is recommended.
        
        Parameters:
        - movie_info (dict): Information about the recommended movie
        - user_preferences (dict): User's preferences and context
        
        Returns:
        - str: Explanation for the recommendation
        """
        if not self.llm:
            return "Explanation generation not available."
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert movie critic and recommendation explainer.
                
                Generate a personalized explanation for why a movie is being recommended to a user.
                The explanation should:
                - Be concise (2-3 sentences)
                - Connect the movie's features to the user's preferences
                - Be engaging and persuasive
                - Highlight the most relevant aspects
                
                Make it sound natural and conversational.
                """),
                ("human", """Movie Information:
                Title: {title}
                Genres: {genres}
                Overview: {overview}
                Rating: {rating}
                
                User Preferences:
                {preferences}
                
                Generate an explanation for why this movie is recommended:""")
            ])
            
            # Format user preferences
            prefs_text = ""
            if isinstance(user_preferences, dict):
                for key, value in user_preferences.items():
                    if value:
                        prefs_text += f"{key}: {value}\n"
            else:
                prefs_text = str(user_preferences)
            
            chain = prompt | self.llm
            
            response = chain.invoke({
                "title": movie_info.get("title", "Unknown"),
                "genres": movie_info.get("genres", "Unknown"),
                "overview": movie_info.get("overview", "No overview available"),
                "rating": movie_info.get("rating", "N/A"),
                "preferences": prefs_text
            })
            
            return response.content.strip()
            
        except Exception as e:
            return f"Could not generate explanation: {str(e)}"
    
    def batch_explain(self, recommendations: List[Dict[str, Any]], user_preferences: Dict[str, Any]) -> List[str]:
        """
        Generate explanations for multiple recommendations.
        
        Parameters:
        - recommendations (list): List of recommended movies
        - user_preferences (dict): User preferences
        
        Returns:
        - list: List of explanations
        """
        explanations = []
        
        for movie in recommendations:
            explanation = self.generate_explanation(movie, user_preferences)
            explanations.append(explanation)
        
        return explanations


class AgenticPipeline:
    """
    Complete agentic pipeline combining all agents.
    """
    
    def __init__(self, movies_df=None, api_key=None, model_name="gpt-4o"):
        """
        Initialize the complete agentic pipeline.
        
        Parameters:
        - movies_df (pd.DataFrame): Movie dataset
        - api_key (str): OpenAI API key
        - model_name (str): OpenAI model name
        """
        self.movies_df = movies_df
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize agents
        self.recommendation_agent = MovieRecommendationAgent(movies_df, api_key, model_name)
        self.sentiment_agent = SentimentAnalysisAgent(api_key, model_name)
        self.explanation_agent = ExplanationAgent(api_key, model_name)
    
    def process_user_request(self, user_input: str) -> Dict[str, Any]:
        """
        Process complete user request through the agentic pipeline.
        
        Parameters:
        - user_input (str): User's movie preference request
        
        Returns:
        - dict: Complete response with recommendations and explanations
        """
        try:
            # Step 1: Analyze sentiment and extract preferences
            print("Analyzing user sentiment and preferences...")
            sentiment_result = self.sentiment_agent.analyze_sentiment(user_input)
            preferences = self.sentiment_agent.extract_preferences(user_input)
            
            # Step 2: Get recommendations from main agent
            print("Getting recommendations...")
            recommendations_text = self.recommendation_agent.get_recommendations(user_input)
            
            # Step 3: Parse recommendations (simplified - in production, would need better parsing)
            recommendations = self._parse_recommendations(recommendations_text)
            
            # Step 4: Generate explanations
            print("Generating explanations...")
            explanations = []
            if recommendations:
                explanations = self.explanation_agent.batch_explain(recommendations, preferences)
            
            # Step 5: Compile final response
            response = {
                "user_input": user_input,
                "sentiment_analysis": sentiment_result,
                "extracted_preferences": preferences,
                "recommendations": recommendations,
                "explanations": explanations,
                "raw_response": recommendations_text
            }
            
            return response
            
        except Exception as e:
            return {
                "error": f"Error in agentic pipeline: {str(e)}",
                "user_input": user_input
            }
    
    def _parse_recommendations(self, recommendations_text: str) -> List[Dict[str, Any]]:
        """
        Parse recommendations from agent response text.
        
        Parameters:
        - recommendations_text (str): Raw text response from agent
        
        Returns:
        - list: Parsed recommendations
        """
        # Simplified parsing - in production would need more sophisticated parsing
        recommendations = []
        
        try:
            lines = recommendations_text.split('\n')
            current_movie = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                    # New movie entry
                    if current_movie:
                        recommendations.append(current_movie)
                        current_movie = {}
                    
                    # Extract title from line
                    parts = line.split(' ', 1)
                    if len(parts) > 1:
                        title_part = parts[1]
                        # Remove rating info if present
                        if '(' in title_part:
                            title = title_part.split('(')[0].strip()
                        else:
                            title = title_part.strip()
                        current_movie['title'] = title
            
            # Add last movie
            if current_movie:
                recommendations.append(current_movie)
            
            # Enrich with database info if available
            if self.movies_df is not None:
                for rec in recommendations:
                    title = rec.get('title', '')
                    matches = self.movies_df[self.movies_df['title'].str.contains(title, case=False, na=False)]
                    if not matches.empty:
                        movie = matches.iloc[0]
                        rec.update({
                            'genres': movie.get('genres', ''),
                            'overview': movie.get('overview', ''),
                            'rating': movie.get('vote_average', 0)
                        })
            
        except Exception as e:
            print(f"Error parsing recommendations: {e}")
        
        return recommendations


def main():
    """
    Example usage of the AI Agents module.
    """
    print("AI Agents Module")
    print("=" * 50)
    
    # Example movie data
    sample_movies = pd.DataFrame({
        'title': ['Inception', 'The Dark Knight', 'Interstellar', 'Pulp Fiction', 'The Shawshank Redemption'],
        'genres': ['Sci-Fi|Action|Thriller', 'Action|Crime|Drama', 'Sci-Fi|Drama', 'Crime|Drama', 'Drama'],
        'overview': [
            'A thief who steals corporate secrets through dream-sharing technology.',
            'Batman fights crime in Gotham City.',
            'A team of explorers travel through a wormhole in space.',
            'The lives of two mob hitmen, a boxer, and others intertwine.',
            'Two imprisoned men bond over years, finding solace through acts of decency.'
        ],
        'vote_average': [8.8, 9.0, 8.6, 8.9, 9.3],
        'vote_count': [28000, 32000, 25000, 27000, 35000]
    })
    
    # Note: You would need to set your OpenAI API key
    # api_key = "your-openai-api-key-here"
    api_key = None  # Set to None for demo
    
    print("\n1. Testing Sentiment Analysis Agent...")
    sentiment_agent = SentimentAnalysisAgent(api_key=api_key)
    
    if sentiment_agent.llm:
        test_text = "I love action movies with lots of explosions and adventure!"
        sentiment_result = sentiment_agent.analyze_sentiment(test_text)
        print(f"Sentiment analysis: {sentiment_result}")
        
        preferences = sentiment_agent.extract_preferences(test_text)
        print(f"Extracted preferences: {preferences}")
    else:
        print("Sentiment agent not available (OpenAI API key not set)")
    
    print("\n2. Testing Movie Recommendation Agent...")
    rec_agent = MovieRecommendationAgent(movies_df=sample_movies, api_key=api_key)
    
    if rec_agent.agent_executor:
        response = rec_agent.get_recommendations("I want sci-fi movies similar to Inception")
        print(f"Recommendations: {response}")
    else:
        print("Recommendation agent not available (OpenAI API key not set)")
    
    print("\n3. Testing Explanation Agent...")
    explanation_agent = ExplanationAgent(api_key=api_key)
    
    if explanation_agent.llm:
        movie_info = {
            "title": "Inception",
            "genres": "Sci-Fi|Action|Thriller",
            "overview": "A thief who steals corporate secrets through dream-sharing technology.",
            "rating": 8.8
        }
        
        user_prefs = {
            "liked_genres": ["sci-fi", "action"],
            "mood_preference": "exciting"
        }
        
        explanation = explanation_agent.generate_explanation(movie_info, user_prefs)
        print(f"Explanation: {explanation}")
    else:
        print("Explanation agent not available (OpenAI API key not set)")
    
    print("\n4. Testing Complete Agentic Pipeline...")
    pipeline = AgenticPipeline(movies_df=sample_movies, api_key=api_key)
    
    if LANGCHAIN_AVAILABLE and api_key:
        user_request = "I'm in the mood for something exciting with sci-fi elements"
        result = pipeline.process_user_request(user_request)
        print(f"Pipeline result: {result}")
    else:
        print("Complete pipeline not available (requires OpenAI API key and LangChain)")
    
    print("\nAI Agents example completed!")
    print("\nNote: To fully test this module, you need:")
    print("1. OpenAI API key set as environment variable or passed as parameter")
    print("2. LangChain and langchain-openai packages installed")
    print("3. A proper movie dataset with the required columns")


if __name__ == "__main__":
    main()
