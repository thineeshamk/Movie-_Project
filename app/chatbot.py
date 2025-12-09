# app/chatbot.py
import os
import numpy as np
from groq import Groq
from app.knowledge_base import KnowledgeBase

class MovieChatbot:
    """Chatbot that provides data-driven recommendations"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

        # Load Groq API key
        groq_api_key = os.environ.get("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_api_key)

        # Supported intents
        self.allowed_intents = [
            "similar_movies",
            "movie_metadata",
            "plot_synopsis",
            "recommendations",
            "statistics",
            "specific_movie",
        ]

    # ==============================================================
    # Intent Classification
    # ==============================================================

    def classify_intent(self, message: str) -> str:
        """Classify user's intent based on keywords"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["similar", "like", "comparable", "related"]):
            return "similar_movies"
        elif any(word in message_lower for word in ["metadata", "data", "budget", "duration", "actor", "director", "rating"]):
            return "movie_metadata"
        elif any(word in message_lower for word in ["plot", "story", "synopsis", "about"]):
            return "plot_synopsis"
        elif any(word in message_lower for word in ["recommend", "improve", "suggestion", "advice", "better", "enhance", "fix"]):
            return "recommendations"
        elif any(word in message_lower for word in ["statistics", "average", "typical", "stats", "mean"]):
            return "statistics"
        elif any(word in message_lower for word in ["tell me about", "show me", "what is"]):
            return "specific_movie"
        else:
            return "unknown"

    # ==============================================================
    # Main Response Handler
    # ==============================================================

    def generate_response(self, message: str, movie_data: dict, similar_movies: list = None) -> tuple:
        """Generate chatbot response based on classified intent"""
        intent = self.classify_intent(message)

        if intent == "unknown":
            return self._out_of_scope_response(), None

        if similar_movies is None:
            # Assumes kb.find_similar_movies returns a list of dicts similar to the dataframe rows
            similar_movies = self.kb.find_similar_movies(movie_data, k=5)

        if intent == "similar_movies":
            return self._similar_movies_response(similar_movies), similar_movies
        elif intent == "movie_metadata":
            return self._metadata_response(message, similar_movies), similar_movies
        elif intent == "plot_synopsis":
            return self._plot_response(message, similar_movies), similar_movies
        elif intent == "recommendations":
            return self._recommendations_response(movie_data, similar_movies), similar_movies
        elif intent == "statistics":
            return self._statistics_response(movie_data), similar_movies
        elif intent == "specific_movie":
            return self._specific_movie_response(message), None

        return self._out_of_scope_response(), None

    # ==============================================================
    # 1. Recommendations Response (UPDATED TO MATCH STANDALONE SCRIPT)
    # ==============================================================

    def _recommendations_response(self, movie_data: dict, similar_movies: list) -> str:
        """Generate data-driven recommendations using successful movies as baseline"""

        # 1. Build Reference Context (Matching the standalone script format)
        # Note: We use .get() to handle potential key naming differences (snake_case vs Title Case)
        context_items = []
        for m in similar_movies:
            item = (
                f"Title: {m.get('title', 'Unknown')}\n"
                f"Genre: {m.get('genre', m.get('1st Genre', 'Unknown'))}\n"
                f"Budget: {m.get('budget', 0)}\n"
                f"Duration: {m.get('duration_minutes', m.get('Duration_Minutes', 0))} minutes\n"
                f"MPA Rating: {m.get('mpa', m.get('MPA', 'Unknown'))}\n"
                f"Director IMDb Avg: {m.get('director_avg', m.get('Director Avg', 0))}\n"
                f"First Actor IMDb Avg: {m.get('first_actor_avg', m.get('First Actor Avg', 0))}\n"
                f"Second Actor IMDb Avg: {m.get('second_actor_avg', m.get('Second Actor Avg', 0))}\n"
                f"Plot: {m.get('plot', m.get('Movie Plot', ''))}"
            )
            context_items.append(item)
        
        context_text = "\n\n---\n\n".join(context_items)

        # 2. Build The Enhanced Prompt
        # This mirrors the "Prompt (Focused on Data-Based Adjustments)" from your standalone file
        prompt = f"""
You are a **Hollywood production data consultant** specializing in making underperforming films more successful
using measurable data — such as IMDb averages of cast and director, budget levels, movie duration, and MPA rating patterns.

Your task is to analyze the given **unsuccessful movie** and provide **data-grounded recommendations** using only
the {len(similar_movies)} successful movies provided below. Use these movies as the baseline of success.

Do NOT use general creative writing advice — your recommendations must be logical, quantitative, and pattern-based.

Below are {len(similar_movies)} successful movies (used as reference data):
{context_text}

Here is the underperforming movie that needs adjustments:
Title: {movie_data.get('title', 'My Movie')}
Genre: {movie_data.get('genre', 'Unknown')}
Budget: {movie_data.get('budget', 0)}
Duration: {movie_data.get('duration_minutes', 0)} minutes
MPA Rating: {movie_data.get('mpa_rating', 'Unknown')}
Director IMDb Avg: {movie_data.get('average_imdb_rating', 0)}
First Actor IMDb Avg: {movie_data.get('first_actor_avg', 0)}
Second Actor IMDb Avg: {movie_data.get('second_actor_avg', 0)}
Plot: {movie_data.get('plot_synopsis', '')}

Your analysis must:
1️⃣ Compare actor, director, and budget statistics with the successful set and identify mismatches.
2️⃣ Suggest optimal ranges or adjustments for:
   - Cast and Director IMDb averages
   - Budget size (relative to success pattern)
   - Duration and pacing
   - MPA rating suitability
3️⃣ Give **plot-related improvements only if supported by the pattern in successful movies** (tone, focus, clarity, or genre alignment).

⚡ Output format (strictly follow this structure):
- 👥 Cast & Director Adjustment Recommendations
- 💰 Budget Allocation and Scaling
- ⏱ Duration and Pacing Optimization
- 🔞 MPA Rating and Target Audience Refinement
- 🧠 Plot Improvement (data-aligned only)
- 📈 Summary of Data-Based Insights
"""

        # 3. Generate structured LLM response
        try:
            response = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-20b", # Or "llama3-70b-8192" based on your preference
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048 # Increased to allow for the full detailed response
            )
            return "**Data-Driven Recommendations:**\n\n" + response.choices[0].message.content

        except Exception as e:
            return f"Error generating AI recommendations: {str(e)}"

    # ==============================================================
    # Other Responses (Standard)
    # ==============================================================

    def _out_of_scope_response(self) -> str:
        return (
            "I apologize, but I can only answer questions related to successful movies in my knowledge base. \n\n"
            "Try asking for: 'Recommendations to improve this movie', 'Similar movies', or 'Movie statistics'."
        )

    def _similar_movies_response(self, similar_movies: list) -> str:
        response = "**Top 5 Similar Successful Movies:**\n\n"
        for i, movie in enumerate(similar_movies, 1):
            # Calculate similarity score if distance exists, otherwise default
            similarity = 1 / (1 + movie.get('distance', 0)) if 'distance' in movie else 0.0
            response += (
                f"**{i}. {movie.get('title', 'Unknown')}**\n"
                f"   • Genre: {movie.get('genre', 'Unknown')}\n"
                f"   • Budget: ${movie.get('budget', 0):,.0f}\n"
                f"   • Duration: {movie.get('duration_minutes', 0)} min\n"
                f"   • Director Rating: {movie.get('director_avg', 0):.2f}\n"
                f"   • Similarity Score: {similarity:.2f}\n\n"
            )
        return response

    def _metadata_response(self, message: str, similar_movies: list) -> str:
        # Extract values safely using list comprehensions
        budgets = [m.get('budget', 0) for m in similar_movies]
        durations = [m.get('duration_minutes', 0) for m in similar_movies]
        directors = [m.get('director_avg', 0) for m in similar_movies]
        
        avg_budget = np.mean(budgets) if budgets else 0
        avg_duration = np.mean(durations) if durations else 0
        avg_director = np.mean(directors) if directors else 0

        response = "**Metadata Comparison:**\n\n"
        response += f"**Average from Top 5 Similar Movies:**\n"
        response += f"• Budget: ${avg_budget:,.0f}\n"
        response += f"• Duration: {avg_duration:.0f} minutes\n"
        response += f"• Director Rating: {avg_director:.2f}\n\n"
        
        return response

    def _plot_response(self, message: str, similar_movies: list) -> str:
        response = "**Plot Synopses of Similar Successful Movies:**\n\n"
        for i, movie in enumerate(similar_movies, 1):
            plot = movie.get('plot', movie.get('Movie Plot', 'No plot available'))
            plot_short = plot[:300] + "..." if len(plot) > 300 else plot
            response += f"**{i}. {movie.get('title')}**\n{plot_short}\n\n"
        return response

    def _statistics_response(self, movie_data: dict) -> str:
        # Assumes kb.get_statistics is implemented
        genre = movie_data.get('genre')
        overall_stats = self.kb.get_statistics()
        
        response = "**Knowledge Base Statistics:**\n\n"
        response += f"**Overall ({overall_stats.get('count', 0)} movies):**\n"
        response += f"• Average Budget: ${overall_stats.get('avg_budget', 0):,.0f}\n"
        return response

    def _specific_movie_response(self, message: str) -> str:
        # Extract title logic
        if '"' in message:
            title = message.split('"')[1]
        elif "about " in message.lower():
            title = message.lower().split("about ")[-1].strip()
        else:
            return 'Please specify a movie title (e.g., Tell me about "Cast Away").'

        movie = self.kb.get_movie_by_title(title)

        if movie:
            return (
                f"🎬 **{movie.get('title')}**\n"
                f"Genre: {movie.get('genre')}\n"
                f"Budget: ${movie.get('budget'):,.0f}\n"
                f"Plot: {movie.get('plot')[:300]}..."
            )
        return f"I couldn't find a movie titled '{title}'."