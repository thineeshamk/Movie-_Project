# app/chatbot.py
"""
Movie chatbot responsible for intent detection and response generation.
Focuses on maintainability, clarity, and predictable behavior.
"""

import os
import numpy as np
from groq import Groq
from app.knowledge_base import KnowledgeBase


class MovieChatbot:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.kb = knowledge_base

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set")

        self.llm = Groq(api_key=api_key)

        self.allowed_intents = {
            "similar_movies",
            "movie_metadata",
            "plot_synopsis",
            "recommendations",
            "statistics",
            "specific_movie",
        }

    # ------------------------------------------------------------------
    # Intent classification (simple keyword-based routing)
    # ------------------------------------------------------------------

    def classify_intent(self, message: str) -> str:
        msg = message.lower()

        if any(k in msg for k in ("similar", "like", "comparable", "related")):
            return "similar_movies"
        if any(k in msg for k in ("metadata", "budget", "duration", "actor", "director", "rating")):
            return "movie_metadata"
        if any(k in msg for k in ("plot", "story", "synopsis", "about")):
            return "plot_synopsis"
        if any(k in msg for k in ("recommend", "suggest", "improve", "enhance", "fix")):
            return "recommendations"
        if any(k in msg for k in ("statistics", "average", "stats", "mean")):
            return "statistics"
        if any(k in msg for k in ("tell me about", "show me", "what is")):
            return "specific_movie"

        return "unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_response(
        self,
        message: str,
        movie_data: dict,
        similar_movies: list | None = None,
    ) -> tuple[str, list | None]:
        intent = self.classify_intent(message)

        if intent == "unknown":
            return self._out_of_scope(), None

        if similar_movies is None and intent != "specific_movie":
            similar_movies = self.kb.find_similar_movies(movie_data, k=5)

        handlers = {
            "similar_movies": lambda: (self._similar_movies(similar_movies), similar_movies),
            "movie_metadata": lambda: (self._metadata(similar_movies), similar_movies),
            "plot_synopsis": lambda: (self._plots(similar_movies), similar_movies),
            "recommendations": lambda: (self._recommendations(movie_data, similar_movies), similar_movies),
            "statistics": lambda: (self._statistics(), similar_movies),
            "specific_movie": lambda: (self._specific_movie(message), None),
        }

        return handlers[intent]()

    # ------------------------------------------------------------------
    # Recommendation logic (LLM-assisted, data constrained)
    # ------------------------------------------------------------------

    def _recommendations(self, movie_data: dict, similar_movies: list) -> str:
        reference_blocks = []
        for m in similar_movies:
            reference_blocks.append(
                "\n".join(
                    [
                        f"Title: {m.get('title', 'Unknown')}",
                        f"Genre: {m.get('genre', 'Unknown')}",
                        f"Budget: {m.get('budget', 0)}",
                        f"Duration: {m.get('duration_minutes', 0)}",
                        f"MPA Rating: {m.get('mpa', 'Unknown')}",
                        f"Director IMDb Avg: {m.get('director_avg', 0)}",
                        f"First Actor IMDb Avg: {m.get('first_actor_avg', 0)}",
                        f"Second Actor IMDb Avg: {m.get('second_actor_avg', 0)}",
                    ]
                )
            )

        context = "\n\n---\n\n".join(reference_blocks)

        prompt = f"""
You are acting as a film production data analyst.

Use ONLY the reference movies below to suggest measurable improvements
for the underperforming movie. Avoid generic or creative writing advice.

Reference movies:
{context}

Target movie:
Title: {movie_data.get('title')}
Genre: {movie_data.get('genre')}
Budget: {movie_data.get('budget')}
Duration: {movie_data.get('duration_minutes')}
MPA Rating: {movie_data.get('mpa_rating')}
Director IMDb Avg: {movie_data.get('average_imdb_rating')}
First Actor IMDb Avg: {movie_data.get('first_actor_avg')}
Second Actor IMDb Avg: {movie_data.get('second_actor_avg')}

Provide:
- Cast & Director adjustments
- Budget scaling suggestions
- Duration / pacing alignment
- MPA rating suitability
- Plot changes ONLY if supported by the data
- A short data-driven summary
"""

        try:
            result = self.llm.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=1800,
            )
            return result.choices[0].message.content
        except Exception as exc:
            return f"Failed to generate recommendations: {exc}"

    # ------------------------------------------------------------------
    # Standard responses
    # ------------------------------------------------------------------

    def _similar_movies(self, movies: list) -> str:
        lines = ["Top similar successful movies:\n"]
        for idx, m in enumerate(movies, 1):
            similarity = 1 / (1 + m.get("distance", 0))
            lines.append(
                f"{idx}. {m.get('title')} | {m.get('genre')} | "
                f"${m.get('budget', 0):,.0f} | {m.get('duration_minutes')} min | "
                f"Similarity: {similarity:.2f}"
            )
        return "\n".join(lines)

    def _metadata(self, movies: list) -> str:
        budgets = [m.get("budget", 0) for m in movies]
        durations = [m.get("duration_minutes", 0) for m in movies]
        directors = [m.get("director_avg", 0) for m in movies]

        return (
            "Metadata averages from similar movies:\n"
            f"Budget: ${np.mean(budgets):,.0f}\n"
            f"Duration: {np.mean(durations):.0f} min\n"
            f"Director IMDb Avg: {np.mean(directors):.2f}"
        )

    def _plots(self, movies: list) -> str:
        output = ["Plot summaries of similar movies:\n"]
        for m in movies:
            plot = m.get("plot", "")
            output.append(f"{m.get('title')}: {plot[:250]}{'...' if len(plot) > 250 else ''}")
        return "\n\n".join(output)

    def _statistics(self) -> str:
        stats = self.kb.get_statistics()
        return (
            f"Knowledge base size: {stats.get('count', 0)} movies\n"
            f"Average budget: ${stats.get('avg_budget', 0):,.0f}"
        )

    def _specific_movie(self, message: str) -> str:
        title = None
        if '"' in message:
            title = message.split('"')[1]
        elif "about " in message.lower():
            title = message.lower().split("about ")[-1].strip()

        if not title:
            return "Please specify a movie title."

        movie = self.kb.get_movie_by_title(title)
        if not movie:
            return f"Movie '{title}' not found."

        return (
            f"{movie.get('title')}\n"
            f"Genre: {movie.get('genre')}\n"
            f"Budget: ${movie.get('budget', 0):,.0f}\n"
            f"Plot: {movie.get('plot', '')[:300]}"
        )

    def _out_of_scope(self) -> str:
        return (
            "I can help with movie comparisons, recommendations, and statistics. "
            "Try asking for similar movies or how to improve a film."
        )
