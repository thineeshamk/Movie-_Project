# app/models.py
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List

class GenreEnum(str, Enum):
    """All available genre options"""
    action = "Action"
    comedy = "Comedy"
    drama = "Drama"
    biography = "Biography"
    crime = "Crime"
    adventure = "Adventure"
    horror = "Horror"
    fantasy = "Fantasy"
    mystery = "Mystery"
    thriller = "Thriller"
    sci_fi = "Sci-Fi"
    western = "Western"

class MpaEnum(str, Enum):
    """All available MPA rating options"""
    pg_13 = "PG-13"
    r = "R"
    pg = "PG"
    g = "G"
    nc_17 = "NC-17"
    tv_ma = "TV-MA"
    tv_g = "TV-G"

class MovieInput(BaseModel):
    """Input model for movie prediction"""
    budget: float = Field(..., example=50000000, description="Movie budget in USD")
    duration_minutes: int = Field(..., example=120, description="Duration in minutes")
    first_actor_avg: float = Field(..., example=7.1, description="Lead actor's average IMDb rating")
    second_actor_avg: float = Field(..., example=6.8, description="Second actor's average IMDb rating")
    average_imdb_rating: float = Field(..., example=7.0, description="Director's average IMDb rating")
    mpa_rating: MpaEnum = Field(..., example="PG-13")
    genre: GenreEnum = Field(..., example="Action")
    plot_synopsis: str = Field(..., min_length=50, example="A thrilling story of...")

    class Config:
        use_enum_values = True

class PredictionOutput(BaseModel):
    """Output model for predictions"""
    prediction: str = Field(example="Success")
    probability_success: float = Field(example=0.82)
    model_version: str = Field(example="1.0.0")

class ChatMessage(BaseModel):
    """Chat message from user"""
    message: str = Field(..., min_length=1, max_length=1000)
    movie_data: dict = Field(..., description="Current movie data for context")

class SimilarMovie(BaseModel):
    """Model for a similar successful movie"""
    title: str
    genre: str
    budget: float
    duration_minutes: int
    first_actor_avg: float
    second_actor_avg: float
    director_avg: float
    mpa: str
    plot_summary: str = Field(..., max_length=500)
    distance: float

class ChatResponse(BaseModel):
    """Response from chatbot"""
    response: str
    similar_movies: Optional[List[SimilarMovie]] = None
    success: bool = True
    error: Optional[str] = None