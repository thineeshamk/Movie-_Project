# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.models import MovieInput, PredictionOutput, ChatMessage, ChatResponse, SimilarMovie
from app.predict import make_prediction, load_model_artifacts, load_longformer_model, artifacts
from app.knowledge_base import KnowledgeBase
from app.chatbot import MovieChatbot

# Create FastAPI app
app = FastAPI(
    title="Movie Success Predictor API",
    description="Predict movie success and get AI recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global instances
kb = KnowledgeBase()
chatbot = None


@app.on_event("startup")
def startup_event():
    """Load all models and systems on startup"""
    global chatbot

    print("Starting Movie Success Predictor API...")

    # Load prediction model
    print("\nLoading prediction model artifacts...")
    load_model_artifacts()
    load_longformer_model()

    # Load knowledge base
    print("\nLoading knowledge base...")
    kb.load(
        tokenizer=artifacts["longformer_tokenizer"],
        model=artifacts["longformer_model"],
        device=artifacts["device"]
    )

    # Initialize chatbot
    print("\nInitializing chatbot...")
    chatbot = MovieChatbot(kb)

    print("\nAll systems loaded successfully!")
    print("=" * 60)


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def read_root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", tags=["Health Check"])
def health_check():
    """Check if API is running"""
    return {
        "status": "ok",
        "message": "Movie Success Predictor API is running!",
        "knowledge_base_loaded": kb.df is not None,
        "movies_count": len(kb.df) if kb.df is not None else 0
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_movie_success(payload: MovieInput):
    """Predict whether a movie will be successful."""
    try:
        prediction_result = make_prediction(payload)
        return prediction_result
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
def chat_with_bot(payload: ChatMessage):
    """Chat with the AI assistant about successful movies."""
    try:
        message = payload.message
        movie_data = payload.movie_data

        response_text, similar_movies = chatbot.generate_response(message, movie_data)

        similar_movies_list = None
        if similar_movies:
            similar_movies_list = [
                SimilarMovie(
                    title=m['title'],
                    genre=m['genre'],
                    budget=m['budget'],
                    duration_minutes=m['duration_minutes'],
                    first_actor_avg=m['first_actor_avg'],
                    second_actor_avg=m['second_actor_avg'],
                    director_avg=m['director_avg'],
                    mpa=m['mpa'],
                    plot_summary=m['plot'][:500],
                    distance=m['distance']
                )
                for m in similar_movies
            ]

        return ChatResponse(
            response=response_text,
            similar_movies=similar_movies_list,
            success=True
        )

    except Exception as e:
        return ChatResponse(
            response="I encountered an error processing your request.",
            success=False,
            error=str(e)
        )
