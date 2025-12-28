# Pre-Release Movie Success Prediction & Recommendation System

**A data-driven machine learning application that forecasts film performance and provides AI-powered recommendations using FAISS knowledge bases and Transformer embeddings.**

> **Academic Context** > This project was developed as the **Final Year Research Project** for the BSc (Hons) in Data Science and Business Analytics at **General Sir John Kotelawala Defence University**.  
> **Supervisor:** Dr. D.U. Vidanagama

---

## 📖 Project Overview

The global film industry is a high-stakes environment where multi-million dollar investments are often made based on intuition rather than data. Commercial success is notoriously unpredictable.

This project bridges the gap between storytelling and data science. It is a system designed to predict a film’s **IMDb rating category (Successful vs. Unsuccessful)** *before* the movie is released. Unlike many existing models that suffer from data leakage (using post-release data like box office gross), this system utilizes a **strictly pre-release** dataset.

It combines structured metadata (cast, budget, genre) with unstructured narrative analysis (plot embeddings via Longformer) to provide evidence-based decision support for producers and investors.

## 🎯 Aim & Objectives

**Aim:** To develop a robust, methodologically sound predictive framework for forecasting film success and providing content-based recommendations.

**Key Objectives:**
1.  **Strict Temporal Validity:** Eliminate data leakage by using only information available prior to a film's release.
2.  **Dynamic Feature Engineering:** Create time-sensitive historical performance metrics for actors and directors (calculating their reputation *at the time* of the specific movie release).
3.  **Hybrid Modeling:** Integrate metadata with deep narrative analysis using NLP (Longformer).
4.  **Decision Support:** Deploy a web-based recommendation engine.

## 💡 Novelty

* **Time-Aware Talent Metrics:** Instead of using a static career average for an actor, this model calculates their average rating dynamically based on their filmography *prior* to the target movie year.
* **Narrative Embeddings:** Utilizes the **Longformer** model (handling up to 4096 tokens) to generate embeddings for full plot summaries, capturing semantic nuances missed by standard BERT models.
* **RAG Recommendation Engine:** Integrates a FAISS vector database with LLMs (Groq) to provide actionable advice based on similar historical movies.

## 📊 Models & Results

To achieve optimal performance, four distinct model architectures were built and evaluated:

### 1. The Models Tested
* **K-Nearest Neighbors (KNN):** Baseline distance-based classifier.
* **Support Vector Machine (SVM):** Kernel-based classifier for high-dimensional data.
* **Random Forest:** Bagging ensemble method.
* **Regularized Stacking Classifier (Final Model):** A sophisticated ensemble combining **XGBoost** (precision-focused) and **LightGBM** (recall-focused) with a Logistic Regression meta-learner.

### 2. Performance Outcomes
The **Regularized Stacking Classifier** was the best performing model.

| Metric | Result | Note |
| :--- | :--- | :--- |
| **Accuracy** | **77%** | Outperformed all baselines |
| **Weighted F1-Score** | **0.77** | Balanced performance |
| **Success Recall** | **0.80** | Highly effective at identifying potential hits |

---

## ⚙️ Technical Setup

### Prerequisites
- Windows 10/11
- Python 3.10+ installed and on PATH
- Git (optional)
- PowerShell / Command Prompt / VS Code terminal

### Installation Steps

1.  **Clone/Navigate to project**
    ```powershell
    cd "path\to\Movie Project New"
    ```

2.  **Create a virtual environment**
    *PowerShell:*
    ```powershell
    python -m venv myt
    .\myt\Scripts\Activate.ps1
    ```
    *(If activation is blocked, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

    *CMD:*
    ```cmd
    python -m venv myt
    .\myt\Scripts\activate
    ```

3.  **Install dependencies**
    If you have a `requirements.txt`:
    ```powershell
    pip install -r requirements.txt
    ```
    Otherwise, install the core packages:
    ```powershell
    pip install pandas numpy faiss-cpu torch transformers groq sentence-transformers xgboost lightgbm scikit-learn fastapi uvicorn
    ```

4.  **Configure Environment Variables**
    Set your GROQ API key (required for the AI Chat/Recommendation feature):
    ```powershell
    $env:GROQ_API_KEY="your_api_key_here"
    ```

5.  **Data Path Configuration**
    Ensure the knowledge base file is correctly located.
    * Default path in code: `C:\Users\Dell\Movie project\DATA\successful_movies_embeddings.xlsx`
    * *Note: Please update `DATA_FILE_PATH` in `app/knowledge_base.py` to match your local directory structure.*

### Running the Application

**Option 1: Command Line Script**
```powershell
python main.py
```

**Option 2: Web API (FastAPI) Launch the backend server**
```powershell
uvicorn main:app --reload --port 8000
```

### Technology Stack

* **Language:** Python 3.10
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
* **Deep Learning / NLP:** PyTorch, Hugging Face Transformers (Longformer)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **LLM Integration:** Groq API
* **Backend:** FastAPI

### Trouble Shooting

* **FAISS on Windows:** If pip install faiss-cpu fails, try using conda install -c pytorch faiss-cpu or download a pre-built wheel.
* **Torch Installation:** If you have a dedicated GPU, install the specific CUDA version of PyTorch from pytorch.org.
* **Model Downloads:** The first run will download the allenai/longformer-base-4096 model (~700MB). Ensure you have a stable internet connection.

### Acknowledgments

**Special thanks to Dr. DU Vidanagama for supervision and guidance throughout this research project.**
