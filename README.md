# Movie Project New

A data-driven movie recommendation + analysis app using a FAISS knowledge base and transformer embeddings.

## Prerequisites
- Windows 10/11
- Python 3.10+ installed and on PATH
- Git (optional)
- PowerShell / Command Prompt / VS Code terminal

## Setup

1. Open project folder
   - cd "c:\Users\Dell\Movie Project New"

2. Create a virtual environment (recommended name: `myt`)
   - PowerShell:
     ```powershell
     python -m venv myt
     .\myt\Scripts\Activate.ps1
     ```
     If activation is blocked:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```
   - CMD:
     ```cmd
     python -m venv myt
     .\myt\Scripts\activate
     ```

3. Install dependencies
   - If you have requirements.txt:
     ```powershell
     pip install -r requirements.txt
     ```
   - Otherwise install minimal packages:
     ```powershell
     pip install pandas numpy faiss-cpu torch transformers groq sentence-transformers
     ```

4. Configure environment variables
   - Set your GROQ API key (PowerShell):
     ```powershell
     $env:GROQ_API_KEY="your_api_key_here"
     ```
   - CMD:
     ```cmd
     set GROQ_API_KEY=your_api_key_here
     ```

5. Ensure data file path is correct
   - Default path used in the code:  
     `C:\Users\Dell\Movie project\DATA\successful_movies_embeddings.xlsx`
   - If your file is stored elsewhere, update `DATA_FILE_PATH` in `app/knowledge_base.py` to point to the correct file.

## Run the app / website

- If you have a `main.py` script in the project root:
  ```powershell
  python main.py
  ```

- If the project exposes a web app (Flask/FastAPI), launch it with the appropriate command (example for FastAPI/uvicorn):
  ```powershell
  uvicorn main:app --reload --port 8000
  ```

- When running from VS Code, ensure the Python interpreter is set to the `myt` venv (`Ctrl+Shift+P` → "Python: Select Interpreter").

## Troubleshooting

- FAISS on Windows: prefer `faiss-cpu` via pip; if it fails try using conda.
- Torch installation: use the recommended wheel from https://pytorch.org/get-started/locally/ for your CUDA setup.
- FileNotFoundError for Excel: check path spelling and spaces — update `DATA_FILE_PATH`.
- Large model downloads: ensure you have a stable internet connection or use local model cache.

## Quick checks
- Verify venv: `python -V`
- Verify packages: `pip list`
- Deactivate venv: `deactivate`

